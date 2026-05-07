"""
SearchRescueAgent — LLM-driven rescue agent with MindForge-style cognitive loop.

Extends LLMAgentBase which handles all infrastructure (navigation, carry
retry, rendezvous, action validation, task injection).

This class implements a multi-stage async pipeline:
    COORDINATION → PLANNING (critic+plan) → REASONING → EXECUTE

COORDINATION fires at simulation start and after each rescue: a rotating
coordinator agent assigns one high-level task to every agent, stored in
GLOBAL_PLAN_ASSIGNMENTS in SharedMemory.

Each LLM stage is non-blocking: submit a call, return Idle, poll next tick.
Stage outputs flow forward via _pipeline_context.
"""

import ast
import json
import logging
import re
from collections import deque
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from helpers.logic_helpers import _chebyshev_distance
from helpers.navigation_helpers import apply_navigation
from matrx.agents.agent_utils.state import State
from matrx.messages.message import Message

from agents1.async_model_prompting import get_llm_result, _strip_thinking
from agents1.capabilities import DISCOVERY_NOTE, get_capability_prompt, get_game_rules
from agents1.llm_agent_base import LLMAgentBase, SM_TASK_ASSIGNMENTS
from agents1.modules.area_tracker import AreaExplorationTracker
from agents1.modules.coordination_module import (
    CoordinationModule,
    COORDINATION_TIMEOUT_TICKS,
    COORDINATOR_LLM_TIMEOUT_TICKS,
    COORDINATION_REGISTRATION_TICKS,
    HISTORY_WINDOW,
    MSG_TYPE_COORD_ASSIGNMENT,
)
from agents1.modules.execution_module import execute_action
from agents1.modules.reasoning_module import (
    ReasoningIO,
    ReasoningReAct,
    REASONING_STRATEGY_REGISTRY,
    FollowupRequest,
    ActionCommit,
)
from agents1.tool_registry import REASONING_STRATEGIES, build_tool_schemas
from memory.episode_memory import EpisodeMemory
from memory.shared_memory import SharedMemory
from worlds1.environment_info import EnvironmentInformation

logger = logging.getLogger('SearchRescueAgent')

# ── Role system ────────────────────────────────────────────────────────────────
ROLE_CLAIM_MSG_TYPE = 'role_claim'  # message_type used for role announcements


# ── Module-level helpers ────────────────────────────────────────────────────

def _extract_action_json(text: str) -> Optional[Dict]:
    """Try to recover a JSON action dict from free-form LLM text.

    Useful when Ollama returns a plain-text response instead of a structured
    tool call.  Tries three strategies in order:

    1. Fenced ``json ... `` block
    2. First ``{...}`` span (greedy)
    3. ``ast.literal_eval`` on the same span (handles single-quoted dicts)
    """
    if not text:
        return None
    # 1. Fenced block
    m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    # 2. First {...} span
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            try:
                return ast.literal_eval(m.group(0))
            except (ValueError, SyntaxError):
                pass
    return None


class PipelineStage(Enum):
    IDLE = 'idle'
    COORDINATION = 'coordination'
    PLANNING = 'planning'
    REASONING = 'reasoning'
    EXECUTE = 'execute'
    COMM_DISPATCH = 'comm_dispatch'


class SearchRescueAgent(LLMAgentBase):
    """Rescue agent with multi-stage cognitive pipeline.

    Pipeline per cycle:
        COORDINATION → PLANNING (critic+plan combined) → REASONING → EXECUTE

    COORDINATION fires at start and after each rescue: a rotating coordinator
    assigns high-level tasks; other agents wait for their assignment.
    PLANNING evaluates the last action (critic) and decides the next atomic
    task, returning JSON {reasoning, success, critique, next_task}.
    """

    def __init__(
        self,
        slowdown: int,
        condition: str,
        name: str,
        folder: str,
        llm_model: str = 'ollama/llama3',
        strategy: str = 'react',
        include_human: bool = True,
        shared_memory: Optional[SharedMemory] = None,
        planning_mode: str = 'simple',
        planning_strategy: str = 'io',
        replanning_policy: str = 'every_turn',
        api_base: Optional[str] = None,
        capabilities: Optional[Dict] = None,
        capability_knowledge: str = 'informed',
        comm_strategy: str = 'always_respond',
        env_info: Optional[EnvironmentInformation] = None,
        use_planner: bool = True,
        initial_role: Optional[str] = None,
    ) -> None:
        super().__init__(
            slowdown=slowdown,
            condition=condition,
            name=name,
            folder=folder,
            llm_model=llm_model,
            include_human=include_human,
            shared_memory=shared_memory,
            planning_mode=planning_mode,
            planning_strategy=planning_strategy,
            api_base=api_base,
            capabilities=capabilities,
            capability_knowledge=capability_knowledge,
            comm_strategy=comm_strategy,
            env_info=env_info,
            use_planner=use_planner,
        )
        # Resolve reasoning strategy via registry (falls back to ReAct).
        self._strategy = strategy if strategy in REASONING_STRATEGY_REGISTRY else 'react'
        self._replanning_policy = (
            replanning_policy if replanning_policy in ('every_turn', 'critic_gated')
            else 'every_turn'
        )
        self.area_tracker = AreaExplorationTracker(self.env_info.get_area_cells())
        self.tools_by_name, self.tool_schemas = build_tool_schemas()

        reasoning_cls = REASONING_STRATEGY_REGISTRY.get(self._strategy, ReasoningReAct)
        self.reasoning = reasoning_cls()

        # Pipeline state
        self._pipeline_stage: PipelineStage = PipelineStage.IDLE
        self._pipeline_context: Dict[str, Any] = {}
        self._is_first_cycle: bool = True

        # Cursor into CommunicationModule.all_messages_raw so we only save new
        # messages each tick without duplicating entries already in memory.
        self._comm_msg_cursor: int = 0
        # Separate cursor advanced only when a new episode opens, so each
        # episode captures exactly the messages received since the last episode.
        self._episode_msg_cursor: int = 0

        # Two-level task hierarchy:
        #   _high_level_task  — coordinator-assigned mission (e.g. "Search area 2")
        #   _current_task     — atomic next step chosen by the planner
        self._high_level_task: str = ''

        # Coordination stage state
        self.coordination = CoordinationModule(
            agent_id=self.agent_id,
            llm_model=llm_model,
            api_base=api_base,
        )
        self._is_first_coord_cycle: bool = True
        self._coord_round_id: Optional[int] = None
        self._coord_trigger_tick: Optional[int] = None
        self._coord_is_coordinator: bool = False
        self._coord_llm_submitted: bool = False

        # Role system — seed from hardcoded initial_role if provided
        self._current_role: str = initial_role or ''
        self._team_roles: Dict[str, str] = {}  # {agent_id: role} from incoming messages

        # Anti-loop detection: track last 3 executed actions
        self._recent_actions: deque = deque(maxlen=3)

        # Structured episode memory
        self.episode_memory = EpisodeMemory()
        # Communication dedup: skip sending identical messages back-to-back
        self._recent_sent_msgs: deque = deque(maxlen=3)
        # Critic dedup: avoid injecting identical failure feedback twice in a row
        self._last_critic_failure_critique: str = ''

        print(
            f'[SearchRescueAgent] Created '
            f'(model={llm_model}, strategy={self._strategy}, '
            f'planning={planning_mode}, plan_strategy={planning_strategy}, '
            f'replan={self._replanning_policy}, caps={capabilities})'
        )

        # Print area tracker initialisation summary so we can verify
        # the correct areas and cell counts were loaded for this preset.
        _area_cells = self.env_info.get_area_cells()
        print(
            f'[{name}] AreaTracker init: {len(_area_cells)} areas — '
            + ', '.join(
                f'{k}({len(v)} cells)'
                for k, v in sorted(_area_cells.items())
            )
        )

    # ── Task injection override ─────────────────────────────────────────

    def set_current_task(self, task: str) -> None:
        super().set_current_task(task)
        self._pipeline_stage = PipelineStage.IDLE
        self._pipeline_context = {}
        self._is_first_cycle = True
        self._comm_msg_cursor = 0
        self._recent_actions.clear()
        self._last_critic_failure_critique = ''

    # ── Perception ──────────────────────────────────────────────────────

    def update_knowledge(self, filtered_state: State) -> None:
        super().update_knowledge(filtered_state)

        # Parse role_claim messages from peers and update local team-role map.
        # MATRX may deliver content as a JSON string; handle both dict and str forms.
        for msg in self.received_messages:
            content = msg.content if hasattr(msg, 'content') else {}
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    content = {}
            if isinstance(content, dict) and content.get('message_type') == ROLE_CLAIM_MSG_TYPE:
                sender = getattr(msg, 'from_id', '')
                role   = content.get('role', '')
                if sender and sender != self.agent_id and role:
                    self._team_roles[sender] = role

        # Register this agent so the coordinator can build a complete team list
        # even before all agents are within perception range of each other.
        if self.shared_memory:
            self.shared_memory.add_to_set('registered_agents', self.agent_id)

        # Remove rescued victims from the global belief so agents don't re-target them.
        if self.shared_memory:
            rescued_ids = {v['victim_id'] for v in (self.shared_memory.retrieve('rescued_victims') or [])}
            if rescued_ids:
                self.WORLD_STATE_GLOBAL['victims'] = [
                    v for v in self.WORLD_STATE_GLOBAL.get('victims', [])
                    if v.get('object_id') not in rescued_ids
                ]

        # Normalize agent location to a strict (int, int) tuple.
        # MATRX may return a list or even float coords depending on the build;
        # floats would silently break Chebyshev matching against int frozensets.
        raw_loc = filtered_state[self.agent_id]['location']
        agent_loc = (int(raw_loc[0]), int(raw_loc[1]))

        vision_str = self._capabilities.get('vision', 'medium') if self._capabilities else 'medium'
        vision = {'low': 1, 'medium': 2, 'high': 3}.get(vision_str, 2)
        self.area_tracker.update(agent_loc, vision_radius=vision)

        # Save newly received messages to memory so all LLM stages can see them
        # even after CommunicationModule discards old ones via summarization.
        all_msgs = self.communication.all_messages_raw
        new_msgs = all_msgs[self._comm_msg_cursor:]
        for msg in new_msgs:
            self.memory.update('received_message', {
                'from': msg.get('from'),
                'type': msg.get('message_type'),
                'text': msg.get('text'),
                'tick': self._tick_count,
            })
        self._comm_msg_cursor = len(all_msgs)

        # Intercept coord_assignment messages when the agent is NOT in the
        # COORDINATION stage.  Handles the race where the coordinator finishes
        # while this agent is already in PLANNING/REASONING/EXECUTE, so
        # coordination_stage() never runs and would silently miss the message.
        if self._pipeline_stage != PipelineStage.COORDINATION:
            coord_task = self.coordination.ingest_assignment_message(self.received_messages)
            if coord_task:
                prev_stage = self._pipeline_stage.value
                coord_state = self.shared_memory.retrieve('coordination_state') if self.shared_memory else {}
                coordinator_id = (coord_state or {}).get('coordinator', 'unknown')
                self._high_level_task = coord_task
                self._current_task = ''
                self._pending_future = None  # cancel any stale LLM call
                self._coord_round_id = None
                self._pipeline_stage = PipelineStage.PLANNING
                self._write_coordination_memory(
                    assignments={self.agent_id: coord_task},
                    coordinator_id=coordinator_id,
                )
                if self.shared_memory:
                    with self.shared_memory.lock:
                        state = self.shared_memory.storage.get('coordination_state') or {}
                        acked = list(state.get('acknowledged_by', []))
                        if self.agent_id not in acked:
                            acked.append(self.agent_id)
                            state['acknowledged_by'] = acked
                            self.shared_memory.storage['coordination_state'] = state
                print(f'[{self.agent_id}] Coord assignment received outside COORDINATION stage '
                      f'(was {prev_stage}): {coord_task!r}')

        # ── Periodic data-verification print (every 50 ticks) ─────────────
        if self._tick_count % 50 == 1:
            n_victims_near = len(self.WORLD_STATE.get('victims', []))
            n_obstacles_near = len(self.WORLD_STATE.get('obstacles', []))
            summaries = self.area_tracker.get_all_summaries()
            avg_coverage = (
                sum(s['coverage'] for s in summaries) / len(summaries)
                if summaries else 0.0
            )
            n_global_victims = len(self.WORLD_STATE_GLOBAL.get('victims', []))
            n_global_obstacles = len(self.WORLD_STATE_GLOBAL.get('obstacles', []))
            print(
                f'[{self.agent_id}] tick={self._tick_count} '
                f'loc={agent_loc} vision={vision} '
                f'nearby(victims={n_victims_near} obstacles={n_obstacles_near}) '
                f'global(victims={n_global_victims} obstacles={n_global_obstacles}) '
                f'area_coverage={avg_coverage:.0%}'
            )

    def decide_on_actions(self, filtered_state: State) -> Tuple[Optional[str], Dict]:
        self.update_knowledge(filtered_state)

        # if not self._current_task:
        #     if self._use_planner and not self._received_planner_task:
        #         # Waiting for planner to assign a task
        #         return self._idle()
        #     elif not self._use_planner:
        #         # Self-assign a default task (no planner)
        #         self.set_current_task('Explore all areas, find and rescue victims')
        #     else:
        #         return self._idle()

        # Infrastructure: carry retry, navigation
        action = self._run_infra(filtered_state)
        if action is not None:
            return action

        # Poll pending LLM future
        if self._pending_future is not None:
            try:
                result = get_llm_result(self._pending_future)
            except Exception as exc:
                logger.warning('[%s] LLM future raised: %s', self.agent_id, exc)
                self._pending_future = None
                self._pipeline_stage = PipelineStage.IDLE
                return self._idle()
            if result is None:
                return self._idle(reason='llm_wait')
            self._pending_future = None
            return self._on_llm_result(result)

        # Advance pipeline
        return self._advance_pipeline()

    def _open_episode_cycle(self) -> None:
        """Close the previous episode (capturing MATRX outcome) and open a new one."""
        # Close whatever was open — previous_action_result is now set by MATRX
        prev = self.episode_memory.get_open_episode()
        if prev is not None and not prev.closed:
            succeeded = None
            reason = None
            if self.previous_action_result is not None:
                succeeded = bool(self.previous_action_result.succeeded)
                reason = str(getattr(self.previous_action_result, 'result', ''))
            self.episode_memory.close_episode(
                tick=self._tick_count,
                succeeded=succeeded,
                reason=reason,
            )

        # Open a fresh episode for this cycle
        self.episode_memory.open_episode(
            tick=self._tick_count,
            agent_id=self.agent_id,
            task=self._current_task or '',
            role=self._current_role or 'unassigned',
        )

        # Snapshot peer messages received since the last episode opened.
        # Uses _episode_msg_cursor (not _comm_msg_cursor) because _comm_msg_cursor
        # is advanced every tick before _advance_pipeline is called.
        all_msgs = list(self.communication.all_messages_raw)
        new_msgs = all_msgs[self._episode_msg_cursor:]
        self._episode_msg_cursor = len(all_msgs)
        self.episode_memory.set_received_messages([
            {
                'from': m.get('from'),
                'type': m.get('message_type'),
                'text': m.get('text'),
            }
            for m in new_msgs
        ])

    def _advance_pipeline(self) -> Tuple[Optional[str], Dict]:
        if self._pipeline_stage == PipelineStage.IDLE:
            self._get_or_assign_role()  # assign role on first cycle only
            self._is_first_cycle = False
            self._pipeline_context = {}
            self._open_episode_cycle()
            self._pipeline_stage = PipelineStage.COORDINATION

        if self._pipeline_stage == PipelineStage.COORDINATION:
            return self.coordination_stage()
        if self._pipeline_stage == PipelineStage.PLANNING:
            return self.plan()
        if self._pipeline_stage == PipelineStage.REASONING:
            return self.reason()
        if self._pipeline_stage == PipelineStage.EXECUTE:
            return self.execute()
        if self._pipeline_stage == PipelineStage.COMM_DISPATCH:
            return self.communicate()

        return self._idle()

    def _on_llm_result(self, result) -> Tuple[Optional[str], Dict]:
        if self.metrics:
            self.metrics.record_llm_call_end()
        if self._pipeline_stage == PipelineStage.COORDINATION:
            return self._handle_coordination_result(result)
        if self._pipeline_stage == PipelineStage.PLANNING:
            return self._handle_planning_result(result)
        if self._pipeline_stage == PipelineStage.REASONING:
            return self._handle_reasoning_result(result)
        return self._idle()

    def _get_rescued_victims(self):
        """Derive rescued victims from SharedMemory."""
        if self.shared_memory:
            rescued = self.shared_memory.retrieve('rescued_victims')
            if rescued:
                return rescued
        return None

    # ── Help-request response tracking ──────────────────────────────────

    _HELP_REPLIES_KEY_PREFIX = 'help_replies::'
    HELP_RESPONSE_TIMEOUT_TICKS = 500

    def _help_replies_key(self, requester_id: str) -> str:
        return f'{self._HELP_REPLIES_KEY_PREFIX}{requester_id}'

    def _register_help_request_sent(self, expected_responders: int) -> None:
        """Create the shared-memory counter when this agent sends an ask_help."""
        if not self.shared_memory:
            return
        self.shared_memory.update(self._help_replies_key(self.agent_id), {
            'tick': self._tick_count,
            'expected': max(expected_responders, 0),
            'replies': {},
        })

    def _record_help_reply(self, requester: str, reply: str) -> None:
        """Record a yes/no reply on the requester's counter."""
        if not self.shared_memory:
            return
        key = self._help_replies_key(requester)
        entry = self.shared_memory.retrieve(key)
        if not entry:
            return
        replies = dict(entry.get('replies', {}))
        replies[self.agent_id] = reply
        self.shared_memory.update(key, {**entry, 'replies': replies})

    def _check_help_request_outcome(self) -> Optional[str]:
        """Return an abandon directive string if this agent's ask_help is resolved.

        Resolution conditions:
          1. All expected teammates replied and none said "yes".
          2. At least HELP_RESPONSE_TIMEOUT_TICKS ticks elapsed since the request
             was sent and no teammate has accepted so far.
        A single "yes" cancels the abandon (the accepter is en route).
        Clears the counter after producing a directive (or on accepted yes).
        """
        if not self.shared_memory:
            return None
        key = self._help_replies_key(self.agent_id)
        entry = self.shared_memory.retrieve(key)
        if not entry:
            return None

        replies = entry.get('replies', {}) or {}
        expected = int(entry.get('expected', 0))
        elapsed = self._tick_count - int(entry.get('tick', self._tick_count))
        timed_out = elapsed >= self.HELP_RESPONSE_TIMEOUT_TICKS
        all_replied = len(replies) >= expected and expected > 0

        any_yes = any(r == 'yes' for r in replies.values())
        if any_yes:
            # A teammate accepted — clear the counter, no abandon needed.
            self.shared_memory.update(key, None)
            return None

        if not (all_replied or timed_out):
            return None

        if replies and all(r == 'no' for r in replies.values()):
            reason = 'All teammates declined your help request.'
        elif timed_out and not replies:
            reason = (
                f'No teammate responded to your help request within '
                f'{self.HELP_RESPONSE_TIMEOUT_TICKS} ticks.'
            )
        elif timed_out:
            reason = (
                f'Only {len(replies)}/{expected} teammates responded '
                f'within {self.HELP_RESPONSE_TIMEOUT_TICKS} ticks, and none accepted.'
            )
        else:
            # Edge case: expected==0 and no replies — nothing to abandon.
            self.shared_memory.update(key, None)
            return None

        self.shared_memory.update(key, None)
        return f'{reason} You MUST abandon this task and choose a different objective.'

    # ── Prompt-input visibility ─────────────────────────────────────────

    def _log_stage_inputs(self, stage: str, inputs: Dict[str, Any]) -> None:
        """Print a concise summary of the data being sent to an LLM stage.

        Each field is rendered on one line showing its type and a short preview
        so it is easy to spot missing, empty, or unexpectedly large values at a
        glance without wading through full prompt text.
        """
        def _fmt(v: Any) -> str:
            if v is None:
                return 'None'
            if isinstance(v, bool):
                return str(v)
            if isinstance(v, str):
                trimmed = v.replace('\n', ' ')
                preview = trimmed[:80] + ('…' if len(trimmed) > 80 else '')
                return f'str({len(v)}c) "{preview}"'
            if isinstance(v, list):
                if not v:
                    return 'list(0)'
                first = str(v[0])[:60].replace('\n', ' ')
                return f'list({len(v)}) [{first}{"…" if len(str(v[0])) > 60 else ""}]'
            if isinstance(v, dict):
                keys = ', '.join(str(k) for k in list(v.keys())[:6])
                suffix = ', …' if len(v) > 6 else ''
                return f'dict({len(v)}) {{{keys}{suffix}}}'
            return repr(v)[:80]

        lines = [f'[{self.agent_id}] ── {stage} inputs (tick {self._tick_count}) ──']
        for key, val in inputs.items():
            lines.append(f'  {key:<22} {_fmt(val)}')
        print('\n'.join(lines))

    # ── Role system ─────────────────────────────────────────────────────

    def _pick_role(self) -> str:
        """Choose the most-needed role based on team state and world situation.

        Uses only team-need logic — capability affinity is a soft prompt hint,
        not a hard code constraint, so agents can flex when the team needs it.
        """
        taken = set(self._team_roles.values())
        unrescued = self.WORLD_STATE_GLOBAL.get('victims', [])
        obstacles  = self.WORLD_STATE_GLOBAL.get('obstacles', [])
        unexplored = [s for s in self.area_tracker.get_all_summaries()
                      if s['status'] != 'complete']

        if 'rescuer' not in taken and unrescued:
            return 'rescuer'
        if 'scout' not in taken and unexplored:
            return 'scout'
        if 'heavy' not in taken and obstacles:
            return 'heavy'
        return 'rescuer'  # safe default

    def _get_or_assign_role(self) -> str:
        """Assign agent's role on first cycle; keep it for the rest of execution.

        Sends a direct role_claim broadcast on first assignment — no LLM, no extra tick cost.
        """
        if self._current_role:
            return self._current_role

        new_role = self._pick_role()
        self._current_role = new_role
        self._role_assigned_tick = self._tick_count
        # Announce to teammates — pure send_message, no LLM
        self.send_message(Message(
            content={'message_type': ROLE_CLAIM_MSG_TYPE, 'role': new_role},
            from_id=self.agent_id,
            to_id=None,  # broadcast
        ))
        print(f'[{self.agent_id}] Role → {new_role} (tick={self._tick_count})')
        return self._current_role

    def _get_role_hint(self) -> str:
        """Return a soft capability-based hint for the LLM prompt (not enforced in code)."""
        caps = self._capabilities or {}
        if caps.get('vision') == 'high':
            return 'Your high vision makes you well-suited for the scout role.'
        if caps.get('strength') == 'high':
            return 'Your high strength makes you well-suited for the heavy role.'
        if caps.get('medical') == 'high':
            return 'Your high medical skill makes you well-suited for the rescuer role.'
        return 'Adapt your role to what the team needs most.'

    _ROLE_DESCRIPTIONS: Dict[str, str] = {
        'scout':        'Your assigned role is SCOUT. Prioritise exploring unmapped areas and reporting discoveries to your team.',
        'medic':        'Your assigned role is MEDIC. Prioritise locating and carrying injured victims to the drop zone.',
        'heavy_lifter': 'Your assigned role is HEAVY LIFTER. Prioritise removing large obstacles (rocks, trees) to open paths for teammates.',
        'rescuer':      'Your assigned role is RESCUER. Prioritise picking up victims and delivering them to the drop zone.',
        'generalist':   'Your assigned role is GENERALIST. Balance exploration, rescue, and obstacle removal according to what the team needs most.',
    }

    def _get_role_prompt(self) -> str:
        """Return the role description for injection into the LLM system prompt."""
        role = self._current_role or 'generalist'
        base = self._ROLE_DESCRIPTIONS.get(
            role, f'Your assigned role is {role.upper()}.'
        )
        return (
            base + ' If you judge that a different role would better serve the team '
            'given the current situation, you may adapt your behaviour accordingly.'
        )

    # ── Shared state builders ───────────────────────────────────────────

    def _build_canonical_state(self) -> Dict[str, Any]:
        """Return a consistent base state dict shared by all pipeline stages."""
        agent_info = self.WORLD_STATE.get('agent', {})
        return {
            'position': agent_info.get('location'),
            'carrying': agent_info.get('carrying'),
            'current_task': self._current_task,
            'last_action': self._recent_actions[-1] if self._recent_actions else {},
            'critic_feedback': self._pipeline_context.get('critic_result'),
            'tick': self._tick_count,
        }

    def _build_common_context(self) -> Dict[str, Any]:
        """Return game rules and capability text for injection into every LLM stage.

        In 'informed' mode the agent receives its full capability profile.
        In 'discovery' mode it receives a short note to learn from failures.
        """
        if self._capability_knowledge == 'informed':
            cap_text = get_capability_prompt(self._capabilities)
        else:
            cap_text = DISCOVERY_NOTE
        return {
            'game_rules': get_game_rules(
                drop_zone=self.env_info.drop_zone,
                capabilities=self._capabilities if self._capability_knowledge == 'informed' else None,
            ),
            'agent_capabilities': cap_text,
            'current_role': self._current_role or 'unassigned',
            'role_hint': self._get_role_hint(),
            'role_prompt': self._get_role_prompt(),
            'team_roles': self._team_roles,
        }

    # ── COORDINATION stage ───────────────────────────────────────────────

    def _all_agent_ids(self) -> list:
        """Return sorted list of all registered agent IDs from SharedMemory.

        Falls back to local WORLD_STATE teammates if SharedMemory is unavailable.
        Using the registry instead of WORLD_STATE avoids split-brain coordinator
        elections when agents start outside each other's vision range.
        """
        if self.shared_memory:
            registered = self.shared_memory.retrieve('registered_agents') or []
            if registered:
                return sorted(set(registered))
        teammate_ids = [t['object_id'] for t in self.WORLD_STATE.get('teammates', [])]
        return sorted([self.agent_id] + teammate_ids)

    def coordination_stage(self) -> Tuple[Optional[str], Dict]:
        """Per-tick handler for the COORDINATION stage.

        One agent per round (round-robin) submits an LLM call to assign
        high-level tasks to all agents.  Others idle until they receive their
        assignment via MATRX message or SharedMemory, then proceed to PLANNING.
        """
        if self.shared_memory is None:
            self._pipeline_stage = PipelineStage.PLANNING
            return self._advance_pipeline()

        rescued_victims = self.shared_memory.retrieve('rescued_victims') or []
        coord_state = self.shared_memory.retrieve('coordination_state')

        should_trigger, trigger_reason = self.coordination.should_trigger(
            rescued_victims=rescued_victims,
            coord_state=coord_state,
            is_first_coord_cycle=self._is_first_coord_cycle,
            tick=self._tick_count,
        )

        if should_trigger:
            self._is_first_coord_cycle = False
            # Always update count so losing agents don't re-trigger for the same rescue
            self.coordination._last_seen_rescued_count = len(rescued_victims)

            agent_ids = self._all_agent_ids()
            # Determine the round_id for this new round
            current_counter = self.shared_memory.retrieve('coordination_round_counter') or -1
            next_round_id = current_counter + 1
            coordinator_id = CoordinationModule.elect_coordinator(next_round_id, agent_ids)

            won = self.shared_memory.try_start_coordination(
                round_id=next_round_id,
                coordinator_id=coordinator_id,
                trigger=trigger_reason,
                trigger_tick=self._tick_count,
                rescued_count=len(rescued_victims),
            )

            if won:
                self._coord_round_id = next_round_id
                self._coord_trigger_tick = self._tick_count
                self._coord_is_coordinator = (coordinator_id == self.agent_id)
                self._coord_llm_submitted = False
                print(
                    f'[{self.agent_id}] Coordination round {next_round_id} started '
                    f'(coordinator={coordinator_id}, trigger={trigger_reason})'
                )
            else:
                # Another agent started this round — discover it from SharedMemory
                coord_state = self.shared_memory.retrieve('coordination_state') or {}
                self._coord_round_id = coord_state.get('round_id', next_round_id)
                self._coord_trigger_tick = coord_state.get('trigger_tick', self._tick_count)
                self._coord_is_coordinator = (coord_state.get('coordinator') == self.agent_id)
                self._coord_llm_submitted = self._coord_is_coordinator  # don't resubmit

            return self._idle()

        # No new trigger — check if we're currently in an active round
        if self._coord_round_id is None:
            status = (coord_state or {}).get('status')
            if status == 'in_progress':
                # Another agent started the round before our should_trigger check — join it.
                self._is_first_coord_cycle = False
                self._coord_round_id = coord_state.get('round_id')
                self._coord_trigger_tick = coord_state.get('trigger_tick', self._tick_count)
                self._coord_is_coordinator = (coord_state.get('coordinator') == self.agent_id)
                self._coord_llm_submitted = self._coord_is_coordinator
                return self._idle()
            elif status in ('complete', 'timed_out') and coord_state.get('assignments'):
                # Round completed before we could join — pick up assignment directly.
                self._is_first_coord_cycle = False
                self._coord_round_id = coord_state.get('round_id')
                self._coord_trigger_tick = coord_state.get('trigger_tick', self._tick_count)
                task = (coord_state.get('assignments') or {}).get(self.agent_id)
                return self._receive_coordination_assignment(task)
            # Genuinely no active round — proceed to planning.
            self._pipeline_stage = PipelineStage.PLANNING
            return self._advance_pipeline()

        # ── Active round handling ──────────────────────────────────────────
        elapsed = self._tick_count - (self._coord_trigger_tick or self._tick_count)

        if self._coord_is_coordinator:
            # Coordinator: submit LLM call once, then wait for _handle_coordination_result
            if not self._coord_llm_submitted:
                agent_ids = self._all_agent_ids()
                agent_roles = {**self._team_roles, self.agent_id: self._current_role or 'unknown'}
                agent_capabilities: Dict[str, Any] = {}
                for aid in agent_ids:
                    if aid == self.agent_id:
                        agent_capabilities[aid] = self._capabilities or {}
                history = (self.shared_memory.retrieve('GLOBAL_PLAN_ASSIGNMENTS') or [])[-HISTORY_WINDOW:]
                prompt = self.coordination.build_coordinator_prompt(
                    tick=self._tick_count,
                    trigger=coord_state.get('trigger', 'start') if coord_state else 'start',
                    agent_ids=agent_ids,
                    agent_roles=agent_roles,
                    agent_capabilities=agent_capabilities,
                    area_summaries=self.area_tracker.get_all_summaries(),
                    known_victims=self.WORLD_STATE_GLOBAL.get('victims', []),
                    rescued_victims=rescued_victims,
                    history=history,
                    own_recent_memory=self.memory.retrieve_all()[-10:],
                )
                self.call_llm(prompt)
                self._coord_llm_submitted = True
                print(f'[{self.agent_id}] Coordinator LLM submitted (round={self._coord_round_id})')

            # Coordinator LLM timeout fallback
            elif elapsed >= COORDINATOR_LLM_TIMEOUT_TICKS:
                logger.warning('[%s] Coordinator LLM timed out — using fallback assignments', self.agent_id)
                agent_ids = self._all_agent_ids()
                area_names = [s['name'] for s in self.area_tracker.get_all_summaries()]
                fallback = {
                    aid: f'Explore {area_names[i % len(area_names)]}'
                    for i, aid in enumerate(agent_ids)
                }
                return self._apply_coordination_result(fallback, timed_out=True)

            return self._idle()

        else:
            # Non-coordinator: check for assignment via message or SharedMemory
            task = self.coordination.ingest_assignment_message(self.received_messages)
            if task:
                return self._receive_coordination_assignment(task)

            complete, task = self.coordination.is_coordination_complete(
                coord_state=self.shared_memory.retrieve('coordination_state'),
                tick=self._tick_count,
                trigger_tick=self._coord_trigger_tick or self._tick_count,
            )
            if complete:
                return self._receive_coordination_assignment(task)

            return self._idle()

    def _receive_coordination_assignment(self, task: Optional[str]) -> Tuple[Optional[str], Dict]:
        """Non-coordinator: accept assigned task and move to PLANNING."""
        assigned = task or self._high_level_task or 'Explore unmapped areas'
        self._high_level_task = assigned
        self._current_task = ''  # reset so planner decomposes fresh

        # Write assignment to per-agent memory for planning context
        coord_state = self.shared_memory.retrieve('coordination_state') if self.shared_memory else {}
        coordinator_id = (coord_state or {}).get('coordinator', 'unknown')
        self._write_coordination_memory(
            assignments={self.agent_id: assigned},
            coordinator_id=coordinator_id,
        )

        # Acknowledge in SharedMemory
        if self.shared_memory:
            with self.shared_memory.lock:
                state = self.shared_memory.storage.get('coordination_state') or {}
                acked = list(state.get('acknowledged_by', []))
                if self.agent_id not in acked:
                    acked.append(self.agent_id)
                    state['acknowledged_by'] = acked
                    self.shared_memory.storage['coordination_state'] = state

        print(f'[{self.agent_id}] Received coordination assignment: {assigned!r}')
        self._coord_round_id = None
        self._pipeline_stage = PipelineStage.PLANNING
        return self._advance_pipeline()

    def _apply_coordination_result(
        self,
        assignments: Dict[str, str],
        timed_out: bool = False,
    ) -> Tuple[Optional[str], Dict]:
        """Coordinator: publish assignments, persist to SharedMemory, advance to PLANNING."""
        coord_state = self.shared_memory.retrieve('coordination_state') or {}
        trigger = coord_state.get('trigger', 'start')
        rescued_victims = self.shared_memory.retrieve('rescued_victims') or []

        # Build the PlanAssignmentRecord for GLOBAL_PLAN_ASSIGNMENTS
        record = {
            'round_id': self._coord_round_id,
            'formed_by': self.agent_id,
            'formed_at_tick': self._tick_count,
            'trigger': trigger,
            'rescued_count_at_formation': len(rescued_victims),
            'area_coverage_at_formation': self.area_tracker.get_all_summaries(),
            'known_victims_at_formation': list(self.WORLD_STATE_GLOBAL.get('victims', [])),
            'assignments': assignments,
            'agent_roles_at_formation': {**self._team_roles, self.agent_id: self._current_role or 'unknown'},
        }
        self.shared_memory.append_to_list('GLOBAL_PLAN_ASSIGNMENTS', record)

        # Update coordination_state to complete
        with self.shared_memory.lock:
            state = self.shared_memory.storage.get('coordination_state') or {}
            state['assignments'] = assignments
            state['status'] = 'timed_out' if timed_out else 'complete'
            state['completed_tick'] = self._tick_count
            self.shared_memory.storage['coordination_state'] = state

        # Send MATRX messages to each agent
        for target_id, task in assignments.items():
            self.send_message(Message(
                content={
                    'message_type': MSG_TYPE_COORD_ASSIGNMENT,
                    'round_id': self._coord_round_id,
                    'task': task,
                },
                from_id=self.agent_id,
                to_id=target_id,
            ))

        print(
            f'[{self.agent_id}] Coordination assignments sent (round={self._coord_round_id}): '
            + ', '.join(f'{k}={v!r}' for k, v in assignments.items())
        )

        # Apply own assignment
        own_task = assignments.get(self.agent_id, 'Explore unmapped areas')
        self._high_level_task = own_task
        self._current_task = ''  # reset for fresh planner decomposition

        self._write_coordination_memory(assignments=assignments, coordinator_id=self.agent_id)

        self._coord_round_id = None
        self._pipeline_stage = PipelineStage.PLANNING
        return self._advance_pipeline()

    def _handle_coordination_result(self, result) -> Tuple[Optional[str], Dict]:
        """LLM result handler for COORDINATION stage (coordinator only)."""
        text = _strip_thinking(getattr(result[0], 'content', '') or '') or ''
        agent_ids = self._all_agent_ids()
        assignments = self.coordination.parse_coordinator_response(text, agent_ids)

        if assignments is None:
            logger.warning('[%s] Coordinator parse failed — using fallback', self.agent_id)
            area_names = [s['name'] for s in self.area_tracker.get_all_summaries()]
            assignments = {
                aid: f'Explore {area_names[i % len(area_names)]}'
                for i, aid in enumerate(agent_ids)
            }

        return self._apply_coordination_result(assignments)

    def _write_coordination_memory(
        self,
        assignments: Dict[str, str],
        coordinator_id: str,
    ) -> None:
        """Write coordination assignment + history snapshot to per-agent BaseMemory."""
        self.memory.update('coordination_assignment', {
            'round_id': self._coord_round_id,
            'tick': self._tick_count,
            'coordinator': coordinator_id,
            'high_level_task': assignments.get(self.agent_id, ''),
            'all_assignments': assignments,
        })
        if self.shared_memory:
            global_plans = self.shared_memory.retrieve('GLOBAL_PLAN_ASSIGNMENTS') or []
            self.memory.update('global_plan_history_snapshot', {
                'tick': self._tick_count,
                'recent_rounds': global_plans[-HISTORY_WINDOW:],
            })

    # ── PLANNING stage (combined critic + plan) ─────────────────────────

    def plan(self) -> Tuple[Optional[str], Dict]:
        agent_info = self.WORLD_STATE.get('agent', {})
        common_ctx = self._build_common_context()

        # Enrich teammate list with known roles
        teammates_enriched = [
            {
                'id': t['object_id'],
                'location': [t.get('x'), t.get('y')],
                'role': self._team_roles.get(t['object_id'], 'unknown'),
            }
            for t in self.WORLD_STATE.get('teammates', [])
        ]

        # Local observation: strip walls (static), agent (already in context), and
        # teammates (already in context.teammates with roles) to avoid duplication.
        observation_clean = {
            k: v for k, v in self.WORLD_STATE.items()
            if k not in ('walls', 'agent', 'teammates')
        }

        # Get raw messages for planner (messages flow into planning, not reasoning)
        agent_busy = (self._nav_target is not None
                      or self._carry_autopilot is not None
                      or self._coop_carry_role is not None
                      or self._coop_remove_role is not None)
        raw_messages = self.communication.get_messages(limit=10, agent_busy=agent_busy)

        # URGENT: surface unanswered ask_help at the top of the prompt, but only
        # when this agent isn't carrying a victim (can't realistically abandon).
        is_carrying = bool(agent_info.get('carrying'))
        open_help = (None if is_carrying
                     else self.communication.get_open_help_request())

        # Requester-side: check if our own ask_help has been resolved (all "no"
        # or timed out) and surface an abandon directive.
        urgent_abandon = self._check_help_request_outcome()

        planning_inputs = {
            'context': {
                'time': self._tick_count,
                'agent': self.agent_id,
                'role': self._current_role or 'unassigned',
                'position': agent_info.get('location'),
                'carrying': agent_info.get('carrying') or 'nothing',
                'capabilities': common_ctx['agent_capabilities'],
                'teammates': teammates_enriched,
            },
            'high_level_task': self._high_level_task or '',
            'past_subtask': self._current_task or '',
            'current_task': self._current_task,
            'last_action': self._recent_actions[-1] if self._recent_actions else {},
            'observation': observation_clean,
            'all_discovered': {
                'victims': [
                    v for v in self.WORLD_STATE_GLOBAL.get('victims', [])
                    if v.get('object_id') not in {
                        r['victim_id'] for r in (self._get_rescued_victims() or [])
                    }
                ],
                'obstacles': self.WORLD_STATE_GLOBAL.get('obstacles', []),
            },
            'history': {
                'previous_tasks': self.episode_memory.to_prompt_previous_tasks(n=5),
                'rescued_victims': self._get_rescued_victims(),
                'messages': [
                    f"[{m['from']}] ({m['message_type']}) {m['text']}"
                    for m in raw_messages
                ] if raw_messages else None,
            },
            'urgent_help_request': open_help,
            'urgent_abandon': urgent_abandon,
        }

        planning_inputs['history']['area_coverage'] = self.area_tracker.get_all_summaries()

        prompt = self.planner.get_planning_prompt(planning_inputs)
        print(prompt)
        self.call_llm(prompt)
        return self._idle()

    def _handle_planning_result(self, result) -> Tuple[Optional[str], Dict]:
        text = _strip_thinking(getattr(result[0], 'content', '') or '') or ''

        # Try to parse as JSON (combined critic+plan response)
        parsed = None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = _extract_action_json(text)

        if parsed and 'next_task' in parsed:
            critic_result = {
                'reasoning': parsed.get('reasoning', ''),
                'success': bool(parsed.get('success', True)),
                'critique': parsed.get('critique', ''),
            }
            self._pipeline_context['critic_result'] = critic_result
            self.episode_memory.set_critic_result(critic_result)
            # Store as a plain string — the raw {reasoning/success/critique} dict must NOT
            # appear in the reasoning prompt's memory context, or Qwen3 pattern-matches
            # that structure and outputs it instead of a tool call.
            status = 'SUCCESS' if critic_result['success'] else f"FAILED: {critic_result['critique']}"
            self.memory.update('plan_status', f"[tick {self._tick_count}] Last action: {status}")
            print(f'[{self.agent_id}] Critic result: success={critic_result["success"]}')
            task = parsed.get('next_task', '').strip()
        else:
            # Fallback: plain-text response (backward compat)
            task = text.strip()

        self._pipeline_context['planned_task'] = task
        self.memory.update('planned_task', {'task': task, 'tick': self._tick_count})
        self.episode_memory.set_planned_task(task)
        print(f'[{self.agent_id}] Planned task: {task[:100]}')

        # Broadcast plan to teammates via normal message bus so they know what
        # this agent is doing next (each agent keeps its own heterogeneous memory).
        if task:
            ep = self.episode_memory.get_open_episode()
            last_ok = (
                ep.critic_feedback.get('success', True)
                if ep is not None and ep.critic_feedback
                else True
            )
            summary = (
                f"[plan_update] tick={self._tick_count} "
                f"task={self._current_task!r} "
                f"next={task!r} "
                f"last_outcome={'ok' if last_ok else 'fail'}"
            )
            self.send_message(Message(
                content={'message_type': 'plan_update', 'text': summary},
                from_id=self.agent_id,
                to_id=None,
            ))

        if task:
            self._current_task = task

        self._pipeline_stage = PipelineStage.REASONING
        return self._advance_pipeline()

    # ── REASONING stage ─────────────────────────────────────────────────

    def reason(self) -> Tuple[Optional[str], Dict]:
        # Reset the reasoning strategy's FSM at the start of each fresh
        # reasoning step. Two-pass strategies advance their internal phase via
        # on_llm_result() — only reset if the previous step has concluded.
        if getattr(self.reasoning, '_phase', 'main') in ('main', 'done'):
            self.reasoning.reset_phase()
        self.memory.compress()
        observation = dict(self.WORLD_STATE)
        nearby_state = self.WORLD_STATE
        if any(nearby_state.get(k) for k in ('victims', 'obstacles', 'doors')):
            observation['known'] = {
                k: v for k, v in nearby_state.items()
                if k != 'teammate_positions' and v
            }
        observation['area_exploration'] = self.area_tracker.get_all_summaries()

        state = self._build_canonical_state()
        recent_actions_list = list(self._recent_actions)
        last_critique = (
            self._pipeline_context.get('critic_result') or {}
        ).get('critique', '')
        reasoning_inputs = {
            **state,
            **self._build_common_context(),
            'agent_id': self.agent_id,
            'task_decomposition': self._pipeline_context.get('planned_task', self._current_task),
            'observation': observation,
            'memory': self.episode_memory.to_prompt_memory(n=15),
            'recent_actions': recent_actions_list,
            'last_critique': last_critique,
            'tools_available': list(self.tools_by_name.keys()),
        }

        # Anti-loop injection: if all recent actions are identical, force a replan
        if len(recent_actions_list) == 3 and len(set(
            json.dumps(a, sort_keys=True) for a in recent_actions_list
        )) == 1:
            loop_msg = (
                f'LOOP DETECTED: last 3 actions are identical ({recent_actions_list[0]}). '
                f'You MUST try a completely different action type to make progress.'
            )
            self.memory.update('loop_warning', {'warning': loop_msg, 'tick': self._tick_count})
            self.episode_memory.set_loop_warning(loop_msg)
            print(f'[{self.agent_id}] {loop_msg}')
            # Inject loop warning into critic_feedback so reasoning sees it
            if not isinstance(reasoning_inputs.get('critic_feedback'), dict):
                reasoning_inputs['critic_feedback'] = {}
            reasoning_inputs['critic_feedback']['loop_warning'] = loop_msg
        prompt = self.reasoning.get_reasoning_prompt(reasoning_inputs)
        self.call_llm(prompt, tools=self.tool_schemas)
        return self._idle()

    def _handle_reasoning_result(self, llm_response) -> Tuple[Optional[str], Dict]:
        message = llm_response[0]

        # Two-pass strategies (Self-Refine, Self-Reflective-ToT) hook here to
        # either (a) request a follow-up LLM call and stay in REASONING, or
        # (b) commit a final (name, args) action directly.
        hook_result = None
        try:
            hook_result = self.reasoning.on_llm_result(message, self)
        except Exception as exc:
            logger.warning('[%s] reasoning.on_llm_result raised: %s', self.agent_id, exc)

        if isinstance(hook_result, FollowupRequest):
            self.call_llm(
                hook_result.messages,
                tools=hook_result.tools,
                tool_choice=hook_result.tool_choice,
            )
            self._pipeline_stage = PipelineStage.REASONING
            return self._idle()

        if isinstance(hook_result, ActionCommit):
            name, args = hook_result.name, hook_result.args
            print(f'[{self.agent_id}] Reasoning strategy committed action: {name}({args})')
            self._pipeline_context['action_name'] = name
            self._pipeline_context['action_args'] = args
            self._pipeline_context['_reasoning_retries'] = 0
            self._pipeline_stage = (
                PipelineStage.COMM_DISPATCH
                if name == 'SendMessage'
                else PipelineStage.EXECUTE
            )
            return self._advance_pipeline()

        # Path A: structured tool_call
        tool_calls = getattr(message, 'tool_calls', None)
        if tool_calls is None:
            content = getattr(message, 'content', '') or ''

            # Path B: plain-text fallback — try to extract a JSON action dict
            extracted = _extract_action_json(content)
            if extracted and extracted.get('name') in self.tools_by_name:
                name = extracted['name']
                args = extracted.get('args', extracted.get('arguments', {}))
                print(
                    f'[{self.agent_id}] Fallback JSON parse succeeded: '
                    f'{name}({args})'
                )
                self._pipeline_context['action_name'] = name
                self._pipeline_context['action_args'] = args
                self._pipeline_context['_reasoning_retries'] = 0
                self._pipeline_stage = (
                    PipelineStage.COMM_DISPATCH
                    if name == 'SendMessage'
                    else PipelineStage.EXECUTE
                )
                return self._advance_pipeline()

            # Path C: no usable action — retry with a hard cap of 3 attempts
            retries = self._pipeline_context.get('_reasoning_retries', 0) + 1
            self._pipeline_context['_reasoning_retries'] = retries
            print(
                f'[{self.agent_id}] Reasoning produced no tool call '
                f'(attempt {retries}/3). '
                f'content={content[:120]!r}'
            )
            logger.warning(
                '[%s] Reasoning result missing tool_calls (attempt %d/3): %s',
                self.agent_id, retries, message,
            )
            if retries >= 3:
                print(
                    f'[{self.agent_id}] Reasoning retry cap reached — '
                    f'resetting pipeline to IDLE'
                )
                self._pipeline_context['_reasoning_retries'] = 0
                self._pipeline_stage = PipelineStage.IDLE
            else:
                self._pipeline_stage = PipelineStage.REASONING
            return self._idle()
        
        tc = tool_calls[0]
        name = tc.function.name
        args_raw = tc.function.arguments
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        print(f'[{self.agent_id}] Tool call: {name}({args})')

        self._pipeline_context['action_name'] = name
        self._pipeline_context['action_args'] = args
        
        if name == 'SendMessage':
            self._pipeline_stage = PipelineStage.COMM_DISPATCH
        else:
            self._pipeline_stage = PipelineStage.EXECUTE
        return self._advance_pipeline()

    # ── EXECUTE stage ───────────────────────────────────────────────────
    
    def communicate(self) -> Tuple[Optional[str], Dict]:
        args = self._pipeline_context.get('action_args', {})
        send_to = args.get('send_to', 'all')
        message_type = args.get('message_type', 'message')
        text = args.get('message', '')

        target = None if send_to == 'all' else send_to
        self.send_message(Message(
            content={'message_type': message_type, 'text': text},
            from_id=self.agent_id,
            to_id=target,
        ))
        self.memory.update('sent_message', {
            'entry_type': 'sent_message',
            'from': self.agent_id,
            'to': send_to,
            'message_type': message_type,
            'text': text,
            'tick': self._tick_count,
        })
        if self.metrics:
            self.metrics.record_message_sent(self._tick_count, send_to, message_type, text)

        # Quick help-reply handling: "yes"/"no" replies bypass the normal
        # announcement broadcast and trigger auto-navigation on acceptance.
        quick_reply = None
        if message_type == 'help' and target is not None:
            stripped = text.strip().lower()
            if stripped in ('yes', 'no'):
                # If already committed to a cooperative action, coerce "yes"
                # to "no" — the help autonav would stomp on the rendezvous.
                engaged = (self._coop_carry_role is not None
                           or self._coop_remove_role is not None
                           or self._carry_autopilot is not None)
                if stripped == 'yes' and engaged:
                    stripped = 'no'
                    print(f'[{self.agent_id}] Help reply coerced to "no" — cooperatively engaged')
                quick_reply = stripped
                self.communication.mark_help_answered(target)
                self._record_help_reply(requester=target, reply=quick_reply)
                if quick_reply == 'yes':
                    self._begin_help_autonav(requester=target)

        # Auto-announce: only for free-form help responses (not quick yes/no)
        if message_type == 'help' and target is not None and quick_reply is None:
            if self.communication.has_pending_ask_help(from_agent=send_to):
                self.send_message(Message(
                    content={'message_type': 'message', 'text': f'{self.agent_id} is responding to {send_to} help request'},
                    from_id=self.agent_id,
                    to_id=None,
                ))

        # Requester bookkeeping: register expected responders for own ask_help
        if message_type == 'ask_help':
            expected = self._count_eligible_responders()
            self._register_help_request_sent(expected)

        self._pipeline_stage = PipelineStage.IDLE
        return self._idle()

    def _count_eligible_responders(self) -> int:
        """Upper-bound count of teammates that could respond.

        Teammate carrying-state isn't reliably visible from perception, so we
        count every known teammate (excluding self, defensively). If some are
        busy carrying and never reply, the 500-tick timeout path covers the
        abandon case.
        """
        return sum(
            1 for t in (self.WORLD_STATE.get('teammates', []) or [])
            if t.get('object_id') != self.agent_id
        )

    def execute(self) -> Tuple[Optional[str], Dict]:
        name = self._pipeline_context['action_name']
        args = self._pipeline_context['action_args']

        if name == 'MoveTo':
            planned = self._pipeline_context.get('planned_task') or self._current_task or ''
            target_str = f"({args.get('x')}, {args.get('y')})"
            if str(args.get('x')) not in planned and str(args.get('y')) not in planned:
                print(
                    f'[{self.agent_id}] MISMATCH: MoveTo{target_str} not found in '
                    f'planned task: {planned[:120]!r}'
                )

        # Intercept cooperative actions BEFORE validation — the validator rejects
        # them when the partner isn't adjacent yet, but the rendezvous state
        # machine handles navigation; validation is irrelevant here.
        if name == 'CarryObjectTogether':
            victim_id = args.get('object_id', '')
            partner_id = args.get('partner_id', '')
            if (victim_id and partner_id
                    and self._setup_coop_carry_rendezvous(victim_id, partner_id)):
                self._pipeline_stage = PipelineStage.IDLE
                return self._idle('rendezvous_initiated')
            # Setup failed (missing/invalid ids, victim location unknown, etc.) —
            # fall through to validate + dispatch so the LLM gets an error signal.

        if name == 'RemoveObjectTogether':
            obj_id = args.get('object_id', '')
            partner_id = args.get('partner_id', '')
            if (obj_id and partner_id
                    and self._setup_coop_remove_rendezvous(obj_id, partner_id)):
                self._pipeline_stage = PipelineStage.IDLE
                return self._idle('remove_rendezvous_initiated')
            # Setup failed — fall through to validate + dispatch

        # Validate
        check = self._validate_action(name, args)
        if check is not None:
            self._recent_actions.append({'name': name, 'args': args, 'result': 'validation_failed'})
            self._pipeline_stage = PipelineStage.REASONING
            return check

        action_name, kwargs, task_completing = execute_action(name, args, self.agent_id)

        # Prune cleared obstacle so stale entries never reappear in prompts.
        # If the action fails, add_new_obs() will re-add it next tick.
        if name in ('RemoveObject', 'RemoveObjectTogether'):
            obj_id = args.get('object_id', '')
            if obj_id:
                self.WORLD_STATE_GLOBAL['obstacles'] = [
                    o for o in self.WORLD_STATE_GLOBAL.get('obstacles', [])
                    if o.get('object_id') != obj_id
                ]

        # Solo Drop at drop zone → record the rescue in SharedMemory so that
        # _get_rescued_victims() returns live data for the planning prompt.
        if action_name == 'Drop' and self.shared_memory:
            drop_zone = tuple(self.env_info.drop_zone)
            if _chebyshev_distance(self.agent_location, drop_zone) <= 1:
                carrying = self.WORLD_STATE.get('agent', {}).get('carrying', [])
                if carrying:
                    rescued_id = carrying[0]
                    rescued = self.shared_memory.retrieve('rescued_victims') or []
                    if not any(v['victim_id'] == rescued_id for v in rescued):
                        rescued = rescued + [{
                            'victim_id': rescued_id,
                            'tick': self._tick_count,
                            'agent': self.agent_id,
                            'method': 'solo',
                        }]
                        self.shared_memory.update('rescued_victims', rescued)
                        print(
                            f'[{self.agent_id}] Recorded solo rescue of '
                            f'{rescued_id} (total rescued: {len(rescued)})'
                        )

        self.memory.update('action', {'action': action_name, 'args': kwargs})
        self.episode_memory.set_action(action_name, kwargs, self._tick_count)
        self._recent_actions.append({'name': action_name, 'args': kwargs})

        if self.metrics:
            loc = self.WORLD_STATE.get('agent', {}).get('location', (0, 0))
            self.metrics.record_action(self._tick_count, action_name, kwargs, tuple(loc))

        action, updates = apply_navigation(action_name, kwargs, navigator=self._navigator, state_tracker=self._state_tracker, env_info=self.env_info, memory=self.memory)
        if 'nav_target' in updates:
            self._nav_target = updates['nav_target']
        self._pipeline_stage = PipelineStage.IDLE
        return action

