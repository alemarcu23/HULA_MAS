"""
SearchRescueAgent — LLM-driven rescue agent with MindForge-style cognitive loop.

Extends LLMAgentBase which handles all infrastructure (navigation, carry
retry, rendezvous, action validation, task injection).

This class implements a multi-stage async pipeline:
    PLANNING (critic+plan) → REASONING → EXECUTE

Each agent's high-level goal is derived directly from its assigned role(s)
and remains fixed throughout the run.  Each LLM stage is non-blocking:
submit a call, return Idle, poll next tick.  Stage outputs flow forward
via _pipeline_context.
"""

import json
import logging
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from helpers.logic_helpers import _chebyshev_distance, extract_action_json, log_stage_inputs
from helpers.navigation_helpers import apply_navigation
from matrx.agents.agent_utils.state import State
from matrx.messages.message import Message

from agents1.async_model_prompting import get_llm_result, _strip_thinking
from agents1.capabilities import DISCOVERY_NOTE, get_capability_prompt, get_game_rules
from agents1.llm_agent_base import LLMAgentBase
from agents1.modules.area_tracker import AreaExplorationTracker
from agents1.modules.execution_module import execute_action
from agents1.modules.help_coordinator import (
    HelpCoordinator,
    ResolutionEvent,
    MSG_ASK_HELP,
    MSG_HELP,
)
from agents1.modules.help_tracker import HelpTracker, count_eligible_responders
from agents1.modules.reasoning_module import (
    ReasoningReAct,
    REASONING_STRATEGY_REGISTRY,
    FollowupRequest,
    ActionCommit,
)
from agents1.modules.role_module import (
    ROLE_CLAIM_MSG_TYPE,
    pick_role,
    get_role_goal,
    get_role_hint,
    get_role_prompt,
)
from agents1.tool_registry import build_tool_schemas
from memory.episode_memory import EpisodeMemory
from memory.shared_memory import SharedMemory
from worlds1.environment_info import EnvironmentInformation

logger = logging.getLogger('SearchRescueAgent')


class PipelineStage(Enum):
    IDLE = 'idle'
    PLANNING = 'planning'
    REASONING = 'reasoning'
    EXECUTE = 'execute'
    COMM_DISPATCH = 'comm_dispatch'


class SearchRescueAgent(LLMAgentBase):
    """Rescue agent with multi-stage cognitive pipeline.

    Pipeline per cycle:
        PLANNING (critic+plan combined) → REASONING → EXECUTE

    PLANNING evaluates the last action (critic) and decides the next atomic
    task, returning JSON {reasoning, success, critique, next_task}.
    High-level goal is derived from the agent's assigned role(s) at startup.
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
        #   _current_task     — atomic next step chosen by THIS cycle's planner
        # The four fields below make the lifecycle explicit. `_last_*` are the
        # previous cycle's pair, rolled forward at the start of each new cycle.
        self._high_level_task: str = ''
        self._current_plan: Optional[str] = None        # set by PLAN handler this cycle
        self._current_action: Optional[Dict[str, Any]] = None  # set by REASONING handler this cycle
        self._last_plan: Optional[str] = None           # _current_plan from previous cycle
        self._last_action: Optional[Dict[str, Any]] = None     # _current_action from previous cycle
        # Optional second action committed by reasoning alongside a MoveTo primary.
        # Promoted to _current_action on the tick after _nav_target clears, without
        # invoking the planner. Cleared on validation rejection, critic failure,
        # urgent banners, or one successful promotion.
        self._queued_action: Optional[Dict[str, Any]] = None
        # Outcome of the most recent executed action, fed back to reasoning.
        self._last_action_outcome: str = ''

        # Cumulative belief over the run, keyed by object_id. Rebuilt each tick
        # from WORLD_STATE_GLOBAL minus rescued victims / removed obstacles.
        self._world_state_belief: Dict[str, Dict[str, Dict[str, Any]]] = {
            'victims': {}, 'obstacles': {},
        }

        # Role system — seed from hardcoded initial_role if provided
        self._current_roles: List[str] = [initial_role] if initial_role else []
        self._team_roles: Dict[str, str] = {}   # {agent_id: comma-separated roles} from incoming messages
        self._team_plans: Dict[str, str] = {}   # {agent_id: next_task} from plan_update messages

        # Anti-loop detection: track last 3 executed actions
        self._recent_actions: deque = deque(maxlen=3)

        # Structured episode memory
        self.episode_memory = EpisodeMemory()
        # Help-request response tracking
        self.help_tracker = HelpTracker(self.agent_id, shared_memory)
        # Message-based help-request coordinator (single-winner protocol).
        # Use the agent name as the stable identifier — MATRX may set
        # self.agent_id only after construction.
        self.help_coord = HelpCoordinator(self.agent_id or name)
        # Communication dedup: skip sending identical messages back-to-back
        self._recent_sent_msgs: deque = deque(maxlen=3)
        # Critic dedup: avoid injecting identical failure feedback twice in a row
        self._last_critic_failure_critique: str = ''
        # Carries the human-readable help-cancel reason into the next planning cycle.
        self._pending_help_abandon: Optional[str] = None

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
                    self._team_roles[sender] = role  # stored as comma-separated string

            if isinstance(content, dict) and content.get('message_type') == 'plan_update':
                sender = getattr(msg, 'from_id', '')
                task = content.get('task', '')
                if sender and sender != self.agent_id and task:
                    self._team_plans[sender] = task

        # Register this agent so the coordinator can build a complete team list
        # even before all agents are within perception range of each other.
        if self.shared_memory:
            self.shared_memory.add_to_set('registered_agents', self.agent_id)

        # Rebuild the cumulative world_state_belief dict once per tick from
        # WORLD_STATE_GLOBAL minus rescued victims (filter removed-obstacles
        # is applied at execute-time; see execute()).
        self._update_world_state_belief()

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

        # ── Help-coordinator ingest (message-based single-winner protocol) ──
        # Refresh agent_id in case MATRX assigned it after __init__.
        self.help_coord.set_agent_id(self.agent_id)
        rescued_ids = {
            v.get('victim_id') for v in (self._get_rescued_victims() or []) if v.get('victim_id')
        }
        outbound, events = self.help_coord.ingest(
            self.received_messages, self._tick_count, rescued_ids,
        )
        for out_msg in outbound:
            self.send_message(out_msg)
        for ev in events:
            self._apply_help_event(ev)

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

        # Roll forward: this cycle's "current" becomes next cycle's "last".
        # Critic + planner read _last_plan / _last_action, NOT _current_*.
        self._last_plan = self._current_plan
        self._last_action = self._current_action
        self._current_plan = None
        self._current_action = None

        # Open a fresh episode for this cycle
        self.episode_memory.open_episode(
            tick=self._tick_count,
            agent_id=self.agent_id,
            task=self._current_task or '',
            role=', '.join(self._current_roles) if self._current_roles else 'unassigned',
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

    def _try_promote_queued_action(self) -> bool:
        """Promote `_queued_action` to `_current_action` and jump to EXECUTE.

        Preconditions:
        - A queued action exists.
        - Navigation has finished (_nav_target is None).
        - The primary MoveTo did not produce a validation error.
        - No urgent help banner is active (those override everything).
        - The last critic did not flag failure (premise of the pair is broken).

        Returns True iff promotion happened.
        """
        if self._queued_action is None:
            return False

        if self._nav_target is not None:
            return False  # still navigating; wait

        # Validator rejected the primary — the queued action's precondition
        # (being at the destination) cannot be assumed.
        if self._last_validation_error:
            print(f'[{self.agent_id}] Dropping queued action: primary MoveTo had validation error')
            self._queued_action = None
            return False

        # Urgent help requests take precedence over any queued plan.
        if self.communication.get_open_help_request():
            print(f'[{self.agent_id}] Dropping queued action: urgent help request pending')
            self._queued_action = None
            return False

        # If recent_actions show the same action 3x, we are looping — bail to planner.
        if (len(self._recent_actions) == 3
                and len({json.dumps(a, sort_keys=True) for a in self._recent_actions}) == 1):
            print(f'[{self.agent_id}] Dropping queued action: loop detected on primary')
            self._queued_action = None
            return False

        queued = self._queued_action
        self._queued_action = None
        name, args = queued['name'], queued['args']

        # Reuse the previous cycle's plan text for continuity in memory/logs.
        self._current_plan = self._last_plan or ''
        self._current_action = {'name': name, 'args': args}
        self._pipeline_context['action_name'] = name
        self._pipeline_context['action_args'] = args
        self._pipeline_context['planned_task'] = self._current_plan
        self._pipeline_context['queued_after'] = 'MoveTo'

        self.episode_memory.set_planned_task(self._current_plan)
        self.episode_memory.set_motivation('[queued follow-up after MoveTo]')

        print(f'[{self.agent_id}] Promoted queued action: {name}({args})')

        self._pipeline_stage = (
            PipelineStage.COMM_DISPATCH if name == 'SendMessage'
            else PipelineStage.EXECUTE
        )
        return True

    def _advance_pipeline(self) -> Tuple[Optional[str], Dict]:
        # Requester suspension: hold the pipeline while our ask_help is pending.
        # Rendezvous / autopilot still run because they live in _run_infra,
        # which executes before this method.
        if self.help_coord.is_requester_waiting():
            return self._idle(reason='awaiting_help_response')
        # Responder suspension: while we have an accepted help commitment AND
        # the coop machinery (autonav / coop-carry / coop-remove rendezvous) is
        # driving us, do NOT replan. Otherwise we spam plan_update messages and
        # never actually help. Suspension lifts when _my_acceptance clears OR
        # the coop slot empties (target gone, etc.).
        if (self.help_coord.is_responder_committed()
                and (self._nav_target is not None
                     or self._coop_carry_role is not None
                     or self._coop_remove_role is not None
                     or self._carry_autopilot is not None)):
            return self._idle(reason='executing_accepted_help')

        if self._pipeline_stage == PipelineStage.IDLE:
            self._get_or_assign_role()  # assign roles and high-level goal on first cycle only
            self._is_first_cycle = False
            self._pipeline_context = {}
            self._open_episode_cycle()
            # Queued-action promotion: if reasoning previously committed a
            # MoveTo + follow-up pair and the navigation just completed cleanly,
            # execute the follow-up this cycle without invoking the planner.
            if self._try_promote_queued_action():
                # Stage was set inside the helper (EXECUTE or COMM_DISPATCH).
                pass
            else:
                self._pipeline_stage = PipelineStage.PLANNING
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
        if self._pipeline_stage == PipelineStage.PLANNING:
            return self._handle_planning_result(result)
        if self._pipeline_stage == PipelineStage.REASONING:
            return self._handle_reasoning_result(result)
        return self._idle()

    def _update_world_state_belief(self) -> None:
        """Rebuild self._world_state_belief from WORLD_STATE_GLOBAL.

        Cumulative across the run, but prunes:
          - Victims listed in SharedMemory['rescued_victims'].
          - Obstacles previously removed (already filtered out of
            WORLD_STATE_GLOBAL by execute() after RemoveObject succeeds).

        Stored as dicts keyed by object_id for O(1) lookup and prompt size.
        """
        rescued_ids: set = set()
        if self.shared_memory:
            rescued_ids = {
                v['victim_id']
                for v in (self.shared_memory.retrieve('rescued_victims') or [])
            }

        victims: Dict[str, Dict[str, Any]] = {}
        for v in self.WORLD_STATE_GLOBAL.get('victims', []) or []:
            oid = v.get('object_id')
            if not oid or oid in rescued_ids:
                continue
            victims[oid] = {
                'location': v.get('location'),
                'severity': v.get('severity'),
            }

        obstacles: Dict[str, Dict[str, Any]] = {}
        for o in self.WORLD_STATE_GLOBAL.get('obstacles', []) or []:
            oid = o.get('object_id')
            if not oid:
                continue
            obstacles[oid] = {
                'location': o.get('location'),
                'type': o.get('type'),
            }

        self._world_state_belief = {'victims': victims, 'obstacles': obstacles}

    def _get_rescued_victims(self):
        """Derive rescued victims from SharedMemory."""
        if self.shared_memory:
            rescued = self.shared_memory.retrieve('rescued_victims')
            if rescued:
                return rescued
        return None

    # ── Role assignment ─────────────────────────────────────────────────

    def _get_or_assign_role(self) -> List[str]:
        """Assign agent's roles on first cycle; keep them for the rest of execution."""
        if self._current_roles:
            return self._current_roles

        self._current_roles = pick_role(
            team_roles=self._team_roles,
            world_victims=self.WORLD_STATE_GLOBAL.get('victims', []),
            world_obstacles=self.WORLD_STATE_GLOBAL.get('obstacles', []),
            area_summaries=self.area_tracker.get_all_summaries(),
        )
        self._role_assigned_tick = self._tick_count
        role_str = ', '.join(self._current_roles)
        self._high_level_task = get_role_goal(self._current_roles)
        self.send_message(Message(
            content={'message_type': ROLE_CLAIM_MSG_TYPE, 'role': role_str},
            from_id=self.agent_id,
            to_id=None,
        ))
        print(f'[{self.agent_id}] Roles → {role_str} (tick={self._tick_count})')
        return self._current_roles

    # ── Common context builder ──────────────────────────────────────────

    def _build_common_context(self) -> Dict[str, Any]:
        """Return game rules, capability text, and role prompts for LLM stages."""
        cap_text = (
            get_capability_prompt(self._capabilities)
            if self._capability_knowledge == 'informed'
            else DISCOVERY_NOTE
        )
        return {
            'game_rules': get_game_rules(
                drop_zone=self.env_info.drop_zone,
                capabilities=self._capabilities if self._capability_knowledge == 'informed' else None,
            ),
            'agent_capabilities': cap_text,
            'current_role': ', '.join(self._current_roles) if self._current_roles else 'unassigned',
            'role_hint': get_role_hint(self._capabilities),
            'role_prompt': get_role_prompt(self._current_roles),
            'team_roles': self._team_roles,
        }

    # ── PLANNING stage (combined critic + plan) ─────────────────────────

    def plan(self) -> Tuple[Optional[str], Dict]:
        agent_info = self.WORLD_STATE.get('agent', {})
        common_ctx = self._build_common_context()

        teammates_enriched = [
            {
                'id': t['object_id'],
                'location': [t.get('x'), t.get('y')],
                'role': self._team_roles.get(t['object_id'], 'unknown'),
                'current_plan': self._team_plans.get(t['object_id'], ''),
            }
            for t in self.WORLD_STATE.get('teammates', [])
        ]

        # OBSERVATION = vision-range only. Strip self-state, walls, doors.
        observation = {
            'victims': self.WORLD_STATE.get('victims', []) or [],
            'teammates': self.WORLD_STATE.get('teammates', []) or [],
            'obstacles': self.WORLD_STATE.get('obstacles', []) or [],
        }

        agent_busy = (self._nav_target is not None
                      or self._carry_autopilot is not None
                      or self._coop_carry_role is not None
                      or self._coop_remove_role is not None)
        messages = self.communication.get_messages(limit=10, agent_busy=agent_busy)

        # URGENT: surface unanswered ask_help only when not already carrying.
        is_carrying = bool(agent_info.get('carrying'))
        open_help = (None if is_carrying
                     else self.communication.get_open_help_request())

        # Prefer the richer coordinator-derived abandon message; fall back to tracker.
        urgent_abandon = self._pending_help_abandon or self.help_tracker.check_outcome(self._tick_count)
        self._pending_help_abandon = None  # consumed — clear for next cycle

        # Carry forward the latest validation error into the planning prompt so the
        # planner knows WHY the last action failed (e.g. obstacle blocking the door).
        # Clear immediately after reading so it doesn't persist to subsequent cycles.
        last_validation_error = self._last_validation_error or ''
        self._last_validation_error = ''

        planning_inputs = {
            'context': {
                'agent': self.agent_id,
                'role': ', '.join(self._current_roles) if self._current_roles else 'unassigned',
                'position': agent_info.get('location'),
                'carrying': agent_info.get('carrying') or 'nothing',
                'capabilities': common_ctx['agent_capabilities'],
                'teammates': teammates_enriched,
            },
            'high_level_task': self._high_level_task or '',
            'last_plan': self._last_plan or '',
            'last_action': self._last_action or {},
            'observation': observation,
            'world_state_belief': self._world_state_belief,
            'memory': self.episode_memory.to_prompt_memory(n=10),
            'messages': messages,
            'urgent_help_request': open_help,
            'urgent_abandon': urgent_abandon,
            'active_help_request': self.help_coord.active_request_snapshot(),
            'my_help_acceptance': self.help_coord.my_acceptance(),
            'last_validation_error': last_validation_error,
            'area_summaries': [
                {**s, 'door': self.env_info.get_door(int(s['name'].split()[-1]))}
                for s in self.area_tracker.get_all_summaries()
            ],
        }

        print(
            f'[{self.agent_id}] PLAN inputs (tick {self._tick_count}):\n'
            + json.dumps(planning_inputs, default=str, indent=2)
        )

        prompt = self.planner.get_planning_prompt(planning_inputs)
        print(prompt)
        self.call_llm(prompt)
        return self._idle()

    def _handle_planning_result(self, result) -> Tuple[Optional[str], Dict]:
        text = _strip_thinking(getattr(result[0], 'content', '') or '') or ''

        parsed = None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = extract_action_json(text)

        critic_result: Dict[str, Any] = {'success': True, 'critique': ''}
        motivation = ''
        task = ''
        if isinstance(parsed, dict):
            critic_raw = parsed.get('critic')
            if isinstance(critic_raw, dict):
                critic_result = {
                    'success': bool(critic_raw.get('success', True)),
                    'critique': critic_raw.get('critique', '') or '',
                }
            else:
                # Back-compat: flat fields
                critic_result = {
                    'success': bool(parsed.get('success', True)),
                    'critique': parsed.get('critique', '') or '',
                }
            task = (parsed.get('next_plan') or parsed.get('next_task') or '').strip()
            motivation = (parsed.get('motivation') or '').strip()
        else:
            task = text.strip()

        self._pipeline_context['critic_result'] = critic_result
        self._pipeline_context['motivation'] = motivation
        self.episode_memory.set_critic_result(critic_result)

        status = 'SUCCESS' if critic_result['success'] else f"FAILED: {critic_result['critique']}"
        self.memory.update('plan_status', f"[tick {self._tick_count}] Last action: {status}")
        print(f'[{self.agent_id}] Critic: success={critic_result["success"]}  '
              f'critique={critic_result["critique"][:80]!r}')

        self._pipeline_context['planned_task'] = task
        self.memory.update('planned_task', {'task': task, 'tick': self._tick_count})
        self.episode_memory.set_planned_task(task)
        self.episode_memory.set_motivation(motivation)
        print(f'[{self.agent_id}] Plan: {task[:100]}  motivation: {motivation[:80]}')

        if task:
            # Broadcast plan_update so peers can populate their TEAM.current_plan.
            self.send_message(Message(
                content={'message_type': 'plan_update', 'text': f'plan: {task}', 'task': task},
                from_id=self.agent_id,
                to_id=None,
            ))
            self._current_task = task
            self._current_plan = task

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

        agent_info = self.WORLD_STATE.get('agent', {})
        observation = {
            'victims': self.WORLD_STATE.get('victims', []) or [],
            'teammates': self.WORLD_STATE.get('teammates', []) or [],
            'obstacles': self.WORLD_STATE.get('obstacles', []) or [],
        }

        recent_actions_list = list(self._recent_actions)
        critic = self._pipeline_context.get('critic_result') or {}
        last_critique = critic.get('critique', '')
        motivation = self._pipeline_context.get('motivation', '')

        common = self._build_common_context()
        is_carrying = bool(agent_info.get('carrying'))
        open_help = (None if is_carrying
                     else self.communication.get_open_help_request())

        reasoning_inputs = {
            'agent_id': self.agent_id,
            'current_role': ', '.join(self._current_roles) if self._current_roles else 'unassigned',
            'agent_capabilities': common['agent_capabilities'],
            'game_rules': common['game_rules'],
            'position': agent_info.get('location'),
            'carrying': agent_info.get('carrying') or 'nothing',
            'current_plan': self._current_plan or self._pipeline_context.get('planned_task', ''),
            'motivation': motivation,
            'last_critique': last_critique,
            'observation': observation,
            'recent_actions': recent_actions_list,
            'last_action': (
                {
                    'name': self._last_action.get('name'),
                    'args': self._last_action.get('args', {}),
                    'outcome': self._last_action_outcome,
                }
                if self._last_action else None
            ),
            'tools_available': list(self.tools_by_name.keys()),
            'urgent_help_request': open_help,
            'active_help_request': self.help_coord.active_request_snapshot(),
            'my_help_acceptance': self.help_coord.my_acceptance(),
            # Carry the latest validation error so the reasoner understands exactly
            # why the last action was rejected (e.g. obstacle blocking the door).
            'last_validation_error': self._last_validation_error or '',
        }

        # Enrich each area summary with its door location for the reasoning prompt
        area_info = []
        for s in self.area_tracker.get_all_summaries():
            try:
                area_id = int(s['name'].split()[-1])
                door = self.env_info.get_door(area_id)
            except (ValueError, AttributeError):
                door = None
            area_info.append({**s, 'door': door})
        reasoning_inputs['area_summaries'] = area_info

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
            reasoning_inputs['critic_feedback'] = {'success': False, 'critique': loop_msg, 'loop_warning': loop_msg}
        elif not critic.get('success', True):
            reasoning_inputs['critic_feedback'] = critic

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
            self._current_action = {'name': name, 'args': args}
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
            extracted = extract_action_json(content)
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
                self._current_action = {'name': name, 'args': args}
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

        # Optional second tool call: only valid as MoveTo + colocated follow-up.
        self._queued_action = None
        if len(tool_calls) >= 2:
            tc2 = tool_calls[1]
            name2 = tc2.function.name
            args2_raw = tc2.function.arguments
            args2 = json.loads(args2_raw) if isinstance(args2_raw, str) else (args2_raw or {})
            allowed_followups = {'PickUp', 'SearchArea', 'RemoveObject', 'CarryObject', 'Drop'}
            if name == 'MoveTo' and name2 in allowed_followups:
                self._queued_action = {'name': name2, 'args': args2}
                print(f'[{self.agent_id}] Queued follow-up action: {name2}({args2})')
            else:
                print(
                    f'[{self.agent_id}] Rejected 2-call pairing ({name} + {name2}); '
                    f'executing only the first.'
                )

        self._pipeline_context['action_name'] = name
        self._pipeline_context['action_args'] = args
        self._current_action = {'name': name, 'args': args}

        if name == 'SendMessage':
            self._pipeline_stage = PipelineStage.COMM_DISPATCH
        else:
            self._pipeline_stage = PipelineStage.EXECUTE
        return self._advance_pipeline()

    # ── Help-coordinator helpers ────────────────────────────────────────

    def _infer_help_victim(self) -> Optional[Tuple[str, Tuple[int, int]]]:
        """Pick the nearest visible heavy victim as the target of an ask_help.

        Falls back to any visible victim when no heavy one is in sight.
        Returns (victim_id, (x, y)) or None.
        """
        agent_info = self.WORLD_STATE.get('agent', {}) or {}
        loc = agent_info.get('location') or (0, 0)
        ax, ay = int(loc[0]), int(loc[1])
        victims = self.WORLD_STATE.get('victims', []) or []
        if not victims:
            return None

        def _dist(v):
            vl = v.get('location') or (v.get('x', 0), v.get('y', 0))
            return abs(int(vl[0]) - ax) + abs(int(vl[1]) - ay)

        heavy = [v for v in victims if 'critical' in (v.get('obj_id', '') or v.get('object_id', ''))]
        pool = heavy or victims
        chosen = min(pool, key=_dist)
        vid = chosen.get('obj_id') or chosen.get('object_id') or ''
        vl = chosen.get('location') or (chosen.get('x', 0), chosen.get('y', 0))
        return (vid, (int(vl[0]), int(vl[1])))

    # ── Help-coordinator event handler ──────────────────────────────────

    def _apply_help_event(self, ev: ResolutionEvent) -> None:
        """React to a resolution emitted by HelpCoordinator.ingest."""
        rid = ev.request_id
        requester = ev.requester or self.agent_id

        # Always purge the exchange from local inbox.
        self.communication.purge_help_exchange(rid, requester)

        if ev.kind == 'rescued':
            collab = {
                'requester': requester,
                'responder': ev.responder or '',
                'victim_id': ev.victim_id or '',
                'duration_ticks': ev.duration_ticks if ev.duration_ticks is not None else -1,
                'outcome': 'rescued',
            }
            self.episode_memory.set_collaboration(collab)
            self.memory.update('collaboration', {**collab, 'tick': self._tick_count})
            print(f'[{self.agent_id}] Collaboration recorded: {collab}')
        elif ev.kind == 'canceled':
            reason = ev.reason or 'unknown'
            collab = {
                'requester': requester, 'victim_id': ev.victim_id or '',
                'outcome': f'canceled:{reason}',
            }
            self.episode_memory.set_collaboration(collab)
            self.memory.update('collaboration', {**collab, 'tick': self._tick_count})
            # If WE were the requester, mark the episode failed and reset pipeline.
            if requester == self.agent_id:
                self.episode_memory.close_episode(
                    tick=self._tick_count, succeeded=False,
                    reason=f'help refused by peers ({reason})',
                )
                self._pipeline_stage = PipelineStage.IDLE
                self._pending_future = None
                # Build the human-readable abandon message for the next planning cycle.
                _abandon_texts = {
                    'all_declined': 'All teammates explicitly refused your help request. You MUST abandon this task and choose a different objective.',
                    'all_ignored': (
                        'All teammates received your help request but did not respond '
                        'after replanning multiple times — they have seen the request and '
                        'chosen not to help. You MUST abandon this task and choose a different objective.'
                    ),
                    'timeout': 'Your help request timed out with no response. You MUST abandon this task and choose a different objective.',
                }
                self._pending_help_abandon = _abandon_texts.get(
                    reason,
                    f'Help request canceled ({reason}). You MUST abandon this task and choose a different objective.',
                )
                # Clear the SharedMemory tracker so check_outcome() does not fire again.
                self.help_tracker.clear_request()
                print(f'[{self.agent_id}] Help canceled ({reason}) — episode failed, pipeline reset')
        elif ev.kind == 'lost_assignment':
            # Another responder was chosen — abort our autonav and resume.
            self._nav_target = None
            self.memory.update('help_lost_assignment', {
                'request_id': rid, 'chosen': ev.responder, 'tick': self._tick_count,
            })
            print(f'[{self.agent_id}] Lost help assignment — {ev.responder} chosen, returning to task')

    # ── EXECUTE stage ───────────────────────────────────────────────────

    def communicate(self) -> Tuple[Optional[str], Dict]:
        args = self._pipeline_context.get('action_args', {})
        send_to = args.get('send_to', 'all')
        message_type = args.get('message_type', 'message')
        text = args.get('message', '')

        target = None if send_to == 'all' else send_to

        # Vet help-related sends against the single-winner protocol.
        teammates = self.WORLD_STATE.get('teammates', [])
        peer_ids = (
            [t['object_id'] for t in teammates
             if t.get('object_id') and t['object_id'] != self.agent_id]
            if message_type == MSG_ASK_HELP else []
        )
        expected = len(peer_ids) if peer_ids else (
            count_eligible_responders(teammates, self.agent_id)
            if message_type == MSG_ASK_HELP else 0
        )
        # Auto-infer victim id/location when the LLM omits them in an ask_help.
        v_id = args.get('victim_id', '')
        v_loc = args.get('victim_location')
        if message_type == MSG_ASK_HELP and (not v_id or not v_loc):
            inferred = self._infer_help_victim()
            if inferred:
                v_id = v_id or inferred[0]
                v_loc = v_loc or inferred[1]
        # Determine help kind: 'carry' for victim coop-carry, 'remove' for obstacle coop-remove.
        # Honor an explicit LLM-provided `kind`; otherwise infer from target id.
        ask_kind = (args.get('kind') or '').strip().lower() if message_type == MSG_ASK_HELP else ''
        ask_target_id = args.get('target_id', '') if message_type == MSG_ASK_HELP else ''
        if message_type == MSG_ASK_HELP and not ask_kind:
            tid_lower = (ask_target_id or v_id or '').lower()
            ask_kind = 'remove' if ('obstacle' in tid_lower or 'rock' in tid_lower or 'tree' in tid_lower) else 'carry'
        if message_type == MSG_ASK_HELP and not ask_target_id:
            ask_target_id = v_id or ''
        action, extra = self.help_coord.vet_outgoing(
            message_type=message_type,
            send_to=send_to,
            text=text,
            tick=self._tick_count,
            victim_id=v_id,
            victim_location=v_loc,
            expected_responders=expected,
            expected_peer_ids=peer_ids if message_type == MSG_ASK_HELP else None,
            kind=ask_kind or 'carry',
            target_id=ask_target_id,
        )
        if action == 'suppress':
            print(f'[{self.agent_id}] Help send suppressed: {extra}')
            self.memory.update('help_blocked', {'reason': extra, 'tick': self._tick_count})
            self._pipeline_stage = PipelineStage.IDLE
            return self._idle()
        if action == 'rewrite':
            if 'text' in extra:
                text = extra['text']

        content_payload = {'message_type': message_type, 'text': text}
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k != 'text':
                    content_payload[k] = v

        self.send_message(Message(
            content=content_payload,
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
                self.help_tracker.record_reply(requester_id=target, reply=quick_reply)
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
            expected = count_eligible_responders(
                self.WORLD_STATE.get('teammates', []), self.agent_id
            )
            self.help_tracker.register_sent(self._tick_count, expected)

        self._pipeline_stage = PipelineStage.IDLE
        return self._idle()

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
            self._last_action_outcome = 'rejected_by_validator'
            # The queued follow-up depends on this primary succeeding; drop it.
            if self._queued_action is not None:
                print(f'[{self.agent_id}] Dropping queued action: primary {name} rejected by validator')
                self._queued_action = None
            self._pipeline_context['critic_result'] = {
                'success': False,
                'critique': self._last_validation_error or f"Action {name} failed validation.",
            }
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
        # Track outcome for reasoning's LAST ACTION block on the next cycle.
        self._last_action_outcome = 'dispatched'

        if self.metrics:
            loc = self.WORLD_STATE.get('agent', {}).get('location', (0, 0))
            self.metrics.record_action(self._tick_count, action_name, kwargs, tuple(loc))

        action, updates = apply_navigation(action_name, kwargs, navigator=self._navigator, state_tracker=self._state_tracker, env_info=self.env_info, memory=self.memory)
        if 'nav_target' in updates:
            self._nav_target = updates['nav_target']
        self._pipeline_stage = PipelineStage.IDLE
        return action

