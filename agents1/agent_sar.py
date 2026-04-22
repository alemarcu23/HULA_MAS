"""
SearchRescueAgent — LLM-driven rescue agent with MindForge-style cognitive loop.

Extends LLMAgentBase which handles all infrastructure (navigation, carry
retry, rendezvous, action validation, task injection).

This class implements a multi-stage async pipeline:
    CRITIC → COMMUNICATION → PLANNING → REASONING → EXECUTE

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

from agents1.async_model_prompting import get_llm_result
from agents1.capabilities import DISCOVERY_NOTE, get_capability_prompt, get_game_rules
from agents1.llm_agent_base import LLMAgentBase, SM_TASK_ASSIGNMENTS
from agents1.modules.area_tracker import AreaExplorationTracker
from agents1.modules.execution_module import execute_action
from agents1.modules.reasoning_module import ReasoningIO
from agents1.modules.task_critic_module import CriticBase
from agents1.tool_registry import REASONING_STRATEGIES, build_tool_schemas
from memory.shared_memory import SharedMemory
from worlds1.environment_info import EnvironmentInformation

logger = logging.getLogger('SearchRescueAgent')


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
    CRITIC = 'critic'
    PLANNING = 'planning'
    REASONING = 'reasoning'
    EXECUTE = 'execute'
    COMMUNICATION = 'communication'
    COMM_DISPATCH = 'comm_dispatch'


class SearchRescueAgent(LLMAgentBase):
    """Rescue agent with multi-stage cognitive pipeline.

    Pipeline per cycle:
        [CRITIC] → COMMUNICATION → PLANNING → REASONING → EXECUTE

    Critic runs on all cycles except the first (no previous action to evaluate)
    and when the last action was Idle/None. Communication broadcasts a
    coordination message to teammates before planning.
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
        api_base: Optional[str] = None,
        capabilities: Optional[Dict] = None,
        capability_knowledge: str = 'informed',
        comm_strategy: str = 'always_respond',
        env_info: Optional[EnvironmentInformation] = None,
        use_planner: bool = True,
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
            api_base=api_base,
            capabilities=capabilities,
            capability_knowledge=capability_knowledge,
            comm_strategy=comm_strategy,
            env_info=env_info,
            use_planner=use_planner,
        )
        self._strategy = strategy if strategy in REASONING_STRATEGIES else 'react'
        self.area_tracker = AreaExplorationTracker(self.env_info.get_area_cells())
        self.tools_by_name, self.tool_schemas = build_tool_schemas()

        self.reasoning = ReasoningIO('EMPTY')
        self.critic_module = CriticBase('EMPTY')

        # Pipeline state
        self._pipeline_stage: PipelineStage = PipelineStage.IDLE
        self._pipeline_context: Dict[str, Any] = {}
        self._is_first_cycle: bool = True
        self._last_action: Dict[str, Any] = {}

        # Cursor into CommunicationModule.all_messages_raw so we only save new
        # messages each tick without duplicating entries already in memory.
        self._comm_msg_cursor: int = 0

        # Communication throttle state
        self._last_comm_tick: int = -1
        self._last_task_at_comm: str = ''
        self._last_msg_count_at_comm: int = 0

        # Anti-loop detection: track last 3 executed actions
        self._recent_actions: deque = deque(maxlen=3)
        # Communication dedup: skip sending identical messages back-to-back
        self._recent_sent_msgs: deque = deque(maxlen=3)
        # Critic dedup: avoid injecting identical failure feedback twice in a row
        self._last_critic_failure_critique: str = ''

        print(
            f'[SearchRescueAgent] Created '
            f'(model={llm_model}, strategy={self._strategy}, '
            f'planning={planning_mode}, caps={capabilities})'
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
        self._last_action = {}
        self._comm_msg_cursor = 0
        self._recent_actions.clear()
        self._last_critic_failure_critique = ''

    # ── Perception ──────────────────────────────────────────────────────

    def update_knowledge(self, filtered_state: State) -> None:
        super().update_knowledge(filtered_state)

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

    def _advance_pipeline(self) -> Tuple[Optional[str], Dict]:
        if self._pipeline_stage == PipelineStage.IDLE:
            if self._is_first_cycle:
                self._is_first_cycle = False
                self._pipeline_stage = PipelineStage.COMMUNICATION
            else:
                self._pipeline_stage = PipelineStage.CRITIC
            self._pipeline_context = {}

        if self._pipeline_stage == PipelineStage.CRITIC:
            return self.critic()
        if self._pipeline_stage == PipelineStage.PLANNING:
            return self.plan()
        if self._pipeline_stage == PipelineStage.REASONING:
            return self.reason()
        if self._pipeline_stage == PipelineStage.EXECUTE:
            return self.execute()
        if self._pipeline_stage == PipelineStage.COMMUNICATION:
            return self.communication_stage()
        if self._pipeline_stage == PipelineStage.COMM_DISPATCH:
            return self.communicate()

        return self._idle()

    def _on_llm_result(self, result) -> Tuple[Optional[str], Dict]:
        if self.metrics:
            self.metrics.record_llm_call_end()
        if self._pipeline_stage == PipelineStage.CRITIC:
            return self._handle_critic_result(result)
        if self._pipeline_stage == PipelineStage.COMMUNICATION:
            return self._handle_communication_result(result)
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

    # ── CRITIC stage ────────────────────────────────────────────────────

    def _build_canonical_state(self) -> Dict[str, Any]:
        """Return a consistent base state dict shared by all pipeline stages."""
        agent_info = self.WORLD_STATE.get('agent', {})
        return {
            'position': agent_info.get('location'),
            'carrying': agent_info.get('carrying'),
            'current_task': self._current_task,
            'last_action': self._last_action,
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
        }

    def critic(self) -> Tuple[Optional[str], Dict]:
        last_name = self._last_action.get('name')
        if not last_name or last_name == 'Idle':
            self._pipeline_stage = PipelineStage.COMMUNICATION
            return self._advance_pipeline()
        
        state = self._build_canonical_state()
        critic_inputs = {
            **state,
            'observation': self.WORLD_STATE,
            'all_observations': self.WORLD_STATE_GLOBAL,
            **self._build_common_context(),
        }
        self._log_stage_inputs('CRITIC', critic_inputs)
        prompt = self.critic_module.get_critic_prompt(critic_inputs)
        self.call_llm(prompt)
        return self._idle()

    def _handle_critic_result(self, result) -> Tuple[Optional[str], Dict]:
        text = getattr(result[0], 'content', '') or ''
        try:
            critic_result = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            critic_result = {'success': False, 'reasoning': text, 'critique': text}

        self._pipeline_context['critic_result'] = critic_result
        self.memory.update('critic_feedback', critic_result)
        print(f'[{self.agent_id}] Critic result: success={critic_result.get("success")}')

        self._pipeline_stage = PipelineStage.COMMUNICATION
        return self._advance_pipeline()

    # ── COMMUNICATION stage (pre-planning coordination) ─────────────────

    def _build_communication_prompt(self) -> list:
        agent_info = self.WORLD_STATE.get('agent', {})
        position = agent_info.get('location')
        carrying = agent_info.get('carrying')
        recent_msgs = self.communication.get_messages(limit=10, agent_busy=False)
        area_summaries = self.area_tracker.get_all_summaries()

        teammate_msgs = ''
        if recent_msgs:
            teammate_msgs = '\n'.join(
                f"- [{m['from']}] ({m['message_type']}) {m['text']}"
                for m in recent_msgs
            )
        else:
            teammate_msgs = 'No recent messages from teammates.'

        # Prepend the last few messages sent by this agent so the LLM knows
        # what it already broadcast and can avoid repeating itself.
        sent_msgs = [
            e for e in self.memory.retrieve_all()[-30:]
            if isinstance(e, dict) and e.get('entry_type') == 'sent_message'
        ]
        if sent_msgs:
            sent_lines = '\n'.join(
                f"- [YOU → {m['to']}] ({m.get('message_type', m.get('type', 'message'))}) {m['text']}"
                for m in sent_msgs[-5:]
            )
            teammate_msgs = f'Messages you sent recently:\n{sent_lines}\n\nMessages from teammates:\n{teammate_msgs}'

        area_status = '\n'.join(
            f"- {s['name']}: {s['coverage']:.0%} explored ({s['status']})"
            for s in area_summaries
        )

        common_ctx = self._build_common_context()
        cap_text = common_ctx['agent_capabilities']

        system_prompt = (
            'You are a coordination module for a Search and Rescue agent team. '
            'Your job is to broadcast a short status update to your teammates '
            'so they can avoid duplicate work and coordinate effectively.\n\n'
            f'Your capabilities:\n{cap_text}\n\n'
            'Respond with ONLY a JSON object:\n'
            '{"message": "<your status update to teammates>"}\n\n'
            'The message should briefly state: what you are currently doing, '
            'where you are, what you plan to do next, and any discoveries '
            '(victims, obstacles) worth sharing. Mention if you need help '
            'due to capability limits (e.g. low medical). Keep it under 2 sentences.'
        )

        user_prompt = (
            f'Agent: {self.agent_id}\n'
            f'Current task: {self._current_task}\n'
            f'Position: {position}\n'
            f'Carrying: {carrying}\n'
            f'Area exploration status:\n{area_status}\n\n'
            f'Recent teammate messages:\n{teammate_msgs}'
        )

        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

    def communication_stage(self) -> Tuple[Optional[str], Dict]:
        critic_failed = (
            self._pipeline_context.get('critic_result') or {}
        ).get('success') is False
        current_msg_count = len(self.communication.get_messages(limit=999, agent_busy=False))
        new_messages = current_msg_count > self._last_msg_count_at_comm
        task_changed = self._current_task != self._last_task_at_comm

        if not critic_failed and not new_messages and not task_changed:
            self._pipeline_stage = PipelineStage.PLANNING
            return self._advance_pipeline()

        self._last_comm_tick = self._tick_count
        self._last_task_at_comm = self._current_task
        self._last_msg_count_at_comm = current_msg_count

        agent_info = self.WORLD_STATE.get('agent', {})
        comm_inputs = {
            'position': agent_info.get('location'),
            'carrying': agent_info.get('carrying'),
            'current_task': self._current_task,
            'critic_feedback': self._pipeline_context.get('critic_result'),
            'recent_received_msgs': self.communication.get_messages(limit=10, agent_busy=False),
            'sent_msgs_in_memory': [
                e for e in self.memory.retrieve_all()[-30:]
                if isinstance(e, dict) and e.get('entry_type') == 'sent_message'
            ],
            'area_summaries': self.area_tracker.get_all_summaries(),
        }
        self._log_stage_inputs('COMMUNICATION', comm_inputs)
        prompt = self._build_communication_prompt()
        # Cap tokens to enforce the "≤ 2 sentences" brevity requirement mechanically
        self.call_llm(prompt, max_tokens=200)
        return self._idle()

    def _handle_communication_result(self, result) -> Tuple[Optional[str], Dict]:
        text = getattr(result[0], 'content', '') or ''

        # Try to parse JSON response
        message_text = None
        parsed = _extract_action_json(text)
        if parsed and 'message' in parsed:
            message_text = parsed['message']
        else:
            # Fallback: use raw text as the message
            message_text = text.strip()

        if message_text:
            # Skip if this exact message was sent recently (dedup)
            if message_text in self._recent_sent_msgs:
                print(f'[{self.agent_id}] Skipping duplicate coordination msg: {message_text[:80]}')
            else:
                self._recent_sent_msgs.append(message_text)
                self.send_message(Message(
                    content={'message_type': 'message', 'text': message_text},
                    from_id=self.agent_id,
                    to_id=None,  # broadcast to all
                ))
                self.memory.update('sent_message', {
                    'entry_type': 'sent_message',
                    'from': self.agent_id,
                    'to': 'all',
                    'message_type': 'message',
                    'text': message_text,
                    'tick': self._tick_count,
                })
                if self.metrics:
                    self.metrics.record_message_sent(self._tick_count, 'all', 'message', message_text)
                print(f'[{self.agent_id}] Coordination msg: {message_text[:120]}')
        else:
            print(f'[{self.agent_id}] Communication stage produced empty message, skipping')

        self._pipeline_stage = PipelineStage.PLANNING
        return self._advance_pipeline()

    # ── PLANNING stage ──────────────────────────────────────────────────

    def plan(self) -> Tuple[Optional[str], Dict]:
        # Get raw messages for planner (messages flow into planning, not reasoning)
        agent_busy = (self._nav_target is not None
                      or self._carry_autopilot is not None
                      or self._coop_carry_role is not None)
        raw_messages = self.communication.get_messages(limit=10, agent_busy=agent_busy)

        state = self._build_canonical_state()
        planning_inputs = {
            'agent_id': self.agent_id,
            **state,
            **self._build_common_context(),
            'previous_tasks': self.memory.retrieve_all()[-5:],
            'nearby_objects': self.WORLD_STATE.get('victims', []) + self.WORLD_STATE.get('obstacles', []),
            'observed_objects': self.WORLD_STATE_GLOBAL,
            'rescued_victims': self._get_rescued_victims(),
            'area_exploration': self.area_tracker.get_all_summaries(),
            'messages': [
                f"[{m['from']}] ({m['message_type']}) {m['text']}"
                for m in raw_messages
            ] if raw_messages else None,
        }
        self._log_stage_inputs('PLANNING', planning_inputs)
        prompt = self.planner.get_planning_prompt(planning_inputs)
        self.call_llm(prompt)
        return self._idle()

    def _handle_planning_result(self, result) -> Tuple[Optional[str], Dict]:
        text = getattr(result[0], 'content', '') or ''
        task = text.strip()
        self._pipeline_context['planned_task'] = task
        self.memory.update('planned_task', {'task': task, 'tick': self._tick_count})
        print(f'[{self.agent_id}] Planned task: {task[:100]}')

        if task:
            # Persist as current task so COMMUNICATION/CRITIC/REASONING all see it.
            # Direct assignment avoids the full pipeline reset that set_current_task() triggers.
            self._current_task = task

        self._pipeline_stage = PipelineStage.REASONING
        return self._advance_pipeline()

    # ── REASONING stage ─────────────────────────────────────────────────

    def reason(self) -> Tuple[Optional[str], Dict]:
        observation = dict(self.WORLD_STATE)
        global_state = self.WORLD_STATE_GLOBAL
        if any(global_state.get(k) for k in ('victims', 'obstacles', 'doors')):
            observation['known'] = {
                k: v for k, v in global_state.items()
                if k != 'teammate_positions' and v
            }
        observation['area_exploration'] = self.area_tracker.get_all_summaries()

        state = self._build_canonical_state()
        recent_actions_list = list(self._recent_actions)
        reasoning_inputs = {
            **state,
            **self._build_common_context(),
            'task_decomposition': self._pipeline_context.get('planned_task', self._current_task),
            'observation': observation,
            'memory': self.memory.retrieve_all()[-15:],
            'recent_actions': recent_actions_list,
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
            print(f'[{self.agent_id}] {loop_msg}')
            # Inject loop warning into critic_feedback so reasoning sees it
            if not isinstance(reasoning_inputs.get('critic_feedback'), dict):
                reasoning_inputs['critic_feedback'] = {}
            reasoning_inputs['critic_feedback']['loop_warning'] = loop_msg
        self._log_stage_inputs('REASONING', reasoning_inputs)
        prompt = self.reasoning.get_reasoning_prompt(reasoning_inputs)
        self.call_llm(prompt, tools=self.tool_schemas)
        return self._idle()

    def _handle_reasoning_result(self, llm_response) -> Tuple[Optional[str], Dict]:
        message = llm_response[0]

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

        # Auto-announce: if responding to a help request, broadcast it
        if message_type == 'help' and target is not None:
            if self.communication.has_pending_ask_help(from_agent=send_to):
                self.send_message(Message(
                    content={'message_type': 'message', 'text': f'{self.agent_id} is responding to {send_to} help request'},
                    from_id=self.agent_id,
                    to_id=None,
                ))

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

        # Intercept CarryObjectTogether BEFORE validation — the validator rejects
        # the action when the partner isn't adjacent yet, but our rendezvous state
        # machine handles navigation; validation is irrelevant here.
        if name == 'CarryObjectTogether':
            victim_id = args.get('object_id', '')
            partner_id = args.get('partner_id', '')
            if victim_id and self._setup_coop_carry_rendezvous(victim_id, partner_id):
                self._pipeline_stage = PipelineStage.IDLE
                return self._idle('rendezvous_initiated')
            # Setup failed (victim location unknown) — fall through to validate + dispatch

        # Validate
        check = self._validate_action(name, args)
        if check is not None:
            self._last_action = {'name': name, 'args': args, 'result': 'validation_failed'}
            self._pipeline_stage = PipelineStage.REASONING
            return check

        action_name, kwargs, task_completing = execute_action(name, args, self.agent_id)

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
        self._last_action = {'name': action_name, 'args': kwargs}
        self._recent_actions.append({'name': action_name, 'args': kwargs})

        if task_completing and task_completing != 'N/A':
            self.planner.advance_task(task_completing)

        if self.metrics:
            loc = self.WORLD_STATE.get('agent', {}).get('location', (0, 0))
            self.metrics.record_action(self._tick_count, action_name, kwargs, tuple(loc))

        action, updates = apply_navigation(action_name, kwargs, navigator=self._navigator, state_tracker=self._state_tracker, env_info=self.env_info, memory=self.memory)
        if 'nav_target' in updates:
            self._nav_target = updates['nav_target']
        self._pipeline_stage = PipelineStage.IDLE
        return action

