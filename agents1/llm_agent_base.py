"""
LLMAgentBase — Framework base class for all LLM-driven MATRX rescue agents.

Subclasses only need to implement two methods:
    - decide_on_actions(filtered_state)  — reasoning / LLM submission step
    - filter_observations(state)         — what to perceive (optional: base
                                           class provides a sensible default)

Everything else — navigation, carry autopilot (auto drop-zone after
cooperative carry), LLM polling, action validation, and task injection —
is handled automatically by this class.
"""

import json
import logging
import re
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple

from helpers.logic_helpers import _chebyshev_distance
from helpers.navigation_helpers import apply_navigation, navigate_to
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message

from actions1.CustomActions import CarryObjectTogether as _CarryObjectTogether
from actions1.CustomActions import Idle as _Idle
from matrx.actions.object_actions import DropObject as _DropObject

from agents1.async_model_prompting import get_llm_result, submit_llm_call
from agents1.modules.communication_module import CommunicationModule
from agents1.modules.execution_module import execute_action
from helpers.perception_module import Perception
from agents1.modules.planning_module import Planning
from helpers.logic_module import ActionValidator
from brains1.ArtificialBrain import ArtificialBrain
from memory.base_memory import BaseMemory
from memory.shared_memory import SharedMemory
from metrics.agent_metrics import AgentMetricsTracker
from worlds1.environment_info import EnvironmentInformation

logger = logging.getLogger('LLMAgentBase')

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_NR_TOKENS: int = 5000
TEMPERATURE: float = 0.3

# SharedMemory key templates
SM_CARRY_AUTOPILOT = 'carry_autopilot'
SM_TASK_ASSIGNMENTS = 'current_task_assignments'

# ── Base class ─────────────────────────────────────────────────────────────────

class LLMAgentBase(ArtificialBrain, Perception):
    """Infrastructure base class for LLM-driven rescue agents.

    Args:
        slowdown:       ArtificialBrain tick slow-down factor.
        condition:      World condition ('normal' | 'strong' | 'weak').
        name:           Cooperative partner name (human or AI agent ID).
        folder:         Working folder path (passed to ArtificialBrain).
        llm_model:      LiteLLM model string, e.g. ``'ollama/llama3'``.
        include_human:  Whether to include the human in the observation filter.
        shared_memory:  Optional SharedMemory instance for cross-agent sharing.
        planning_mode:  Planning strategy: ``'simple'`` (flat list) or
                        ``'dag'`` (task graph with conditional branching).
    """

    # ── Constructor ───────────────────────────────────────────────────────

    def __init__(
        self,
        slowdown: int,
        condition: str,
        name: str,
        folder: str,
        llm_model: str = 'ollama/llama3',
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
        super().__init__(slowdown, condition, name, folder)

        self._llm_model = llm_model
        self._api_base = api_base
        self._include_human = include_human
        self._partner_name = name
        self.teammates: set = set()

        # ── Planner integration ───────────────────────────────────────────
        self._use_planner = use_planner
        self._planner_agent_id = 'PlannerAgent'
        self._received_planner_task = False

        # ── Environment info ─────────────────────────────────────────────
        self.env_info: EnvironmentInformation = env_info or EnvironmentInformation()

        # ── Capabilities ──────────────────────────────────────────────────
        self._capabilities = capabilities
        self._capability_knowledge = capability_knowledge

        # ── Action validator ──────────────────────────────────────────────
        self._validator = ActionValidator(
            capabilities=capabilities,
            capability_knowledge=capability_knowledge,
            grid_size=self.env_info.grid_size,
            valid_areas=frozenset(self.env_info.areas.keys()) if self.env_info.areas else None,
            env_info=self.env_info,
        )

        # ── Communication ─────────────────────────────────────────────────
        self._comm_strategy = comm_strategy

        # ── Memory ─────────────────────────────────────────────────────────
        self.memory = BaseMemory()
        self.shared_memory: Optional[SharedMemory] = shared_memory

        # ── Planning ───────────────────────────────────────────────────────
        self.planner = Planning(mode=planning_mode)
        self.task_num: int = 0

        # ── Navigation ─────────────────────────────────────────────────────
        self._state_tracker: Optional[StateTracker] = None
        self._navigator: Optional[Navigator] = None
        self.state_for_navigation: Optional[State] = None
        self._nav_target: Optional[Tuple[int, int]] = None

        # ── Async LLM state ────────────────────────────────────────────────
        self._pending_future: Optional[Future] = None

        # ── Task ───────────────────────────────────────────────────────────
        self._current_task: Optional[str] = None

        # ── Cooperative carry autopilot ─────────────────────────────────────
        self._carry_autopilot: Optional[Dict] = None  # {'victim_id', 'destination', 'role'}

        # ── Metrics ────────────────────────────────────────────────────────
        self.metrics: Optional[AgentMetricsTracker] = None  # initialized in initialize()
        self._tick_count: int = 0

        # ── World state (populated each tick by _tick_setup) ───────────────
        self.WORLD_STATE: Dict = {}

    def initialize(self) -> None:
        """Called once before the simulation starts."""
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(
            agent_id=self.agent_id,
            action_set=self.action_set,
            algorithm=Navigator.A_STAR_ALGORITHM,
        )
        self.metrics = AgentMetricsTracker(agent_id=self.agent_id)
        self.init_global_state()
        self.communication = CommunicationModule(
            agent_id=self.agent_id,
            strategy=self._comm_strategy,
            llm_model=self._llm_model,
            api_base=self._api_base,
        )
        logger.info('[%s] LLMAgentBase ready (model=%s)', self.agent_id, self._llm_model)

    # ── Default perception filter ─────────────────────────────────────────

    def filter_observations(self, state: State) -> State:
        """Restrict observations to 1-block Chebyshev radius + doors + self.

        Saves the full unfiltered state to ``state_for_navigation`` for A*.
        Subclasses may override this for custom perception ranges.
        """
        agent_loc = state[self.agent_id]['location']
        self.state_for_navigation = state.copy()
        filtered = state.copy()
        self.teammates = set()

        keep = {self.agent_id, 'World'}
        if self._include_human:
            keep.add(self._partner_name)
            self.teammates.add((
                self._partner_name,
                state.get(self._partner_name, {}).get('location', [0, 0]),
            ))

        for obj_id, obj_data in filtered.items():
            if obj_id in keep:
                continue
            if isinstance(obj_id, str) and obj_id.startswith('rescuebot'):
                keep.add(obj_id)
                self.teammates.add((obj_id, state.get(obj_id, {}).get('location', [0, 0])))
            if _chebyshev_distance(agent_loc, obj_data.get('location')) <= 1:
                keep.add(obj_id)
            if 'door' in str(obj_id).lower():
                keep.add(obj_id)

        for obj_id in list(filtered.keys()):
            if obj_id not in keep:
                filtered.remove(obj_id)

        return filtered

    # ── Task injection ────────────────────────────────────────────────────

    def set_current_task(self, task: str) -> None:
        """Inject a high-level task from the EnginePlanner.

        Resets navigation and pending LLM state. Per-agent memory is
        preserved across task transitions.
        """
        if not isinstance(task, str):
            task = json.dumps(task, default=str)
        self._current_task = task
        self._pending_future = None
        self._nav_target = None
        self.planner.set_current_task(task)
        print(f'[{self.agent_id}] Task: {task}')

    def set_manual_task_decomposition(self, decomposition: str) -> None:
        """Inject a manual plan (numbered list, possibly with sub-bullets).

        Must be called after ``set_current_task()``.
        """
        lines = decomposition.strip().split('\n')
        entries: list[str] = []
        for line in lines:
            stripped = line.strip()
            if re.match(r'\d+\.', stripped):
                parts = stripped.split('. ', 1)
                entries.append(parts[1] if len(parts) > 1 else stripped)
            elif stripped.startswith('-') and entries:
                entries[-1] += '\n' + line   # merge sub-bullet into parent
        self.planner.set_manual_task_decomposition(entries)
        self.task_num = len(entries)

    # ── Per-tick infrastructure entry-points ──────────────────────────────

    def update_knowledge(self, filtered_state: State) -> None:
        """Update state tracker and perception. Call at the top of ``decide_on_actions``."""
        self._tick_count += 1
        self._state_tracker.update(self.state_for_navigation)
        self.WORLD_STATE = self.percept_state(
            filtered_state, agent_id=self.agent_id, teammates=self.teammates
        )
        self.update_world_belief(filtered_state)

        # Record location for metrics
        if self.metrics:
            loc = self.WORLD_STATE.get('agent', {}).get('location')
            if loc:
                self.metrics.record_location(self._tick_count, loc[0], loc[1])

        # Detect newly visible victims for metrics
        if self.metrics:
            known_ids = {v['victim_id'] for v in self.metrics.victims_found}
            for v in self.WORLD_STATE.get('victims', []):
                vid = v.get('obj_id', '')
                if vid and vid not in known_ids:
                    severity = 'critical' if 'critical' in vid else 'mild'
                    vloc = v.get('location', (0, 0))
                    self.metrics.record_victim_found(self._tick_count, vid, severity, tuple(vloc))

        # Separate planner protocol messages from peer messages
        planner_msgs, peer_msgs = self._split_planner_messages(self.received_messages)
        self._handle_planner_messages(planner_msgs)
        self.communication.process_messages(peer_msgs)

        # Record received messages for metrics
        if self.metrics:
            for msg in peer_msgs:
                content = msg.content if hasattr(msg, 'content') else msg
                if isinstance(content, dict):
                    self.metrics.record_message_received(
                        self._tick_count,
                        getattr(msg, 'from_id', ''),
                        content.get('message_type', 'message'),
                        content.get('text', ''),
                    )

    # ── Planner message handling ────────────────────────────────────────

    _PLANNER_MSG_TYPES = frozenset({'task_assignment', 'planner_answer'})

    def _split_planner_messages(self, messages: list):
        """Separate planner protocol messages from peer messages.

        Returns (planner_msgs, peer_msgs).
        """
        planner_msgs = []
        peer_msgs = []
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else msg
            if (isinstance(content, dict)
                    and content.get('type') in self._PLANNER_MSG_TYPES):
                planner_msgs.append(msg)
            else:
                peer_msgs.append(msg)
        return planner_msgs, peer_msgs

    def _handle_planner_messages(self, msgs: list) -> None:
        """Process planner protocol messages (task assignments, answers)."""
        for msg in msgs:
            content = msg.content if hasattr(msg, 'content') else msg
            if not isinstance(content, dict):
                continue

            msg_type = content.get('type', '')

            if msg_type == 'task_assignment':
                task = content.get('task', '')
                if task:
                    self.set_current_task(task)
                    self._received_planner_task = True

                # Apply manual plan if included
                plan = content.get('plan')
                if plan:
                    self.set_manual_task_decomposition(plan)

                # Update SharedMemory with all task assignments
                all_assignments = content.get('task_assignments')
                if all_assignments and self.shared_memory:
                    self.shared_memory.update(SM_TASK_ASSIGNMENTS, all_assignments)

            elif msg_type == 'planner_answer':
                answer = content.get('answer', '')
                if answer:
                    self.memory.update('planner_answer', answer)

    def ask_planner(self, question: str, context: Optional[Dict] = None) -> None:
        """Send a question to the planner agent via MATRX message."""
        self.send_message(Message(
            content={
                'type': 'planner_question',
                'question': question,
                'context': context or {},
            },
            from_id=self.agent_id,
            to_id=self._planner_agent_id,
        ))

    def _run_infra(self, filtered_state: State) -> Optional[Tuple[str, Dict]]:
        """Handle infrastructure concerns only (no LLM polling).
        """
        self._check_carry_success()

        autopilot = self._handle_carry_autopilot()
        if autopilot is not None:
            return autopilot

        nav = self._handle_navigation_tick()
        if nav is not None:
            return nav

        return None

    def _check_carry_success(self) -> None:
        """Detect successful CarryObjectTogether and enter autopilot to drop zone.

        Uses ``previous_action_result.succeeded`` (set by MATRX engine on
        ArtificialBrain) instead of polling nearby objects.
        """
        if self._carry_autopilot is not None:
            return
        if (self.previous_action == _CarryObjectTogether.__name__
                and self.previous_action_result is not None
                and self.previous_action_result.succeeded):
            carrying = self.WORLD_STATE.get('agent', {}).get('carrying', [])
            if not carrying:
                return
            obj_id = carrying[0]
            dest = self.env_info.drop_zone
            print(f"[{self.agent_id}] Carry succeeded — entering autopilot to drop zone")
            self._carry_autopilot = {
                'victim_id': obj_id,
                'destination': dest,
                'role': 'carrier',
            }
            if self.shared_memory:
                self.shared_memory.update(SM_CARRY_AUTOPILOT, {
                    'carrier': self.agent_id,
                    'victim_id': obj_id,
                    'destination': dest,
                })

    def _handle_carry_autopilot(self) -> Optional[Tuple[str, Dict]]:
        """Handle autopilot navigation to drop zone after a cooperative carry.

        Both the carrier (who has the victim in inventory) and the partner
        are forced to navigate to the drop zone simultaneously.  When the
        carrier arrives it returns a DropObject action; the partner simply
        idles once it arrives.
        """
        # ── Carrier agent path ────────────────────────────────────────────
        if self._carry_autopilot and self._carry_autopilot['role'] == 'carrier':
            if self._nav_target is not None:
                return None  # let _handle_navigation_tick do the movement
            # Arrived at drop zone — drop the victim
            victim_id = self._carry_autopilot['victim_id']
            print(f'[{self.agent_id}] Autopilot: arrived at drop zone, dropping {victim_id}')
            self.memory.update('action', {
                'action': 'CarryObjectTogether',
                'result': 'delivered',
                'victim_id': victim_id,
            })
            if self.shared_memory:
                # Record the rescue so plan() can report it via _get_rescued_victims().
                # Use list-append semantics (not overwrite) so concurrent agents don't
                # clobber each other's entries; the shared memory lock protects the
                # read-modify-write sequence via retrieve() + update().
                rescued = self.shared_memory.retrieve('rescued_victims') or []
                if not any(v['victim_id'] == victim_id for v in rescued):
                    rescued = rescued + [{
                        'victim_id': victim_id,
                        'tick': self._tick_count,
                        'agent': self.agent_id,
                        'method': 'cooperative',
                    }]
                    self.shared_memory.update('rescued_victims', rescued)
                    print(
                        f'[{self.agent_id}] Recorded cooperative rescue of '
                        f'{victim_id} (total rescued: {len(rescued)})'
                    )
            self._carry_autopilot = None
            if self.shared_memory:
                self.shared_memory.update(SM_CARRY_AUTOPILOT, None)
            return _DropObject.__name__, {'object_id': victim_id}

        if self._carry_autopilot and self._carry_autopilot['role'] == 'partner':
            # Check if carrier already finished (signal cleared)
            if self.shared_memory:
                ap = self.shared_memory.retrieve(SM_CARRY_AUTOPILOT)
                if ap is None:
                    # Carrier done — clear own autopilot
                    print(f'[{self.agent_id}] Autopilot: carrier finished, resuming pipeline')
                    self._carry_autopilot = None
                    return self._idle()
            if self._nav_target is not None:
                return None  # still navigating
            # Arrived at drop zone — wait for carrier to finish
            print(f'[{self.agent_id}] Autopilot: partner arrived at drop zone, waiting')
            self.memory.update('carry_participation', {
                'victim_id': self._carry_autopilot['victim_id'],
                'status': 'delivered',
            })
            self._carry_autopilot = None
            return self._idle()

        # ── Partner detection: check if another agent initiated autopilot ─
        if not self.shared_memory:
            return None
        ap = self.shared_memory.retrieve(SM_CARRY_AUTOPILOT)
        if not ap or ap.get('carrier') == self.agent_id:
            return None

        # Another agent is carrying to drop zone — join as partner
        dest = tuple(ap['destination'])
        if _chebyshev_distance(self.agent_location, dest) <= 1:
            return None  # Already at drop zone

        print(f'[{self.agent_id}] Autopilot: joining partner carry to drop zone {dest}')
        self._carry_autopilot = {
            'victim_id': ap['victim_id'],
            'destination': dest,
            'role': 'partner',
        }
        self._pending_future = None  # cancel any pending LLM call
        nav_target, action = navigate_to(dest, navigator=self._navigator, state_tracker=self._state_tracker)
        self._nav_target = nav_target
        return action

    def _handle_navigation_tick(self) -> Optional[Tuple[str, Dict]]:
        """Continue A* navigation if a target is set."""
        if self._nav_target is None:
            return None
        move = self._navigator.get_move_action(self._state_tracker)
        if move is not None:
            return move, {}
        # Destination reached
        self._nav_target = None
        return None
    
    def _validate_action(
        self, name: str, args: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict]]:
        """Validate an action before dispatching to MATRX.
        """
        result = self._validator.validate(name, args, self.WORLD_STATE, self.teammates)
        if result.valid:
            return None
        self.memory.update("action_failure", result.feedback)
        if self.metrics:
            self.metrics.record_validation_failure(self._tick_count, name, result.feedback)
        return self._idle()

    # ── LLM submission ────────────────────────────────────────────────────

    def call_llm(
        self,
        messages: List[Dict],
        tools: Optional[List] = None,
        tool_choice: str = 'auto',
    ) -> None:
        """Submit an async LLM call."""
        if self.metrics:
            self.metrics.record_llm_call_start()
        self._pending_future = submit_llm_call(
            llm_model=self._llm_model,
            messages=messages,
            max_token_num=MAX_NR_TOKENS,
            temperature=TEMPERATURE,
            tools=tools,
            tool_choice=tool_choice if tools else 'none',
            api_base=self._api_base,
        )

    # ── Utilities ─────────────────────────────────────────────────────────

    @property
    def agent_location(self) -> Tuple[int, int]:
        """Current agent (x, y) from world state."""
        if isinstance(self.WORLD_STATE, dict):
            return tuple(self.WORLD_STATE.get('agent', {}).get('location', (0, 0)))
        return (0, 0)

    def _idle(self, reason: str = 'idle') -> Tuple[str, Dict]:
        """Convenience: return an Idle action and record it in metrics."""
        if self.metrics:
            self.metrics.record_idle(self._tick_count, reason)
        return _Idle.__name__, {'duration_in_ticks': 1}