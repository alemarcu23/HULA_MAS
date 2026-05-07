"""
CoordinationModule — rotating coordinator assigns high-level tasks to all agents.

Triggered at simulation start and after each victim rescue.  One agent per
round (round-robin) submits an LLM call to formulate task assignments; others
idle until they receive their assignment via MATRX message or SharedMemory.
"""

import json
import logging
import re
import ast
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger('CoordinationModule')

# ── Constants ─────────────────────────────────────────────────────────────────

COORDINATION_TIMEOUT_TICKS = 80      # non-coordinator gives up waiting after this
COORDINATOR_LLM_TIMEOUT_TICKS = 60  # coordinator abandons LLM call after this
HISTORY_WINDOW = 5                   # past rounds shown in coordinator prompt
MSG_TYPE_COORD_ASSIGNMENT = 'coord_assignment'
# Ticks to wait before the first coordination round so all agents can register
# their IDs in SharedMemory.  By this tick every agent has run at least once.
COORDINATION_REGISTRATION_TICKS = 10


# ── Module ────────────────────────────────────────────────────────────────────

class CoordinationModule:
    """Manages one agent's participation in a coordination round."""

    def __init__(self, agent_id: str, llm_model: str, api_base: Optional[str] = None) -> None:
        self.agent_id = agent_id
        self._llm_model = llm_model
        self._api_base = api_base
        # How many rescued victims we saw at the last check — used for trigger detection
        self._last_seen_rescued_count: int = 0

    # ── Trigger detection ─────────────────────────────────────────────────

    def should_trigger(
        self,
        rescued_victims: List[Dict],
        coord_state: Optional[Dict],
        is_first_coord_cycle: bool,
        tick: int = 0,
    ) -> Tuple[bool, str]:
        """Return (should_trigger, trigger_reason).

        Triggers when:
        - is_first_coord_cycle is True AND tick >= COORDINATION_REGISTRATION_TICKS, OR
        - a new victim rescue has been detected.

        Does NOT trigger if a round is currently in_progress.
        The startup delay ensures all agents have registered their IDs in
        SharedMemory before the first coordinator is elected.
        """
        if coord_state is not None and coord_state.get('status') == 'in_progress':
            return False, ''

        if is_first_coord_cycle:
            if tick < COORDINATION_REGISTRATION_TICKS:
                return False, ''  # wait for all agents to register
            return True, 'start'

        current_count = len(rescued_victims)
        if current_count > self._last_seen_rescued_count:
            return True, 'rescue'

        return False, ''

    # ── Coordinator election ──────────────────────────────────────────────

    @staticmethod
    def elect_coordinator(round_id: int, agent_ids: List[str]) -> str:
        """Pure round-robin election; agent_ids must be sorted for determinism."""
        return sorted(agent_ids)[round_id % len(agent_ids)]

    def am_i_coordinator(self, round_id: int, agent_ids: List[str]) -> bool:
        return self.elect_coordinator(round_id, agent_ids) == self.agent_id

    # ── Prompt building ───────────────────────────────────────────────────

    def build_coordinator_prompt(
        self,
        tick: int,
        trigger: str,
        agent_ids: List[str],
        agent_roles: Dict[str, str],       # {agent_id: role}
        agent_capabilities: Dict[str, Dict],  # {agent_id: capability_dict}
        area_summaries: List[Dict],
        known_victims: List[Dict],
        rescued_victims: List[Dict],
        history: List[Dict],               # last HISTORY_WINDOW GLOBAL_PLAN_ASSIGNMENTS entries
        own_recent_memory: List[Any],
    ) -> List[Dict]:
        """Build an OpenAI-format message list for the coordinator LLM call."""

        system_prompt = (
            'You are the coordinating agent for a Search and Rescue team.\n'
            'Your job: assign exactly ONE high-level task to each agent to eliminate duplicate work.\n\n'
            'Rules:\n'
            '- Assign each agent exactly one task.\n'
            '- Match tasks to agent roles and capabilities as well as possible, while also keeping in mind the task needs at the current time.'
            '(scouts - unexplored areas, rescuers - known victims, heavy_lifters - remove obstacles).\n'
            '- Consult coordination_history: if a task from a prior round was likely not completed '
            '(area still uncovered, victim still unrescued), consider reassigning it.\n'
            '- NEVER assign a task to rescue victims already in rescued_victims.\n'
            '- Cover all unexplored areas across the team — avoid sending two agents to the same area.\n\n'
            'Output ONLY valid JSON:\n'
            '{"assignments": {"<agent_id>": "<task string>", ...}}\n\n'
            'Task format examples:\n'
            '- "Search area 3 for victims"\n'
            '- "Pick up mildly_injured_woman at [4, 7] and carry to drop zone"\n'
            '- "Clear obstacle blocking area 5 entrance"\n'
            '- "Cooperate with RescueBot1 to carry critically_injured_man at [9, 3]"\n'
        )

        agents_info = []
        for aid in sorted(agent_ids):
            agents_info.append({
                'id': aid,
                'role': agent_roles.get(aid, 'unknown'),
                'capabilities': agent_capabilities.get(aid, {}),
            })

        history_formatted = []
        for entry in history:
            rescued_then = entry.get('rescued_count_at_formation', 0)
            rescued_now = len(rescued_victims)
            history_formatted.append({
                'round_id': entry.get('round_id'),
                'formed_by': entry.get('formed_by'),
                'formed_at_tick': entry.get('formed_at_tick'),
                'trigger': entry.get('trigger'),
                'assignments': entry.get('assignments', {}),
                'agent_roles_at_formation': entry.get('agent_roles_at_formation', {}),
                'rescues_since_this_round': rescued_now - rescued_then,
                'area_coverage_at_formation': entry.get('area_coverage_at_formation', []),
            })

        user_content = {
            'tick': tick,
            'trigger': trigger,
            'you': self.agent_id,
            'agents': agents_info,
            'area_coverage': area_summaries,
            'known_unrescued_victims': known_victims,
            'rescued_victims': [
                {'victim_id': v.get('victim_id'), 'tick': v.get('tick'), 'agent': v.get('agent')}
                for v in rescued_victims
            ],
            'coordination_history': history_formatted,
            'your_recent_memory': own_recent_memory[-10:] if own_recent_memory else [],
        }

        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': json.dumps(user_content, default=str)},
        ]

    # ── Response parsing ──────────────────────────────────────────────────

    def parse_coordinator_response(
        self,
        text: str,
        agent_ids: List[str],
    ) -> Optional[Dict[str, str]]:
        """Extract {"assignments": {...}} from LLM response.

        Fills in any missing agent IDs with a safe fallback task.
        Returns None only if parsing fails entirely.
        """
        parsed = _extract_json(text)
        assignments: Optional[Dict[str, str]] = None

        if parsed and isinstance(parsed, dict):
            if 'assignments' in parsed and isinstance(parsed['assignments'], dict):
                assignments = parsed['assignments']
            elif all(k in agent_ids for k in parsed):
                # LLM returned the dict directly without wrapper
                assignments = parsed

        if assignments is None:
            logger.warning('[%s] Could not parse coordinator response: %s', self.agent_id, text[:200])
            return None

        # Fill missing agents
        for aid in agent_ids:
            if aid not in assignments:
                logger.warning('[%s] Missing assignment for %s — using fallback', self.agent_id, aid)
                assignments[aid] = 'Explore unmapped areas'

        return assignments

    # ── Assignment message ingestion ──────────────────────────────────────

    def ingest_assignment_message(self, received_messages: List) -> Optional[str]:
        """Scan received MATRX messages for a coord_assignment addressed to self.

        Returns the task string if found, None otherwise.
        """
        for msg in received_messages:
            content = getattr(msg, 'content', {})
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    continue
            if not isinstance(content, dict):
                continue
            if content.get('message_type') != MSG_TYPE_COORD_ASSIGNMENT:
                continue
            to_id = getattr(msg, 'to_id', None)
            if to_id is None or to_id == self.agent_id:
                task = content.get('task', '')
                if task:
                    return task
        return None

    # ── Completion check ──────────────────────────────────────────────────

    def is_coordination_complete(
        self,
        coord_state: Optional[Dict],
        tick: int,
        trigger_tick: int,
    ) -> Tuple[bool, Optional[str]]:
        """Check if coordination is done for non-coordinator agents.

        Returns (is_complete, assigned_task_or_None).
        """
        # Timeout path
        if (tick - trigger_tick) >= COORDINATION_TIMEOUT_TICKS:
            return True, None

        if coord_state is None:
            return False, None

        if coord_state.get('status') in ('complete', 'timed_out'):
            assignments = coord_state.get('assignments') or {}
            task = assignments.get(self.agent_id)
            return True, task

        return False, None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict]:
    """Try to extract a JSON object from free-form LLM text."""
    if not text:
        return None
    m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
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
