"""
AgentMetricsTracker — lightweight per-agent metrics accumulator.

Pure data class with zero dependencies on agent internals.
Any agent type can instantiate and call record_*() methods.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple


COOPERATIVE_ACTIONS = frozenset({
    'CarryObjectTogether', 'DropObjectTogether', 'RemoveObjectTogether',
})


class AgentMetricsTracker:

    def __init__(self, agent_id: str = '') -> None:
        self.agent_id = agent_id

        # Spatial
        self.cells_visited: Set[Tuple[int, int]] = set()
        self.location_trace: List[Tuple[int, int, int]] = []  # (tick, x, y)

        # Actions
        self.action_log: List[Dict[str, Any]] = []
        self.cooperative_actions: List[Dict[str, Any]] = []
        self.idle_ticks: int = 0
        self.llm_wait_ticks: int = 0

        # Communication
        self.messages_sent: List[Dict[str, Any]] = []
        self.messages_received: List[Dict[str, Any]] = []
        self.help_requests_sent: int = 0
        self.help_requests_received: int = 0
        self.help_responses_sent: int = 0

        # Victims
        self.victims_found: List[Dict[str, Any]] = []

        # LLM performance
        self.llm_call_count: int = 0
        self._llm_call_start: float = 0.0
        self.llm_latencies: List[float] = []

        # Validation
        self.validation_failures: int = 0
        self.validation_failure_log: List[Dict[str, Any]] = []

    # ── Recording methods ────────────────────────────────────────────────

    def record_location(self, tick: int, x: int, y: int) -> None:
        self.location_trace.append((tick, x, y))
        self.cells_visited.add((x, y))

    def record_action(
        self, tick: int, action_name: str, args: Dict, location: Tuple[int, int],
    ) -> None:
        entry = {
            'tick': tick,
            'action_name': action_name,
            'args': args,
            'location': location,
        }
        self.action_log.append(entry)
        if action_name in COOPERATIVE_ACTIONS:
            self.cooperative_actions.append({
                'tick': tick,
                'action': action_name,
                'partner': args.get('human_name', ''),
                'object_id': args.get('object_id', ''),
            })

    def record_message_sent(
        self, tick: int, to: str, message_type: str, text: str,
    ) -> None:
        self.messages_sent.append({
            'tick': tick, 'to': to, 'message_type': message_type, 'text': text,
        })
        if message_type == 'ask_help':
            self.help_requests_sent += 1
        elif message_type == 'help':
            self.help_responses_sent += 1

    def record_message_received(
        self, tick: int, from_id: str, message_type: str, text: str,
    ) -> None:
        self.messages_received.append({
            'tick': tick, 'from': from_id, 'message_type': message_type, 'text': text,
        })
        if message_type == 'ask_help':
            self.help_requests_received += 1

    def record_idle(self, tick: int, reason: str = 'idle') -> None:
        if reason == 'llm_wait':
            self.llm_wait_ticks += 1
        else:
            self.idle_ticks += 1

    def record_llm_call_start(self) -> None:
        self._llm_call_start = time.monotonic()
        self.llm_call_count += 1

    def record_llm_call_end(self) -> None:
        if self._llm_call_start > 0:
            latency = time.monotonic() - self._llm_call_start
            self.llm_latencies.append(latency)
            self._llm_call_start = 0.0

    def record_validation_failure(
        self, tick: int, action_name: str, feedback: str,
    ) -> None:
        self.validation_failures += 1
        self.validation_failure_log.append({
            'tick': tick, 'action_name': action_name, 'feedback': feedback,
        })

    def record_victim_found(
        self, tick: int, victim_id: str, severity: str, location: Tuple[int, int],
    ) -> None:
        self.victims_found.append({
            'tick': tick, 'victim_id': victim_id,
            'severity': severity, 'location': location,
        })

    # ── Export ────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'cells_visited': list(self.cells_visited),
            'location_trace': self.location_trace,
            'action_log': self.action_log,
            'cooperative_actions': self.cooperative_actions,
            'idle_ticks': self.idle_ticks,
            'llm_wait_ticks': self.llm_wait_ticks,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'help_requests_sent': self.help_requests_sent,
            'help_requests_received': self.help_requests_received,
            'help_responses_sent': self.help_responses_sent,
            'victims_found': self.victims_found,
            'llm_call_count': self.llm_call_count,
            'llm_latencies': self.llm_latencies,
            'validation_failures': self.validation_failures,
            'validation_failure_log': self.validation_failure_log,
        }
