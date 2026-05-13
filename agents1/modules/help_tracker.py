"""Help-request response tracking for SAR agents.

Tracks the lifecycle of ask_help requests: who sent one, who replied,
whether to issue an abandon directive if nobody accepted.
"""

from typing import Any, Dict, List, Optional

HELP_RESPONSE_TIMEOUT_TICKS = 500
_KEY_PREFIX = 'help_replies::'


class HelpTracker:
    """Wraps SharedMemory operations for ask_help request/response bookkeeping.

    All state is stored in SharedMemory so multiple agents can read it.
    """

    def __init__(self, agent_id: str, shared_memory: Any) -> None:
        self.agent_id = agent_id
        self._sm = shared_memory

    def _key(self, requester_id: str) -> str:
        return f'{_KEY_PREFIX}{requester_id}'

    def register_sent(self, tick: int, expected_responders: int) -> None:
        """Record that this agent just sent an ask_help; prime the reply counter."""
        if not self._sm:
            return
        self._sm.update(self._key(self.agent_id), {
            'tick': tick,
            'expected': max(expected_responders, 0),
            'replies': {},
        })

    def record_reply(self, requester_id: str, reply: str) -> None:
        """Record this agent's yes/no reply against the requester's counter."""
        if not self._sm:
            return
        key = self._key(requester_id)
        entry = self._sm.retrieve(key)
        if not entry:
            return
        replies = dict(entry.get('replies', {}))
        replies[self.agent_id] = reply
        self._sm.update(key, {**entry, 'replies': replies})

    def clear_request(self) -> None:
        """Clear this agent's pending ask_help entry from SharedMemory."""
        if self._sm:
            self._sm.update(self._key(self.agent_id), None)

    def check_outcome(self, tick: int) -> Optional[str]:
        """Return an abandon directive if this agent's ask_help is resolved, else None.

        Resolution: all expected teammates replied "no", OR the request timed out.
        A single "yes" clears the counter and returns None (help is on the way).
        """
        if not self._sm:
            return None
        key = self._key(self.agent_id)
        entry = self._sm.retrieve(key)
        if not entry:
            return None

        replies = entry.get('replies', {}) or {}
        expected = int(entry.get('expected', 0))
        elapsed = tick - int(entry.get('tick', tick))
        timed_out = elapsed >= HELP_RESPONSE_TIMEOUT_TICKS
        all_replied = len(replies) >= expected and expected > 0

        if any(r == 'yes' for r in replies.values()):
            self._sm.update(key, None)
            return None

        if not (all_replied or timed_out):
            return None

        if replies and all(r == 'no' for r in replies.values()):
            reason = 'All teammates declined your help request.'
        elif timed_out and not replies:
            reason = (
                f'No teammate responded to your help request within '
                f'{HELP_RESPONSE_TIMEOUT_TICKS} ticks.'
            )
        elif timed_out:
            reason = (
                f'Only {len(replies)}/{expected} teammates responded '
                f'within {HELP_RESPONSE_TIMEOUT_TICKS} ticks, and none accepted.'
            )
        else:
            self._sm.update(key, None)
            return None

        self._sm.update(key, None)
        return f'{reason} You MUST abandon this task and choose a different objective.'


def count_eligible_responders(teammates: List[Dict], self_id: str) -> int:
    """Count teammates that could respond to an ask_help (all except self)."""
    return sum(1 for t in (teammates or []) if t.get('object_id') != self_id)
