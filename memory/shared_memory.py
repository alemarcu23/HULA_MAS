"""
Shared memory module allowing agents to communicate.
"""

from threading import Lock
from typing import Any, Dict, Optional


class SharedMemory:
    """
    Shared memory accessible by multiple agents.
    """

    def __init__(self) -> None:
        """
        Initialize the shared memory with thread-safe access.
        """
        self.storage: Dict[str, Any] = {}
        self.lock = Lock()

    def update(self, key: str, information: Any) -> None:
        """
        Update shared memory with new information.

        Args:
            key (str): Key under which to store the information.
            information (Any): Information to store.
        """
        with self.lock:
            self.storage[key] = information

    def retrieve(self, key: str) -> Any:
        """
        Retrieve information from shared memory.

        Args:
            key (str): Key of the information to retrieve.

        Returns:
            Any: The retrieved information, or None if key does not exist.
        """
        with self.lock:
            return self.storage.get(key)

    def retrieve_all(self) -> Dict[str, Any]:
        """
        Retrieve all information from shared memory.

        Returns:
            Dict[str, Any]: A copy of all stored information.
        """
        with self.lock:
            return self.storage.copy()

    def increment(self, key: str, default: int = 0) -> int:
        """Atomically increment an integer key and return the new value."""
        with self.lock:
            val = self.storage.get(key, default) + 1
            self.storage[key] = val
            return val

    def append_to_list(self, key: str, item: Any) -> None:
        """Atomically append an item to a list key, creating it if absent."""
        with self.lock:
            lst = self.storage.get(key, [])
            self.storage[key] = lst + [item]

    def add_to_set(self, key: str, value: Any) -> None:
        """Atomically add a value to a list-backed set (no duplicates)."""
        with self.lock:
            lst = self.storage.get(key, [])
            if value not in lst:
                self.storage[key] = lst + [value]

    def try_start_rendezvous(
        self,
        key: str,
        entry: Dict[str, Any],
        dedupe_key: str,
    ) -> bool:
        """Atomic check-and-set for rendezvous keys.

        Writes ``entry`` under ``key`` only when no existing entry has the
        same ``dedupe_key`` value. Returns True if this call won the race.
        """
        with self.lock:
            existing = self.storage.get(key)
            if (existing is not None
                    and existing.get(dedupe_key) == entry.get(dedupe_key)):
                return False
            self.storage[key] = entry
            return True

    def add_unique_record(
        self,
        key: str,
        record: Dict[str, Any],
        dedupe_field: str,
    ) -> bool:
        """Atomically append a record to a list-backed key, deduped by ``dedupe_field``.

        Returns True if the record was appended; False if a record with the
        same dedupe field value already existed.
        """
        with self.lock:
            lst = self.storage.get(key, [])
            target = record.get(dedupe_field)
            if any(r.get(dedupe_field) == target for r in lst):
                return False
            self.storage[key] = lst + [record]
            return True

    def clear_if_initiator(self, key: str, agent_id: str) -> bool:
        """Atomically clear an SM key only if its 'initiator' matches ``agent_id``.

        Returns True if the entry was cleared.
        """
        with self.lock:
            existing = self.storage.get(key)
            if existing is not None and existing.get('initiator') == agent_id:
                self.storage[key] = None
                return True
            return False

    def try_start_coordination(
        self,
        round_id: int,
        coordinator_id: str,
        trigger: str,
        trigger_tick: int,
        rescued_count: int,
    ) -> bool:
        """Atomic check-and-set to start a new coordination round.

        Writes coordination_state only when no round is currently active
        (state is None or status is 'complete'/'timed_out').
        Returns True if this call won the race, False if another agent beat us.
        """
        with self.lock:
            existing = self.storage.get('coordination_state')
            if existing is not None and existing.get('status') == 'in_progress':
                return False
            self.storage['coordination_round_counter'] = round_id
            self.storage['coordination_state'] = {
                'round_id': round_id,
                'coordinator': coordinator_id,
                'status': 'in_progress',
                'trigger': trigger,
                'trigger_tick': trigger_tick,
                'rescued_count_at_trigger': rescued_count,
                'started_tick': trigger_tick,
                'completed_tick': None,
                'assignments': None,
                'acknowledged_by': [],
            }
            return True
