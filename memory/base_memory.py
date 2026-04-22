"""
Base memory module for agents.
"""

import json
from collections import deque
from typing import Any, List, Optional


class BaseMemory:
    """Base class for agent memory modules."""

    def __init__(self, maxlen: int = 200) -> None:
        """Initialize the memory module.

        Args:
            maxlen: Maximum number of entries to retain. Oldest entries
                    are automatically discarded when the limit is reached.
        """
        self.storage: deque = deque(maxlen=maxlen)

    def update(self, key: str, information: Any, tick: Optional[int] = None) -> None:
        """Update memory with new information.

        Args:
            key:         Only here to keep the signature consistent with SharedMemory.
            information: Information to store.
            tick:        Current simulation tick. When provided and information is a
                         dict without a 'tick' key, the tick is stamped into a copy
                         of the dict before storing.
        """
        # Stamp tick into dict entries that don't already have one
        if tick is not None and isinstance(information, dict) and 'tick' not in information:
            information = {**information, 'tick': tick}

        # Dedup: skip if information matches any of the last 3 entries
        recent = list(self.storage)[-3:]
        for last in recent:
            if isinstance(information, dict) and isinstance(last, dict):
                if information == last:
                    return
            elif information == last:
                return
        self.storage.append(information)

    def retrieve_latest(self) -> Any:
        """
        Retrieve the most recent information from memory.

        Returns:
            Any: The most recently stored information, or None if empty.
        """
        return self.storage[-1] if self.storage else None

    def retrieve_all(self) -> List[Any]:
        """Retrieve all stored information."""
        return list(self.storage)

    def retrieve_by_type(self, kinds: List[str]) -> List[Any]:
        """Retrieve only entries whose 'kind' field matches one of the given kinds.

        Entries that are not dicts or that lack a 'kind' key are excluded.

        Args:
            kinds: List of kind strings to include, e.g. ['action', 'victim_found'].

        Returns:
            Filtered list of matching entries in insertion order.
        """
        return [
            entry for entry in self.storage
            if isinstance(entry, dict) and entry.get('kind') in kinds
        ]

    def compress(self, threshold: int = 10, keep_recent: int = 5) -> None:
        """Collapse entries older than *keep_recent* into a single summary dict.

        Called before injecting memory into LLM prompts so the window stays
        compact even during long episodes.

        Args:
            threshold:   Minimum total entries before compression triggers.
            keep_recent: Number of most-recent entries to preserve verbatim.
        """
        entries = list(self.storage)
        if len(entries) <= threshold:
            return
        old, recent = entries[:-keep_recent], entries[-keep_recent:]
        action_counts: dict = {}
        notable = []
        for e in old:
            if not isinstance(e, dict):
                continue
            kind = e.get('kind') or e.get('entry_type') or e.get('action')
            if kind:
                action_counts[kind] = action_counts.get(kind, 0) + 1
            if e.get('kind') in ('loop_warning', 'critic_feedback', 'planned_task'):
                notable.append(e)
        summary: dict = {'kind': 'summary', 'compressed': len(old), 'action_counts': action_counts}
        if notable:
            summary['notable'] = notable[-3:]
        self.storage.clear()
        self.storage.append(summary)
        for e in recent:
            self.storage.append(e)

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the memory.

        Returns:
            str: Formatted string showing memory contents.
        """
        if not self.storage:
            return "Memory: Empty"

        items = [f"    {i}: {str(item)}" for i, item in enumerate(self.storage)]
        return "Memory Contents:\n" + "\n".join(items)

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the memory object.

        Returns:
            str: Technical string representation.
        """
        return f"BaseMemory(storage={self.storage})"

    def get_memory_str(self) -> str:
        """
        Get a string representation of the memory.

        Returns:
            str: String representation of the memory.
        """
        memory_str = " ".join([json.dumps(info) for info in self.storage])
        return memory_str
