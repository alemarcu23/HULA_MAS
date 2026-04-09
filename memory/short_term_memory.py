"""
Short-term memory module with LLM-based summarization.

Extends BaseMemory with:
- Structured dict storage (each entry has a 'type' key)
- Configurable capacity limit
- Automatic compression of oldest entries via Ollama LLM when full
- Compact serialization for token-efficient LLM prompt injection
"""

import json
import logging
from typing import Any, Dict, List, Optional

from memory.base_memory import BaseMemory

logger = logging.getLogger('ShortTermMemory')


class ShortTermMemory(BaseMemory):
    """
    Short-term memory that stores structured dict entries and automatically
    compresses the oldest entries when the capacity limit is reached.

    Each entry must be a dict with at least a ``type`` key (e.g.
    ``victim_found``, ``room_explored``).  Compression uses an Ollama LLM
    to merge old entries, keeping the storage within ``memory_limit``.
    """

    def __init__(self, memory_limit: int = 20, llm_model: str = 'qwen3:8b', api_url: str = None) -> None:
        """
        Args:
            memory_limit: Maximum number of entries before compression kicks in.
            api_url: Ollama base URL for this agent (None = default).
        """
        super().__init__()
        self.memory_limit: int = memory_limit
        self.llm_model: str = llm_model
        self._api_url: str = api_url
        self.storage: List[Dict[str, Any]] = []

    def update(self, key: str, information: Dict[str, Any]) -> None:
        """
        Update memory with new information.

        Args:
            key (str): Only here to keep the signature consistent with SharedMemory.
            information (Dict[str, Union[str, Any]]): Information to store.
        """
        if len(self.storage) >= self.memory_limit:
            self._compress_oldest()
        self.storage.append(information)

    def get_compact_str(self) -> str:
        """
        Compact JSON serialization of all entries.

        Uses minimal separators (no spaces) for token efficiency when
        injected into an LLM prompt.
        """
        return json.dumps(self.storage, default=str, separators=(',', ':'))

    def retrieve_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recently added entry, or ``None``."""
        return self.storage[-1] if self.storage else None

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of all entries."""
        return self.storage.copy()

    def _compress_oldest(self) -> None:
        """Pop up to 2 oldest entries and replace with a compressed summary."""
        count = min(2, len(self.storage))
        if count == 0:
            return
        oldest = [self.storage.pop(0) for _ in range(count)]
        compressed = self._summarize_entries(oldest)
        self.storage.insert(0, {
            'type': 'old_memory_summary',
            'entries': compressed,
        })

    def _summarize_entries(self, memory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from agents1.async_model_prompting import call_llm_sync
        from engine.parsing_utils import parse_json_response

        system_prompt = "You are a helpful assistant that can concisely summarize the following json format content which is listed in temporally sequential order.\n"

        user_prompt = (
            "Summarize these entries into a single entry and return a valid JSON {\"entry\": [...]}. The entries are:\n"
            f"{json.dumps(memory, default=str)}\n\n"        )

        try:
            response = call_llm_sync(
                llm_model=self.llm_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_base=self._api_url,
            )
            parsed = parse_json_response(response)
            if parsed and 'entry' in parsed:
                entry = parsed['entry']
                if isinstance(entry, dict):
                    entry = [entry]   # wrap single dict into a list
                if isinstance(entry, list):
                    return entry
        except Exception as exc:
            logger.warning(f"LLM summarization failed ({exc}), keeping originals")

        return memory
