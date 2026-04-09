"""
Parsing utilities relocated from engine/llm_utils.py.

Contains:
- parse_json_response(): 3-stage JSON extractor for LLM responses
- load_few_shot(): YAML few-shot example loader with caching
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

logger = logging.getLogger('parsing_utils')

# ---------------------------------------------------------------------------
# Few-shot loader
# ---------------------------------------------------------------------------

_FEW_SHOT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'few_shot_examples.yaml'
)
_few_shot_cache: Optional[Dict] = None


def load_few_shot(key: str) -> List[Dict]:
    """Return few-shot messages for *key* from few_shot_examples.yaml.

    Returns a list of {"role": "user"/"assistant", "content": str} dicts
    ready to be injected between the system message and the real user prompt.
    Returns [] if the key is absent, empty, or the file is missing.
    """
    global _few_shot_cache
    if _few_shot_cache is None:
        try:
            import yaml
            with open(_FEW_SHOT_FILE, 'r') as f:
                _few_shot_cache = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(
                "few_shot_examples.yaml not found at %s", _FEW_SHOT_FILE
            )
            _few_shot_cache = {}
        except Exception as e:
            logger.warning(
                "Failed to load few_shot_examples.yaml: %s", e
            )
            _few_shot_cache = {}

    examples = _few_shot_cache.get(key) or []
    messages = []
    for ex in examples:
        if isinstance(ex, dict) and 'user' in ex and 'assistant' in ex:
            messages.append({"role": "user", "content": ex['user'].strip()})
            messages.append({"role": "assistant", "content": ex['assistant'].strip()})
    return messages


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> Optional[dict]:
    """
    Extract JSON from LLM response text.

    Tries in order:
    1. ```json ... ``` fenced code block
    2. Raw JSON object (strict json.loads)
    3. Python dict literal via ast.literal_eval

    Args:
        text: Raw LLM response text

    Returns:
        Parsed dict, or None if parsing fails
    """
    import ast

    if not text:
        return None

    # 1. Fenced ```json block
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Isolate the outermost { ... } span for attempts 2 & 3
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1]

    # 2. Strict JSON (double-quoted keys/values)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 3. Python dict literal — handles single-quoted strings from the LLM
    try:
        result = ast.literal_eval(candidate)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    logger.warning("Failed to parse JSON/dict from LLM response: %s", text[:200])
    return None
