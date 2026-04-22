"""
Token-Oriented Object Notation (TOON) encoding utility.

TOON is a compact, LLM-native serialization format. This module provides a
self-contained encoder that produces indented key: value lines with array
notation, closely matching the format shown in few-shot examples and already
familiar to the LLM.

Format summary:
  - Dicts:   key: value  (one per line, indented for nested)
  - Lists:   name[N]{field1,field2,...}: item1 | item2 | ...
             where N is the count and the field hint is auto-derived
  - Scalars: printed as-is

Public API:
  to_toon(data) -> str
      Encode a dict or list to TOON.
  to_toon_sectioned(sections: dict) -> str
      Emit each top-level key as a labeled == SECTION == header,
      then encode its value with to_toon().
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _field_hint(lst: list) -> str:
    """Derive a {field1,field2,...} hint from the first dict in a list."""
    if lst and isinstance(lst[0], dict):
        keys = list(lst[0].keys())
        if keys:
            return '{' + ','.join(str(k) for k in keys) + '}'
    return ''


def _encode_value(val: Any, indent: int = 0) -> str:
    """Recursively encode a value to TOON string."""
    pad = '  ' * indent
    if isinstance(val, dict):
        if not val:
            return '{}'
        lines = []
        for k, v in val.items():
            encoded = _encode_value(v, indent + 1)
            if '\n' in encoded:
                lines.append(f"{pad}  {k}:\n{encoded}")
            else:
                lines.append(f"{pad}  {k}: {encoded.strip()}")
        return '\n'.join(lines)
    elif isinstance(val, list):
        if not val:
            return '[]'
        hint = _field_hint(val)
        items = []
        for item in val:
            if isinstance(item, dict):
                # Render dict items as pipe-separated key=value pairs
                parts = []
                for ik, iv in item.items():
                    parts.append(f"{ik}={_scalar(iv)}")
                items.append(' | '.join(parts))
            else:
                items.append(_scalar(item))
        count = len(val)
        prefix = f"[{count}]{hint}"
        # Short lists on one line; long lists one item per line
        one_line = f"{prefix}: {', '.join(items)}"
        if len(one_line) <= 120:
            return one_line
        sep = f"\n{pad}  - "
        return f"{prefix}:{sep}" + sep.join(items)
    else:
        return _scalar(val)


def _scalar(val: Any) -> str:
    if val is None:
        return 'null'
    if isinstance(val, bool):
        return 'true' if val else 'false'
    return str(val)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_toon(data: Any) -> str:
    """Encode a Python dict or list to TOON format for token-efficient LLM prompts."""
    try:
        if isinstance(data, (dict, list)):
            result = _encode_value(data)
            return result
    except Exception:
        pass  # Fall back to JSON on any encoding error
    return json.dumps(data, default=str, indent=2)


def to_toon_sectioned(sections: dict) -> str:
    """Emit each top-level key as a labeled == SECTION == header, then encode its value.

    Example:
        to_toon_sectioned({'OBSERVATION': obs_dict, 'MEMORY': mem_list})
    produces:
        == OBSERVATION ==
        ...toon-encoded obs_dict...

        == MEMORY ==
        ...toon-encoded mem_list...
    """
    parts = []
    for key, value in sections.items():
        header = f"== {str(key).upper()} =="
        body = to_toon(value)
        parts.append(f"{header}\n{body}")
    return '\n\n'.join(parts)
