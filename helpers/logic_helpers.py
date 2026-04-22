from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from helpers.object_types import _OBJECT_TYPES


@dataclass
class ValidationResult:
    """Result of a pre-dispatch action validation."""
    valid: bool
    feedback: str = ''


def _agent_location(ws: Dict) -> Optional[Tuple[int, int]]:
    """Extract agent (x, y) from world_state."""
    if not isinstance(ws, dict):
        return None
    loc = ws.get('agent', {}).get('location')
    if loc is None:
        return None
    return (int(loc[0]), int(loc[1]))

def _get_carrying(ws: Dict) -> List[str]:
    """Extract list of carried object IDs from world_state."""
    if not isinstance(ws, dict):
        return []
    return ws.get('agent', {}).get('carrying', [])

def _find_object(obj_id: str, ws: Dict) -> Optional[Dict]:
    """Find a specific object in current observations by ID."""
    for o in _get_adjacent(ws):
        if o.get('id') == obj_id:
            return o
    return None

def _is_teammate_adjacent(
        obj_id: str,
        ws: Dict,
        teammates: Set[Tuple[str, Tuple[int, int]]],
        partner_id: Optional[str] = None,
    ) -> bool:
        """Check if a teammate is within Chebyshev distance 1 of the object.

        When *partner_id* is given, only that specific teammate is checked.
        Otherwise any non-self teammate qualifies.
        """
        nearby = _get_adjacent(ws)
        obj_loc = None
        for o in nearby:
            if o.get('id') == obj_id:
                loc = o.get('location')
                if loc is not None:
                    obj_loc = (int(loc[0]), int(loc[1]))
                break
        if obj_loc is None:
            return False

        agent_loc = _agent_location(ws)
        for t_id, t_loc in teammates:
            if agent_loc and tuple(t_loc) == tuple(agent_loc):
                continue
            if partner_id and t_id != partner_id:
                continue
            if _chebyshev_distance(t_loc, obj_loc) <= 1:
                return True
        return False

def is_object_adjacent(
        args: Dict,
        ws: Dict,
        allowed_types: set) -> Optional['ValidationResult']:
    """Validate if object_id is adjacent to agent.

    Returns ``None`` when valid, or a ``ValidationResult`` with ``valid=False``
    and a descriptive feedback message when the check fails.
    """
    obj_id = args.get('object_id', '')
    summary = _adjacent_summary(ws, allowed_types)
    if not obj_id:
        return ValidationResult(False,
            f"This action requires an object_id but none was provided. Nearby objects: [{summary}].")

    if obj_id not in summary:
        return ValidationResult(False,
            f"Object '{obj_id}' is not within reach. Move closer to the target or choose a different object. Nearby objects: [{summary}].")

    obj = _find_object(obj_id, ws)
    if obj and obj.get('type') not in allowed_types:
        obj_type = obj.get('type', 'unknown')
        return ValidationResult(False,
            f"Object '{obj_id}' is a {obj_type}, not a valid target for this action. Nearby valid objects: [{summary}].")

def _chebyshev_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def _extract_location(obj: Dict[str, Any]) -> Tuple[int, int]:
    """
    Extract a single (x, y) position from an object.
    Supports either:
      - {"x": ..., "y": ...}
      - {"location": [x, y]} or {"location": (x, y)}
    """
    if "x" in obj and "y" in obj:
        return int(obj["x"]), int(obj["y"])

    if "location" in obj:
        loc = obj["location"]
        if isinstance(loc, (list, tuple)) and len(loc) == 2:
            return int(loc[0]), int(loc[1])

    raise ValueError(f"Cannot extract position from object: {obj}")

def _extract_locations(obj: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Extract one or more locations from an object.
    Supports:
      - {"location": [x, y]}
      - {"location": (x, y)}
      - {"location": [[x1, y1], [x2, y2], ...]}
    """
    if "location" not in obj:
        raise ValueError(f"Object has no 'location': {obj}")

    loc = obj["location"]

    # Single location: [x, y] or (x, y)
    if isinstance(loc, (list, tuple)) and len(loc) == 2 and all(isinstance(v, (int, float)) for v in loc):
        return [(int(loc[0]), int(loc[1]))]

    # Multiple locations: [[x1, y1], [x2, y2], ...]
    if isinstance(loc, list):
        result = []
        for p in loc:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                result.append((int(p[0]), int(p[1])))
        return result

    raise ValueError(f"Cannot extract locations from object: {obj}")

def _get_adjacent(ws: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return all objects that are at Chebyshev distance exactly 1 from the agent.
    """
    agent_pos = _extract_location(ws["agent"])
    nearby = []

    for group_name, obj_type in (
        ("victims", "victim"),
        ("obstacles", "obstacle"),
    ):
        for obj in ws.get(group_name, []):
            for loc in _extract_locations(obj):
                if _chebyshev_distance(agent_pos, loc) <= 1:
                    enriched = dict(obj)
                    enriched.setdefault("type", obj_type)
                    enriched["location"] = loc
                    nearby.append(enriched)
                    break 

    # Teammates: expected as {"object_id": ..., "x": ..., "y": ...}
    for t in ws.get("teammates", []):
        loc = _extract_location(t)
        if _chebyshev_distance(agent_pos, loc) == 1:
            nearby.append({
                "object_id": t.get("object_id", t.get("id", "unknown")),
                "type": "teammate",
                "location": loc,
            })

    # Walls: expected as [[x, y], [x, y], ...]
    for i, wall in enumerate(ws.get("walls", [])):
        if isinstance(wall, (list, tuple)) and len(wall) == 2:
            loc = (int(wall[0]), int(wall[1]))
            if _chebyshev_distance(agent_pos, loc) == 1:
                nearby.append({
                    "id": f"wall_{i}",
                    "type": "wall",
                    "location": loc,
                })

    return nearby

def _adjacent_summary(ws: Dict[str, Any], type_filter: Optional[Set[str]] = None) -> str:
    """Human-readable summary of nearby actionable objects."""
    types = type_filter or _OBJECT_TYPES
    parts = []

    for o in _get_adjacent(ws):
        if o.get("type") in types:
            desc = f"{o.get('object_id', o.get('id', '?'))} ({o['type']}"
            if o.get("severity"):
                desc += f", {o['severity']}"
            desc += f" at {o['location']})"
            parts.append(desc)

    return ", ".join(parts) or "none"