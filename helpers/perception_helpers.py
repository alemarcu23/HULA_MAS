from typing import Any, Dict, List, Optional, Tuple

def _serialize_agent(state: Dict[str, Any], agent_id: str
) -> Dict[str, Any]:
    """Extract self-position and carrying status."""
    agent_data = state.get(agent_id, {})
    loc = agent_data.get('location', [0, 0])
    raw_carrying = agent_data.get('is_carrying', [])
    carrying_ids = [
        obj.get('obj_id', str(obj)) if isinstance(obj, dict) else str(obj)
        for obj in raw_carrying
    ]
    return {
        "location": list(loc),
        "carrying": carrying_ids,
    }

def _serialize_nearby(
    state: Dict[str, Any],
    agent_id: str,
    teammate_ids: set,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[List[int]]]:
    """Extract all interesting nearby objects.

    Returns:
        (victims, obstacles, walls) — three separate lists.
    """
    skip = {agent_id, 'World'}
    victims: List[Dict[str, Any]] = []
    obstacles: List[Dict[str, Any]] = []
    walls: List[List[int]] = []

    for obj_id, obj_data in state.items():
        if obj_id in skip:
            continue
        if not isinstance(obj_data, dict):
            continue
        loc = obj_data.get('location')
        if loc is None:
            continue

        obj_type = _classify_type(obj_id, obj_data, teammate_ids)
        if obj_type is None:
            continue

        if obj_type == 'door':
            continue

        if 'wall' in obj_type:
            walls.append([int(c) for c in loc])
            continue

        pos = [int(c) for c in loc]

        if obj_type == 'victim':
            img = str(obj_data.get('img_name', '')).lower()
            if 'critical' in img:
                severity = "critical"
            elif 'mild' in img:
                severity = "mild"
            else:
                severity = "healthy"
            victims.append({
                "id": obj_id,
                "type": obj_type,
                "location": pos,
                "severity": severity,
            })
        elif obj_type in ('rock', 'stone', 'tree'):
            obstacles.append({
                "id": obj_id,
                "type": obj_type,
                "location": pos,
            })

    return victims, obstacles, walls

def _classify_type(
    obj_id: str,
    obj_data: Dict[str, Any],
    teammate_ids: set,
) -> Optional[str]:
    """Return a semantic type string, or None to skip the object.

    Types: 'victim', 'rock', 'stone', 'tree', 'door', 'agent', 'wall'
    """
    oid_lower = str(obj_id).lower()
    class_inh = obj_data.get('class_inheritance', [])

    # Human / other AI agents — check against plain ID set, not tuple set.
    if obj_id in teammate_ids:
        return 'agent'
    if 'AgentBody' in class_inh:
        return 'agent'

    # Victims (collectable)
    if obj_data.get('is_collectable', False):
        return 'victim'

    # Obstacles
    if 'ObstacleObject' in class_inh:
        if 'rock' in oid_lower:
            return 'rock'
        if 'tree' in oid_lower:
            return 'tree'
        return 'stone'

    # Doors
    if 'door' in oid_lower:
        return 'door'

    # Walls
    if 'Wall' in class_inh:
        return 'wall'

    if not obj_data.get('is_traversable', True):
        return 'blocked'

    return None




