from typing import Any, Dict, Optional, Tuple

from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker

from actions1.CustomActions import Idle as _Idle
from memory.base_memory import BaseMemory
from worlds1.environment_info import EnvironmentInformation

_IDLE_ACTION: Tuple[str, Dict] = (_Idle.__name__, {'duration_in_ticks': 1})


def navigate_to(
    target: Tuple[int, int],
    navigator: Navigator,
    state_tracker: StateTracker,
) -> Tuple[Optional[Tuple[int, int]], Tuple[str, Dict]]:
    """Reset navigator and begin A* navigation to target.

    Returns (new_nav_target, action). Caller is responsible for storing new_nav_target.
    """
    navigator.reset_full()
    navigator.add_waypoints([target])
    move = navigator.get_move_action(state_tracker)
    return target, (move, {}) if move else _IDLE_ACTION


def apply_navigation(
    action_name: str,
    kwargs: Dict[str, Any],
    navigator: Navigator,
    state_tracker: StateTracker,
    env_info: EnvironmentInformation,
    memory: BaseMemory,
) -> Tuple[Tuple[str, Dict], Dict[str, Any]]:
    """Set up A* navigation for MoveTo / NavigateToDropZone.
    All other actions are passed through unchanged.

    Returns (action, state_updates) where state_updates is a dict with any subset of: 'nav_target'
    Caller applies these to its own attributes.
    """
    state_updates: Dict[str, Any] = {}

    if action_name == 'MoveTo':
        coords = (int(kwargs.get('x', 0)), int(kwargs.get('y', 0)))
        nav_target, action = navigate_to(coords, navigator, state_tracker)
        state_updates['nav_target'] = nav_target
        return action, state_updates

    if action_name == 'NavigateToDropZone':
        nav_target, action = navigate_to(env_info.drop_zone, navigator, state_tracker)
        state_updates['nav_target'] = nav_target
        return action, state_updates

    if action_name == 'MoveToArea':
        target = int(kwargs.get('area', 0))
        door = env_info.get_door(target)
        if door is None:
            memory.update("action_failure", f"Area {target} does not exist. Try a different one.")
            return _IDLE_ACTION, state_updates
        nav_target, action = navigate_to(door, navigator, state_tracker)
        state_updates['nav_target'] = nav_target
        return action, state_updates

    if action_name == 'EnterArea':
        target = int(kwargs.get('area', 0))
        direction = env_info.get_enter_direction(target)
        if direction == 'North':
            return ('MoveNorth', {}), state_updates
        if direction == 'South':
            return ('MoveSouth', {}), state_updates

    if action_name == 'SearchArea':
        area_id = int(kwargs.get('area', 0))
        area_meta = env_info.areas.get(area_id)
        if area_meta is None:
            memory.update("action_failure", f"Area {area_id} does not exist.")
            return _IDLE_ACTION, state_updates
        # Build serpentine (boustrophedon) waypoint order through inside cells
        cells = sorted(area_meta.inside_cells)
        rows: dict = {}
        for x, y in cells:
            rows.setdefault(y, []).append(x)
        ordered = []
        for i, y in enumerate(sorted(rows.keys())):
            xs = sorted(rows[y], reverse=(i % 2 == 1))
            for x in xs:
                ordered.append((x, y))
        # End at the door
        if area_meta.door:
            ordered.append(area_meta.door)
        # Prepend door so agent navigates there first if not already adjacent
        if area_meta.door:
            ordered = [area_meta.door] + ordered
        if not ordered:
            return _IDLE_ACTION, state_updates
        navigator.reset_full()
        navigator.add_waypoints(ordered)
        state_updates['nav_target'] = ordered[-1]
        move = navigator.get_move_action(state_tracker)
        return (move, {}) if move else _IDLE_ACTION, state_updates

    return (action_name, kwargs), state_updates
