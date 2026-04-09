"""
Unified action dispatch: maps a tool/action name + LLM-supplied arguments
to a MATRX-ready (action_class_name, kwargs) pair.

Replaces the near-identical dispatch logic that previously lived in both
SearchRescueAgent._dispatch_tool_call() and ActionMapper._dispatch().

Cooperative actions are enriched with ``partner_name`` — an internal MATRX
parameter that is never exposed to the LLM.  The partner can be a human
agent or another AI agent.
"""

import logging
from typing import Dict, Any, Tuple, Optional

from actions1.CustomActions import Idle as _Idle
from actions1.CustomActions import CarryObject as _CarryObject
from actions1.CustomActions import Drop as _Drop
from actions1.CustomActions import CarryObjectTogether as _CarryObjectTogether
from actions1.CustomActions import DropObjectTogether as _DropObjectTogether
from actions1.CustomActions import RemoveObjectTogether as _RemoveObjectTogether
from matrx.actions.object_actions import RemoveObject as _RemoveObject

logger = logging.getLogger('action_dispatch')

# Direction moves require no kwargs.
_MOVE_ACTIONS = frozenset({'MoveNorth', 'MoveSouth', 'MoveEast', 'MoveWest'})

_DEFAULT_TASK_COMPLETING = {
    'Drop': 'dropping carried victim',
    'DropObjectTogether': 'dropping carried victim cooperatively',
    'Idle': 'idling',
}

def execute_action(
    name: str,
    args: Dict[str, Any],
    agent_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], str]:
    """Map an action *name* + LLM-supplied *args* to a MATRX (action, kwargs) pair.

    Args:
        name:         Action name as returned by the LLM (tool_call or text-JSON).
        args:         Argument dict from the LLM (may be empty).
        agent_id:     Optional agent ID for logging context.

    Returns:
        ``(action_class_name, kwargs, task_completing)`` ready for MATRX.
    """
    log_prefix = f"[{agent_id}] " if agent_id else ""

    # Extract task_completing before dispatching (falls back to defaults)
    task_completing = args.pop('task_completing', '') or _DEFAULT_TASK_COMPLETING.get(name, '')
    print(f"{log_prefix}Dispatching action '{name}' with args {args} and partner '{args.get('partner_id', '')}'")

    # ── Movement ──────────────────────────────────────────────────────────
    if name in _MOVE_ACTIONS:
        return name, {}, task_completing

    if name == 'MoveTo':
        return 'MoveTo', {'x': args.get('x', 0), 'y': args.get('y', 0)}, task_completing

    if name == "MoveToArea":
        return 'MoveToArea', {'area': args.get('area', 1)}, task_completing
    
    if name == "EnterArea":
        return 'EnterArea', {'area': args.get('area', 1)}, task_completing

    if name == "SearchArea":
        return 'SearchArea', {'area': args.get('area', 1)}, task_completing

    if name == 'NavigateToDropZone':
        return 'NavigateToDropZone', {}, task_completing

    if name == 'SendMessage':
        return 'SendMessage', {
            'message': args.get('message', "Empty"),
            'send_to': args.get('send_to', "all"),
            'message_type': args.get('message_type', 'message'),
        }, task_completing

    # ── Solo carry / drop ─────────────────────────────────────────────────
    if name == 'CarryObject':
        obj_id = args.get('object_id', '')
        if not obj_id:
            logger.warning("%sCarryObject called without object_id", log_prefix)
            return _Idle.__name__, {'duration_in_ticks': 1}, task_completing
        return _CarryObject.__name__, {'object_id': obj_id}, task_completing

    if name == 'Drop':
        return _Drop.__name__, {}, task_completing

    # ── Joint actions ──────────────────────────────────────────
    if name == 'CarryObjectTogether':
        obj_id = args.get('object_id', '')
        if not obj_id:
            logger.warning("%sCarryObjectTogether called without object_id", log_prefix)
            return _Idle.__name__, {'duration_in_ticks': 1}, task_completing
        resolved_partner = args.pop('partner_id', '')
        return _CarryObjectTogether.__name__, {'object_id': obj_id, 'partner_name': resolved_partner}, task_completing
    
    if name == 'RemoveObjectTogether':
        obj_id = args.get('object_id', '')
        if not obj_id:
            logger.warning("%sRemoveObjectTogether called without object_id", log_prefix)
            return _Idle.__name__, {'duration_in_ticks': 1}, task_completing
        resolved_partner = args.pop('partner_id', '')
        return _RemoveObjectTogether.__name__, {
            'object_id': obj_id,
            'remove_range': 1,
            'partner_name': resolved_partner,
        }, task_completing

    # ── Remove obstacle (solo) ────────────────────────────────────────────
    if name == 'RemoveObject':
        obj_id = args.get('object_id', '')
        if not obj_id:
            logger.warning("%sRemoveObject called without object_id", log_prefix)
            return _Idle.__name__, {'duration_in_ticks': 1}, task_completing
        return _RemoveObject.__name__, {'object_id': obj_id, 'remove_range': 1}, task_completing

    # ── Idle ──────────────────────────────────────────────────────────────
    if name == 'Idle':
        ticks = int(args.get('duration_in_ticks', 1))
        return _Idle.__name__, {'duration_in_ticks': ticks}, task_completing
    
    logger.warning("%sUnknown action '%s', defaulting to Idle", log_prefix, name)
    return _Idle.__name__, {'duration_in_ticks': 1}, ''
