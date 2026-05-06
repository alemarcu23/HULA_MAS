"""
Tool registry: @tool-decorated MATRX action functions, reasoning strategy
prompts, and a schema-builder helper.

All 13 LangChain @tool functions live here so search_rescue_agent.py stays
focused on tick-loop orchestration.  Each tool returns a plain
(action_name, args_dict) tuple — the actual MATRX dispatch and partner_name
enrichment happen in action_dispatch.py at runtime.

Game rules are the single source of truth in agents1.capabilities.get_game_rules().

Usage:
    from agents1.tool_registry import (
        ALL_ACTION_TOOLS, REASONING_STRATEGIES,
        build_tool_schemas,
    )
"""

from typing import Dict, Any, List, Tuple

from langchain.tools import tool


# ── Reasoning strategy system prompts ─────────────────────────────────────────

REASONING_STRATEGIES: Dict[str, str] = {
    'cot': (
        "You are a search and rescue robot. Use Chain-of-Thought reasoning: "
        "think step-by-step about your goal and current situation, "
        "then call the single best action tool."
    ),
    'react': (
        "You are a search and rescue robot. Use ReAct reasoning:\n"
        "Thought: <reason about goal, observations, and constraints>\n"
        "Then call the single best action tool."
    ),
    'reflexion': (
        "You are a search and rescue robot. Use Reflexion reasoning: "
        "before acting, reflect on what you have done and what failed.\n"
        "If a previous action failed, try a completely different approach, "
        "then call the single best action tool."
    ),
}

# Game rules: single source of truth in agents1.capabilities.get_game_rules()

# ── Action tools ──────────────────────────────────────────────────────────────
# Defined at module level so they don't shadow the aliased MATRX action-class
# imports used elsewhere.  Each tool returns the MATRX action name and the
# LLM-supplied arguments; partner_name enrichment happens in action_dispatch.py.


@tool
def MoveNorth(task_completing: str):
    """Move one cell north (decreases y by 1)."""
    return 'MoveNorth', {}, {'task_completing': task_completing}


@tool
def MoveSouth(task_completing: str):
    """Move one cell south (increases y by 1)."""
    return 'MoveSouth', {}, {'task_completing': task_completing}


@tool
def MoveEast(task_completing: str):
    """Move one cell east (increases x by 1)."""
    return 'MoveEast', {}, {'task_completing': task_completing}


@tool
def MoveWest(task_completing: str):
    """Move one cell west (decreases x by 1)."""
    return 'MoveWest', {}, {'task_completing': task_completing}


@tool
def MoveTo(x: int, y: int, task_completing: str):
    """Navigate to a specific grid coordinate using A* pathfinding.

    Args:
        x: Target column (east-west axis).
        y: Target row (north-south axis).
        task_completing: A brief description of the subtask this move will complete
                        (e.g. "approaching victim to carry" or "navigating to drop zone").
    """
    return 'MoveTo', {'x': x, 'y': y, 'task_completing': task_completing}

@tool
def MoveToArea(area: int, task_completing: str):
    """Navigate to a specific Area using A* pathfinding.

    Args:
        Area: the number of the area to navigate to
        task_completing: A brief description of the subtask this move will complete
                        (e.g. "approaching victim to carry" or "navigating to drop zone").
    """
    return 'MoveToArea', {'area': area}, {'task_completing': task_completing}


@tool
def NavigateToDropZone(task_completing: str = "navigating to drop zone"):
    """Navigate to the rescue drop zone to deliver a carried victim.
    Use this after CarryObject or CarryObjectTogether; follow with Drop to score points."""
    return 'NavigateToDropZone', {'task_completing': task_completing}


@tool
def CarryObject(object_id: str, task_completing: str = "carrying victim"):
    """Pick up and carry a victim solo. Only valid if your medical capability allows it.
    - medical=high: can carry ALL victims (mild and critical) alone.
    - medical=medium: can carry MILDLY injured victims alone; critical requires CarryObjectTogether.
    - medical=low: CANNOT carry any victim alone; always use CarryObjectTogether.
    You must be adjacent (Chebyshev distance ≤ 1) to the victim. After picking up, use
    NavigateToDropZone then Drop to score points.

    Args:
        object_id: The ID of the victim to carry (from observation.nearby_victims).
        task_completing: Brief description of the subtask this action completes.
    """
    return 'CarryObject', {'object_id': object_id}, {'task_completing': task_completing}


@tool
def CarryObjectTogether(object_id: str, partner_id: str, task_completing: str = "carrying victim cooperatively"):
    """Cooperatively carry a critically injured victim with a partner agent.
    Required when your medical capability is low, or when carrying a critical victim regardless of capability.
    BOTH agents must be adjacent (Chebyshev distance ≤ 1) to the victim before calling this.
    After picking up, BOTH agents must call NavigateToDropZone then DropObjectTogether to score points.

    Args:
        object_id: The ID of the victim to carry cooperatively (from observation.nearby_victims).
        partner_id: REQUIRED — the object_id of the adjacent teammate from observation.teammates.
        task_completing: Brief description of the subtask this action completes.
    """
    return 'CarryObjectTogether', {'object_id': object_id, 'partner_id': partner_id}, {'task_completing': task_completing}


@tool
def Drop():
    """Drop the currently carried object at the current grid position.
    Use this at the drop zone after NavigateToDropZone to score rescue points."""
    return 'Drop', {}, {'task_completing': "dropping carried victim"}


@tool
def RemoveObject(object_id: str, task_completing: str = "removing obstacle"):
    """Remove a small stone or fallen tree obstacle solo. Capability constraints apply:
    - strength=high: can remove trees, stones, and rocks alone.
    - strength=medium: can remove trees and small stones alone; big rocks require RemoveObjectTogether.
    - strength=low: can only remove fallen trees alone; stones and rocks require RemoveObjectTogether.
    Note: ONLY the rescue robot (RescueBot) can remove trees; human agents cannot.
    You must be adjacent (Chebyshev distance ≤ 1) to the obstacle.
    Big grey rocks ALWAYS require RemoveObjectTogether regardless of strength.

    Args:
        object_id: The ID of the obstacle to remove (from observation.nearby_obstacles).
        task_completing: Brief description of the subtask this action completes.
    """
    return 'RemoveObject', {'object_id': object_id}, {'task_completing': task_completing}


@tool
def RemoveObjectTogether(object_id: str, partner_id: str, task_completing: str = "removing obstacle cooperatively"):
    """Cooperatively remove a big grey rock obstacle with a partner agent.
    Big rocks ALWAYS require both agents — solo removal is never possible regardless of strength.
    BOTH agents must be adjacent (Chebyshev distance ≤ 1) to the rock before calling this.

    Args:
        object_id: The ID of the rock to remove cooperatively (from observation.nearby_obstacles).
        partner_id: REQUIRED — the object_id of the adjacent teammate from observation.teammates.
        task_completing: Brief description of the subtask this action completes.
    """
    return 'RemoveObjectTogether', {'object_id': object_id, 'partner_id': partner_id}, {'task_completing': task_completing}

@tool
def SearchArea(area: int, task_completing: str = "searching area"):
    """Systematically search all cells inside an area for victims and obstacles.
    You must be at the door of the area before calling this action.
    The agent will visit every cell in the area and return to the door.

    Args:
        area: The number of the area to search (1-14).
        task_completing: Description of the subtask being completed.
    """
    return 'SearchArea', {'area': area}, {'task_completing': task_completing}


@tool
def SendMessage(message: str, send_to: str, message_type: str = "message"):
    """Send a message to one or all teammates. This uses your action for this tick.
    Use sparingly — keep messages to 1-2 sentences.

    Args:
        message: The message content to send (1-2 sentences max).
        send_to: Agent name for a directed message, or "all" for a broadcast.
        message_type: One of:
            - "ask_help": request assistance (partner is expected to reply with "help").
            - "help": respond to an ask_help request from a teammate.
            - "message": general status update or information sharing.
    """
    return 'SendMessage', {'message': message, 'send_to': send_to,
        'message_type': message_type}, {'task_completing': f"sending {message_type} message"}


# Ordered list of every action tool — used to build the registry + LLM schemas.
# MoveNorth/South/East/West removed: MoveTo with A* is always preferable.
ALL_ACTION_TOOLS = [
    MoveTo, MoveToArea, NavigateToDropZone, SearchArea,
    CarryObject, CarryObjectTogether,
    Drop,
    RemoveObject, RemoveObjectTogether, SendMessage
]


# ── Schema builder ────────────────────────────────────────────────────────────

def build_tool_schemas() -> Tuple[Dict[str, Any], List[Dict]]:
    """Build (tools_by_name, tool_schemas) from ALL_ACTION_TOOLS.

    Returns:
        tools_by_name:  ``{name: StructuredTool}`` lookup dict.
        tool_schemas:   OpenAI-compatible tool schema list for Ollama.
    """
    import logging
    from langchain_core.utils.function_calling import convert_to_openai_tool

    logger = logging.getLogger('tool_registry')

    tools_by_name: Dict[str, Any] = {t.name: t for t in ALL_ACTION_TOOLS}

    try:
        tool_schemas: List[Dict] = [
            convert_to_openai_tool(t) for t in ALL_ACTION_TOOLS
        ]
    except Exception as exc:
        logger.warning(
            "convert_to_openai_tool failed (%s); using matrx_tool_description fallback", exc
        )

    return tools_by_name, tool_schemas