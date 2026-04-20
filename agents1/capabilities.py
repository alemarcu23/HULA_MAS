"""
Agent Capabilities System — Presets, resolver, prompt generator, tool filter.

Each agent has 4 capability dimensions:
    - Vision:   1/2/3 blocks visible around agent
    - Strength: low/medium/high — obstacle removal ability
    - Medical:  low/high — victim carrying ability
    - Speed:    slow/normal/fast — movement delay

The environment enforces these via is_possible() checks and action_duration.
"""

from typing import Any, Dict, List, Tuple


# ── Presets ──────────────────────────────────────────────────────────────────

CAPABILITY_PRESETS: Dict[str, Dict[str, Any]] = {
    'scout': {
        'vision': 'high',
        'strength': 'low',
        'medical': 'low',
        'speed': 'fast',
    },
    'medic': {
        'vision': 'low',
        'strength': 'low',
        'medical': 'high',
        'speed': 'normal',
    },
    'heavy_lifter': {
        'vision': 'low',
        'strength': 'high',
        'medical': 'low',
        'speed': 'normal',
    },
    'generalist': {
        'vision': 'medium',
        'strength': 'medium',
        'medical': 'medium',
        'speed': 'normal',
    },
}

DEFAULT_PRESET = 'generalist'

# Extra move ticks for slow agents
SPEED_MOVE_DELAY: Dict[str, int] = {
    'slow': 3,
    'normal': 0,
    'fast': 0,
}

# Valid values per dimension
CAPABILITIES_MAP = {
    'vision': {'low', 'medium', 'high'},
    'strength': {'low', 'medium', 'high'},
    'medical': {'low', 'medium', 'high'},
    'speed': {'slow', 'normal', 'fast'},
}

def resolve_capabilities(preset_or_dict) -> Dict[str, Any]:
    """Resolve a preset name or custom dict into a validated capability dict."""
    if isinstance(preset_or_dict, str):
        if preset_or_dict not in CAPABILITY_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_or_dict}'. "
                f"Available: {list(CAPABILITY_PRESETS.keys())}"
            )
        caps = dict(CAPABILITY_PRESETS[preset_or_dict])
    elif isinstance(preset_or_dict, dict):
        caps = dict(CAPABILITY_PRESETS[DEFAULT_PRESET])
        caps.update(preset_or_dict)
    else:
        caps = dict(CAPABILITY_PRESETS[DEFAULT_PRESET])

    for dim, valid_vals in CAPABILITIES_MAP.items():
        if caps.get(dim) not in valid_vals:
            raise ValueError(
                f"Invalid capability '{dim}': {caps.get(dim)}. Valid: {valid_vals}"
            )
    return caps

def get_capability_prompt(capabilities: Dict[str, Any]) -> str:
    """Return a human-readable description of agent capabilities for the LLM prompt."""
    lines = ["Your agent capabilities:"]

    # Vision
    v = capabilities.get('vision', 'medium')
    vis_desc = {'low': 'low (1 block)', 'medium': 'medium (2 blocks)', 'high': 'high (3 blocks)'}
    lines.append(f"- Vision: you can see objects within {vis_desc.get(v, str(v))}")

    medical = capabilities.get('medical', 'low')
    strength = capabilities.get('strength', 'medium')

    # Medical rules
    if medical == 'high':
        lines.append(
            "- You can carry ALL victims alone (CarryObject)."
        )
    elif medical == 'medium':
            lines.append(
            "- You can carry mildly injured victims alone (CarryObject)."
        )
            lines.append(
            "- Critically injured victims require CarryObjectTogether with an adjacent partner."
        )
    else:
        lines.append(
            "- You can NOT carry any victims alone."
        )
        lines.append(
            "- All victims require CarryObjectTogether with an adjacent partner."
        )

    # Strength rules
    if strength == 'high':
        lines.extend([
            "- You can remove trees, small stones, and big rocks alone (RemoveObject).",
        ])
    elif strength == 'medium':
        lines.extend([
            "- Trees and small stones can be removed alone (RemoveObject).",
            "- Big rocks require RemoveObjectTogether with an adjacent partner.",
        ])
    else:  # low
        lines.extend([
            "- You can only remove fallen trees alone (RemoveObject).",
            "- Small stones and big rocks are too heavy for you alone, "
            "but you can remove them with an adjacent partner (RemoveObjectTogether).",
        ])

    # Speed
    sp = capabilities.get('speed', 'normal')
    if sp == 'slow':
        lines.append("- Speed: slow — each move costs 3 extra ticks (you move significantly slower than other agents).")
    elif sp == 'fast':
        lines.append("- Speed: fast — you move at full speed with no delays.")
    else:
        lines.append("- Speed: normal — standard movement speed.")

    return '\n'.join(lines)

def filter_tools_for_capabilities(
    tool_schemas: List[Dict],
    tools_by_name: Dict[str, Any],
    capabilities: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict]]:
    """Remove tools the agent can NEVER use based on capabilities.

    Returns filtered (tools_by_name, tool_schemas).
    """
    remove_names = set()

    # Joint/cooperative actions (CarryObjectTogether, RemoveObjectTogether) are
    # never filtered — they always succeed regardless of individual capability.

    filtered_tools_by_name = {
        k: v for k, v in tools_by_name.items() if k not in remove_names
    }
    filtered_schemas = [
        s for s in tool_schemas
        if s.get('function', {}).get('name', s.get('name', '')) not in remove_names
    ]

    return filtered_tools_by_name, filtered_schemas

def get_game_rules(drop_zone=(23, 8)) -> str:
    dz = drop_zone
    base_rules = [
        "Goal:",
        f"- Search all areas for victims and rescue them by carrying them to the drop zone at {dz} and using Drop.",
        "- You can only carry one victim at a time.",
        "",
        "Carrying victims:",
        "- Mildly injured victim: use CarryObject (solo) if your medical capability allows, then NavigateToDropZone, then Drop.",
        "- Critically injured victim: use CarryObjectTogether with a partner (both must be adjacent to the victim first).",
        "- After carrying, always navigate to the drop zone and Drop the victim there to score points.",
        "",
        "Removing obstacles:",
        "- Small stone: use RemoveObject (solo). You must be adjacent (distance ≤ 1).",
        "- Fallen tree: use RemoveObject (solo, rescue robot only).",
        "- Big grey rock: use RemoveObjectTogether with a partner (both must be adjacent).",
        "- Remove obstacles that are blocking your path to victims or areas.",
        "",
        "Navigation:",
        "- Use MoveTo(x, y) or MoveToArea(area) to navigate. You must be adjacent to an object to interact with it.",
        "- Use EnterArea(area) when you are at the door of an area to enter it.",
        "- Use SearchArea(area) to systematically cover all cells in an area.",
    ]

    return '\n'.join(base_rules)