"""
Agent Capabilities System — Presets, resolver, prompt generator, tool filter.

Each agent has 4 capability dimensions:
    - Vision:   1/2/3 blocks visible around agent
    - Strength: low/medium/high — obstacle removal ability
    - Medical:  low/high — victim carrying ability
    - Speed:    slow/normal/fast — movement delay

The environment enforces these via is_possible() checks and action_duration.
"""

from typing import Any, Dict, List, Optional, Tuple


# Short note injected instead of full capability text when capability_knowledge='discovery'
DISCOVERY_NOTE = (
    "You do not yet know your exact capability limits. "
    "Attempt tasks; if you fail, the critic will tell you why and you should adjust accordingly."
)


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

def get_game_rules(drop_zone=(23, 8), capabilities: Optional[Dict[str, Any]] = None) -> str:
    """Return game rules, optionally with capability-specific carrying/removal lines.

    Args:
        drop_zone: (x, y) coordinates of the rescue drop zone.
        capabilities: Agent capability dict (vision/strength/medical/speed). When
            provided, replaces the generic 'check your capabilities' note with
            concrete rules derived from the agent's actual profile.
    """
    dz = drop_zone

    # --- Carrying rules ---
    if capabilities is None:
        carry_lines = [
            "- Mildly injured victim: use CarryObject (solo) if your medical capability allows, then NavigateToDropZone, then Drop.",
            "- Critically injured victim: always use CarryObjectTogether with a partner (both must be adjacent to the victim first).",
        ]
    else:
        medical = capabilities.get('medical', 'low')
        if medical == 'high':
            carry_lines = [
                "- You can carry ALL victims alone: use CarryObject, then NavigateToDropZone, then Drop.",
                "- Critically injured victims STILL score more points when carried cooperatively, but you may do it solo.",
            ]
        elif medical == 'medium':
            carry_lines = [
                "- Mildly injured victim: use CarryObject (solo), then NavigateToDropZone, then Drop.",
                "- Critically injured victim: you CANNOT carry these alone — use CarryObjectTogether with an adjacent partner.",
            ]
        else:  # low
            carry_lines = [
                "- Your medical level is LOW: you CANNOT carry any victim alone.",
                "- ALL victims require CarryObjectTogether with an adjacent partner (partner_id REQUIRED).",
            ]

    # --- Removal rules ---
    if capabilities is None:
        removal_lines = [
            "- Small stone: use RemoveObject (solo). You must be adjacent (distance ≤ 1).",
            "- Fallen tree: use RemoveObject (solo, rescue robot only).",
            "- Big grey rock: ALWAYS use RemoveObjectTogether with a partner (both must be adjacent).",
        ]
    else:
        strength = capabilities.get('strength', 'low')
        if strength == 'high':
            removal_lines = [
                "- You can remove trees, small stones, and big grey rocks alone (RemoveObject).",
                "- You must be adjacent (distance ≤ 1) to the obstacle.",
            ]
        elif strength == 'medium':
            removal_lines = [
                "- Trees and small stones: use RemoveObject (solo). You must be adjacent (distance ≤ 1).",
                "- Big grey rock: use RemoveObjectTogether with a partner (both must be adjacent).",
            ]
        else:  # low
            removal_lines = [
                "- Your strength is LOW: you can only remove fallen trees alone (RemoveObject).",
                "- Small stones and big grey rocks require RemoveObjectTogether with an adjacent partner.",
            ]

    base_rules = (
        ["== GAME RULES ==",
         "Goal:",
         f"- Search all areas for victims and rescue them by carrying them to the drop zone at {dz} and using Drop.",
         "- You can only carry one victim at a time.",
         "",
         "Carrying victims:"]
        + carry_lines
        + ["- After carrying, always navigate to the drop zone and Drop the victim there to score points.",
           "",
           "Removing obstacles:"]
        + removal_lines
        + ["- Remove obstacles that are blocking your path to victims or areas.",
           "",
           "Cooperative actions:",
           "- CarryObjectTogether and RemoveObjectTogether require partner_id from observation.teammates.",
           "- Both agents must be adjacent (Chebyshev distance ≤ 1) to the target before calling a cooperative action.",
           "",
           "Navigation:",
           "- Use MoveTo(x, y) or MoveToArea(area) to navigate. You must be adjacent to an object to interact with it.",
           "- Use EnterArea(area) when you are at the door of an area to enter it.",
           "- Use SearchArea(area) to systematically cover all cells in an area.",
           "- NEVER call MoveTo with your own current position — that is a no-op.",
           ]
    )

    return '\n'.join(base_rules)