"""Role assignment and prompt-text helpers for SAR agents."""

from typing import Any, Dict, List, Optional

ROLE_CLAIM_MSG_TYPE = 'role_claim'

_ROLE_DESCRIPTIONS: Dict[str, str] = {
    'scout':        'Your assigned role is SCOUT. Prioritise exploring unmapped areas and reporting discoveries to your team.',
    'medic':        'Your assigned role is MEDIC. Prioritise locating and carrying injured victims to the drop zone.',
    'heavy_lifter': 'Your assigned role is HEAVY LIFTER. Prioritise removing large obstacles (rocks, trees) to open paths for teammates.',
    'support':      'Your assigned role is SUPPORT. Prioritise helping teammates in cooperative actions and always accepting help requests.',
}

ROLE_GOALS: Dict[str, str] = {
    'scout': (
        "Search as many tiles as possible, map all areas, and inform teammates about victims "
        "and obstacles. Do not waste time rescuing victims unless you have mapped the entire area."
    ),
    'medic': (
        "Your goal is to save as many victims as possible. Pick up victims and carry them to "
        "the drop zone as efficiently as you can."
    ),
    'heavy_lifter': (
        "Remove obstacles blocking paths for your teammates. Your priority is clearing the way "
        "so others can access victims and unexplored areas."
    ),
    'support': (
        "Help your teammates in cooperative actions as often as possible. Always accept help "
        "requests from others and assist actively in joint rescues."
    ),
}


def pick_role(
    team_roles: Dict[str, str],
    world_victims: List[Dict],
    world_obstacles: List[Dict],
    area_summaries: List[Dict],
    teammates: Optional[List] = None,
) -> List[str]:
    """Return the list of roles best suited to the current world state."""
    taken = set()
    for role_str in team_roles.values():
        for r in role_str.split(','):
            taken.add(r.strip())

    unexplored = [s for s in area_summaries if s.get('status') != 'complete']
    has_teammates = bool(teammates) or bool(team_roles)

    roles: List[str] = []
    if 'scout' not in taken and unexplored:
        roles.append('scout')
    if 'medic' not in taken and world_victims:
        roles.append('medic')
    if 'heavy_lifter' not in taken and world_obstacles:
        roles.append('heavy_lifter')
    if has_teammates:
        roles.append('support')

    return roles if roles else ['medic']


def get_role_goal(roles: List[str]) -> str:
    """Return combined high-level goal string for the given list of roles."""
    parts = [ROLE_GOALS[r] for r in roles if r in ROLE_GOALS]
    return ' '.join(parts) if parts else ROLE_GOALS['medic']


def get_role_hint(capabilities: Optional[Dict]) -> str:
    """Return a soft capability-based hint for the LLM prompt."""
    caps = capabilities or {}
    if caps.get('vision') == 'high':
        return 'Your high vision makes you well-suited for the scout role.'
    if caps.get('strength') == 'high':
        return 'Your high strength makes you well-suited for the heavy role.'
    if caps.get('medical') == 'high':
        return 'Your high medical skill makes you well-suited for the medic role.'
    return 'Adapt your role to what the team needs most.'


def get_role_prompt(current_roles: Any) -> str:
    """Return the role description for injection into the LLM system prompt."""
    if isinstance(current_roles, list):
        roles = current_roles
    elif isinstance(current_roles, str) and current_roles:
        roles = [r.strip() for r in current_roles.split(',')]
    else:
        roles = ['medic']

    parts = [_ROLE_DESCRIPTIONS.get(r, f'Your assigned role is {r.upper()}.') for r in roles]
    base = ' '.join(parts)
    return (
        base + ' If you judge that a different role would better serve the team '
        'given the current situation, you may adapt your behaviour accordingly.'
    )
