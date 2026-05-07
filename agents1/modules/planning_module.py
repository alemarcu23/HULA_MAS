import json
import logging
from typing import Dict, List, Any, Optional
from helpers.toon_utils import to_toon

try:
    from engine.parsing_utils import load_few_shot
except ImportError:
    def load_few_shot(key):
        return []

logger = logging.getLogger('Planning')

SYSTEM_ROLE_PROMPT = (
    "You are a combined evaluator and task planner for a search-and-rescue agent. "
    "Follow the instructions in the user message exactly."
)

PLANNING_INSTRUCTIONS = """
## STEP 1: EVALUATE LAST ACTION
Assess whether the `last_action` advanced the goal.
*   SUCCESS: The action achieved or exceeded its intent. (Note: If `last_action` is empty or null, treat it as a success).
*   FAILURE: The action did not achieve its intent.
       Rule: A `MoveTo` action where the target coordinates exactly match the agent's current position is ALWAYS a no-op failure. However, this might be because something is blocking the path. You should check for this next instead of trying the same plan again.
*   CRITIQUE: If an action fails, your critique MUST be actionable. State what went wrong, suggest a specific fix, and note preconditions to verify (e.g., "Check partner_id", "Choose new destination").

Examples:
INPUT: Position [3,5], Carrying mildly_injured_woman, Last action: CarryObject(object_id="mildly_injured_woman"), Task: Pick up mild victim at [3, 5]
OUT: success=true, critique=""

INPUT: Position [5,3], Last action: MoveTo(x=5, y=3), Task: Navigate to victim at [8, 7]
OUT: success=false, critique="MoveTo target (5,3) equals current position — no-op. Choose a different destination toward [8, 7]."

INPUT: Position [5,10], Carrying: None, Last action: CarryObjectTogether(object_id="critically_injured_man"), Task: Carry critical victim cooperatively
OUT: success=false, critique="CarryObjectTogether failed — check that teammate is adjacent."

## STEP 2: PLAN NEXT TASK

Your `high_level_task` is the mission assigned to you by the coordination round (e.g. "Search area 3 for victims", "Rescue mildly_injured_woman at [4,7]"). Use the last planned sub-task and `high_level_task` together to decide the single best NEXT atomic step to execute right now.

If `high_level_task` is empty, choose the most valuable action given the world state and your role.

IF the critique from STEP 1 indicates a failure, the next plan should be different from the previous one and address the failure. For example, if the last action was a failed MoveTo, check for obstacles in the observations and plan to remove them or choose a different area to search.
If you are a searching agent, you should focus on exploring new areas, especially those with low coverage. If you are a rescuing agent, you should focus on picking up or delivering victims.
If you are a heavy_duty agent, you should focus on removing obstacles or assisting teammates.
You are free to ignore your role for the next task, but you should choose actions that fit the current situation, your agent's capabilities and your team's needs.

Rules:
- Do NOT assign tasks that are already completed or currently being executed by another agent.
- Include object IDs, agent IDs, area names, and [x, y] coordinates whenever known.
- Victims listed in HISTORY > rescued_victims are already safe — NEVER plan to rescue, carry, or interact with them.
- If the same task appears 3 or more times in HISTORY > previous_tasks without progress, switch to a completely different task: explore a new room or area not yet covered (use HISTORY > area_coverage to find uncovered areas).
- Output exactly ONE next_task — a single atomic step, not a list or multi-step plan.
- Do NOT repeat a task that a teammate already has as their current_plan (see TEAM section).

## INPUT DATA REFERENCE
*   `YOU`: Your identity, role, capabilities, position, and what you are carrying.
*   `CURRENT WORK`: Your high-level assignment, the last sub-task you were executing, and the last action you took.
*   `TEAM`: Your teammates — their positions, roles, and current plans. Coordinate to avoid duplication.
*   `WORLD STATE`: Current observations and all discovered objects.
*   `HISTORY > previous_tasks`: Your recent planning history (last_task / planned_task / action / outcome). Avoid repeating past failures.
*   `HISTORY > rescued_victims`: Victims already delivered to the drop zone — never re-rescue these.
*   `HISTORY > area_coverage`: Coverage % per area — prioritize lowest coverage when exploring.
*   `MESSAGES`: Recent inter-agent messages including help requests and coordination.

Good examples of next_task:
- "Respond to rescuebot0's message to carry victim_3 together by replying to the message."
- "Deliver victim_2 to the drop zone."
- "Pick up victim_1 at [2, 5] alone."
- "Remove the rock blocking Area 2 at [4, 7]."
- "Move to Area 1 door."

## OUTPUT FORMAT

Respond with a single valid JSON object — nothing else:
{
  "reasoning": "brief explanation of what the last action did and why",
  "success": true or false,
  "critique": "actionable next step if failed, empty string if succeeded",
  "next_task": "single-sentence next task for this agent that is immediately actionable and addresses any critique if failed"
}
"""

# Keep backward-compat alias so any external code referencing CRITIC_PLAN_PROMPT still works.
CRITIC_PLAN_PROMPT = SYSTEM_ROLE_PROMPT + "\n" + PLANNING_INSTRUCTIONS


# ── Strategy-specific prompt addenda ────────────────────────────────────────
# Each strategy prepends a short instruction block to the base prompt,
# shaping how the agent picks the single next sub-task from (high_level_task,
# past_subtask, observations). All strategies still emit exactly one next_task.

_STRATEGY_ADDENDA: Dict[str, str] = {
    'io': "",  # baseline — no extra guidance
    'deps': (
        "[DEPS] Treat the high-level task as a chain of dependent sub-goals. "
        "Prefer the sub-task that directly depends on the PAST sub-task having been "
        "completed. If the past sub-task failed, the next sub-task should re-attempt "
        "or unblock it. Output exactly ONE next_task.\n\n"
    ),
    'td': (
        "[TD] Think in temporal dependencies: identify the immediate prerequisite "
        "that must be satisfied before the rest of the high-level task can proceed. "
        "Pick the one prerequisite that is now actionable given the past sub-task and "
        "emit it as the single next_task. Output exactly ONE next_task.\n\n"
    ),
    'voyager': (
        "[Voyager] Pick the next sub-task that builds incrementally on the past "
        "sub-task's result, expanding the frontier of useful progress. Prefer novel "
        "areas or targets over repetition of what the past sub-task already covered. "
        "Output exactly ONE next_task.\n\n"
    ),
}


# ── Strategy classes ────────────────────────────────────────────────────────

class PlanningBase:
    """Base planning strategy — no prompt decoration, IO-style behavior.

    Subclasses override ``prefix`` to prepend strategy-specific instructions
    to the base planning prompt.
    """

    prefix: str = _STRATEGY_ADDENDA['io']

    def decorate_system_prompt(self, base_prompt: str) -> str:
        return (self.prefix + base_prompt) if self.prefix else base_prompt


class PlanningIO(PlanningBase):
    prefix = _STRATEGY_ADDENDA['io']


class PlanningDEPS(PlanningBase):
    prefix = _STRATEGY_ADDENDA['deps']


class PlanningTD(PlanningBase):
    prefix = _STRATEGY_ADDENDA['td']


class PlanningVoyager(PlanningBase):
    prefix = _STRATEGY_ADDENDA['voyager']


PLANNING_STRATEGY_REGISTRY: Dict[str, type] = {
    'io':         PlanningIO,
    'deps':       PlanningDEPS,
    'td':         PlanningTD,
    'voyager':    PlanningVoyager,
}


def build_planning_strategy(name: str) -> PlanningBase:
    cls = PLANNING_STRATEGY_REGISTRY.get(name, PlanningIO)
    return cls()


def _format_planning_user_content(information: Dict[str, Any]) -> str:
    ctx = information.get('context', {})
    agent = ctx.get('agent', 'unknown')
    role = ctx.get('role', 'unassigned')
    capabilities = ctx.get('capabilities', '')
    position = ctx.get('position', '?')
    carrying = ctx.get('carrying', 'nothing')
    teammates = ctx.get('teammates', [])

    current_task = information.get('current_task', '') or 'none'
    high_level_task = information.get('high_level_task', '') or 'none'

    last_action = information.get('last_action') or {}
    last_action_name = last_action.get('name', '') if isinstance(last_action, dict) else ''
    raw_args = (
        last_action.get('args')
        if isinstance(last_action, dict) and 'args' in last_action
        else {k: v for k, v in last_action.items() if k != 'name'}
        if isinstance(last_action, dict)
        else {}
    )
    args_str = ', '.join(f'{k}={v}' for k, v in raw_args.items()) if raw_args else ''
    last_action_str = f'{last_action_name}({args_str})' if last_action_name else 'none'

    history = information.get('history', {})
    previous_tasks_raw = history.get('previous_tasks') or []
    previous_tasks = [
        {
            'tick': ep.get('tick'),
            'last_task': ep.get('task'),
            'planned_task': ep.get('planned_task'),
            'action': ep.get('action'),
            'outcome': 'success' if ep.get('outcome_succeeded') else 'failure',
            'critique': ep.get('critique') or '',
        }
        for ep in previous_tasks_raw
    ]

    messages = history.get('messages')

    lines = []

    # ── URGENT blocks (always first) ────────────────────────────────────────
    urgent_help = information.get('urgent_help_request')
    if urgent_help:
        from_agent = urgent_help.get('from', 'teammate')
        req_text = urgent_help.get('text', '')
        lines.extend([
            "== URGENT: INCOMING HELP REQUEST ==",
            f'{from_agent} asks: "{req_text}"',
            "You MUST respond this tick by calling SendMessage with:",
            '    message="yes" or "no" (exact lowercase word, no extra text)',
            f'    send_to="{from_agent}"',
            '    message_type="help"',
            'Saying "yes" commits you to auto-navigate to the victim.',
            'Say "no" if you\'re busy or already committed to a higher-priority task.',
            "====================================",
            "",
        ])

    urgent_abandon = information.get('urgent_abandon')
    if urgent_abandon:
        lines.extend([
            "== URGENT: ABANDON CURRENT TASK ==",
            urgent_abandon,
            "====================================",
            "",
        ])

    # ── YOU ──────────────────────────────────────────────────────────────────
    lines.extend([
        "== YOU ==",
        f"Name: {agent}",
        f"Role: {role}",
        "Capabilities:",
        capabilities,
        f"Position: {position}",
        f"Carrying: {carrying}",
        "",
    ])

    # ── CURRENT WORK ─────────────────────────────────────────────────────────
    lines.extend([
        "== CURRENT WORK ==",
        f"High-level assignment: \"{high_level_task}\"",
        f"Last planned sub-task (what the last action tried to complete): \"{current_task}\"",
        f"Last action: {last_action_str}",
        "",
    ])

    # ── TEAM ─────────────────────────────────────────────────────────────────
    lines.extend([
        "== TEAM ==",
        to_toon(teammates) if teammates else "none",
        "",
    ])

    # ── INSTRUCTIONS ─────────────────────────────────────────────────────────
    lines.extend([
        "== YOUR TASK NOW ==",
        PLANNING_INSTRUCTIONS.strip(),
        "",
    ])

    # ── WORLD STATE ──────────────────────────────────────────────────────────
    lines.extend([
        "== WORLD STATE ==",
        to_toon({
            'observation': information.get('observation', {}),
            'all_discovered': information.get('all_discovered', {}),
        }),
        "",
    ])

    # ── HISTORY ──────────────────────────────────────────────────────────────
    lines.extend([
        "== HISTORY ==",
        to_toon({
            'previous_tasks': previous_tasks,
            'rescued_victims': history.get('rescued_victims'),
            'area_coverage': history.get('area_coverage'),
        }),
        "",
    ])

    # ── MESSAGES ─────────────────────────────────────────────────────────────
    lines.extend([
        "== MESSAGES ==",
        to_toon(messages) if messages else "none",
    ])

    return '\n'.join(lines)


class Planning:
    def __init__(self, mode: str = 'simple', strategy: str = 'io') -> None:
        self.mode = mode
        self.strategy_name = strategy
        self.strategy = build_planning_strategy(strategy)
        self.current_task = ''

    def set_current_task(self, task: str) -> None:
        self.current_task = task

    def get_planning_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        system_content = self.strategy.decorate_system_prompt(SYSTEM_ROLE_PROMPT)

        messages = [{"role": "system", "content": system_content}]

        try:
            examples = load_few_shot('planning_next_task')
            for ex in examples:
                if 'user' in ex and 'assistant' in ex:
                    messages.append({"role": "user", "content": ex['user'].strip()})
                    messages.append({"role": "assistant", "content": ex['assistant'].strip()})
        except Exception:
            pass

        messages.append({"role": "user", "content": _format_planning_user_content(information)})
        return messages
