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
    "You are a combined critic and task planner for one Search-and-Rescue agent. "
    "Each cycle you (1) judge whether the last executed action completed the last "
    "planned sub-task, and (2) emit ONE atomic next sub-task. Follow the user "
    "message exactly and return a single JSON object — no prose outside the JSON."
)

PLANNING_INSTRUCTIONS = """
## PART 1 — CRITIC

Judge whether `last_action` completed `last_plan`, using the current OBSERVATION,
WORLD STATE BELIEF, and the YOU block (position, carrying).

Special rules — apply IN ORDER:

1. If `last_action.name == "SendMessage"`, ALWAYS emit:
       success = true, critique = ""
   Communication actions do not change the physical world; critiquing them is
   pure noise. Do not analyse the message contents.

2. If `last_action` is empty/null (first cycle), emit success = true, critique = "".

3. A `MoveTo` whose target [x, y] equals the agent's current position is a
   no-op failure. Critique must suggest checking for obstacles or picking a
   different destination.

4. Otherwise judge based on observable evidence:
   - Pick-up plan ⇒ success iff `carrying` now matches the targeted victim.
   - Move-to-[x,y] plan ⇒ success iff position now equals [x,y].
   - Remove-obstacle plan ⇒ success iff the obstacle is GONE from OBSERVATION
     and WORLD STATE BELIEF.
   - Drop-at-zone plan ⇒ success iff `carrying` is empty.

The critique (if failure) MUST be ONE actionable sentence: name the precondition
that was missing, or the obstacle/teammate that needs attention.


## PART 2 — PLANNER

### Alignment with high-level goal (REQUIRED)

Your `high_level_task` is your role-based directive for the entire run. Every
`next_plan` you emit MUST be a concrete step toward that goal. Before choosing
a plan, ask: "Does this action advance my high-level task?" If the answer is no,
pick a different action. The only exceptions are URGENT help requests (see below)
and genuine blockers that must be cleared first (e.g. an obstacle preventing you
from reaching your target).

### Coherence with recent history (REQUIRED)

Before emitting a plan, review MOST RECENT WORK and MEMORY (newest entries
first). Your plan must be consistent with what you have been doing:

- If `last_plan` is still in progress (target not reached, victim not picked up,
  obstacle not removed), your `next_plan` MUST continue it — do not abandon a
  task mid-way without a clear reason (blocker, success, or loop).
- If MEMORY shows you repeating the same `planned_task` 3+ times with no change
  in position or carrying, that is a loop: change approach (e.g. try a different
  route, remove a blocking obstacle, or pick the next available target).
- Do not jump to a completely different objective just because it appears in your
  OBSERVATION. Stay on your current sub-task unless it is complete or blocked.

### Allowed plan shapes (always include explicit IDs / coordinates)

Emit ONE atomic sub-task completable by a SINGLE action. Forbidden shapes:
"Explore all areas", "Search unexplored areas", "Rescue all victims", or any
composite of multiple steps.

- "Move to [x, y]"
- "Pick up <victim_id> at [x, y]"
- "Carry <victim_id>" (when already adjacent / colocated)
- "Drop <victim_id> at drop zone [x, y]"
- "Remove obstacle <obstacle_id> at [x, y]"
- "Search area <N> for victims"
- "Help <agent_id> rescue <victim_id> at [x, y]"
- "Send message to <agent_id>: <intent>" (only when coordination is truly needed)

### Area search rule (REQUIRED)

`SearchArea` is the ONLY action that guarantees all victims in an area are
found. Plain navigation (MoveTo, MoveToArea, EnterArea) does NOT search —
it only moves the agent and may incidentally mark cells explored in the
coverage tracker, but victims will be MISSED.

- If an area's coverage is < 100%: your next plan MUST be
  "Search area <N> for victims" (executed via SearchArea, not MoveTo).
- Do NOT plan MoveTo or MoveToArea as a substitute for SearchArea.
- An area showing "complete" (100%) coverage that was reached WITHOUT a
  SearchArea call is still unsearched — schedule "Search area <N> for
  victims" for it.
- Only move on to a different area after coverage shows 100% AND the area
  was reached via a SearchArea action.
- If SearchArea is not progressing (same position in recent_actions, coverage
  not increasing), there is likely an obstacle blocking the path inside or
  near the door. Check observation.obstacles and clear the blocker first
  (RemoveObject / RemoveObjectTogether), then retry SearchArea.

### Continuity examples

- last_plan = "Pick up victim_42 at [4, 7]"; position = [2, 1] ⇒
    next_plan = "Move to [4, 7]"
- last_plan = "Pick up victim_42 at [4, 7]"; position = [4, 7]; carrying empty ⇒
    next_plan = "Pick up victim_42 at [4, 7]"
- last_plan = "Move to [4, 7]"; critique = "MoveTo no-op, obstacle in path";
    OBSERVATION shows rock_3 adjacent ⇒
    next_plan = "Remove obstacle rock_3 at [4, 7]"
- last_plan = "Search area 2 for victims"; area_coverage shows area 2 complete (100%) ⇒
    next_plan = "Search area <next area whose coverage < 100%> for victims"
  (If all areas are complete, move to rescuing detected victims from WORLD STATE BELIEF.)

### When to switch tasks

Switch to a new sub-task only when ONE of these is true:
- The last plan SUCCEEDED and no further single-action progress toward
  `high_level_task` is possible with the same sub-task, OR
- MEMORY shows the same `planned_task` 3+ times with no progress (loop), OR
- An URGENT block at the top of the prompt forces a different response.

Never switch to a sub-task that is unrelated to `high_level_task` unless URGENT.

### Obstacle-blocked failures (DO NOT replan from scratch)

If the LAST ACTION REJECTED BY VALIDATOR block is present, the previous action
was rejected — NOT failed by MATRX. Common obstacle-specific cases:

- "obstacle '<id>' is blocking the door" → next_plan = "Remove obstacle <id> at [x, y]".
  After removal, the NEXT cycle should return to the original plan unchanged.
- "navigate to the door first" → next_plan = "Move to [x, y]".
- "strength too low" → next_plan = "Send message to <teammate>: ask_help for <obstacle>".

In all these cases, the high-level task is still valid. Emit ONE bridging
sub-task to clear the blocker, then the next cycle will resume the original plan.
Do NOT abandon or replace the high-level task because of a transient blocker.

### Coordination rules

- An unanswered help request in the URGENT banner MUST be answered this cycle
  with a `SendMessage` plan ("yes" or "no"). This overrides continuity.
- Do NOT plan a task that a teammate's TEAM.current_plan already covers
  (treat `current_plan` as a soft signal — it may be stale, so verify against
  OBSERVATION when in doubt).
- Never target a victim that is missing from WORLD STATE BELIEF (rescued).
- Never target an obstacle that is missing from WORLD STATE BELIEF (removed).

The `high_level_task` in MOST RECENT WORK is derived from your assigned role(s).
Treat it as a persistent directional goal for the entire run. Verify the current
state against WORLD STATE BELIEF and MEMORY before acting; if one target is done,
move to the next one that serves the same goal.


## OUTPUT

Return a single valid JSON object — and nothing else:

{
  "critic": {
    "success": true | false,
    "critique": "actionable single sentence if failed, empty string if success or last_action was SendMessage"
  },
  "next_plan": "one atomic sub-task using the shapes above, with explicit IDs and coordinates so the Reasoning stage needs no extra context",
  "motivation": "<=50 words explaining how this plan advances your high_level_task and why it is consistent with your recent history"
}
"""

# Back-compat alias.
CRITIC_PLAN_PROMPT = SYSTEM_ROLE_PROMPT + "\n" + PLANNING_INSTRUCTIONS


# ── Strategy-specific prompt addenda ────────────────────────────────────────

_STRATEGY_ADDENDA: Dict[str, str] = {
    'io': "",
    'deps': (
        "[DEPS] Treat the high-level task as a chain of dependent sub-goals. "
        "Prefer the sub-task that directly depends on the last plan having been "
        "completed. If the last plan failed, the next plan should unblock it. "
        "Output exactly ONE next_plan.\n\n"
    ),
    'td': (
        "[TD] Think in temporal dependencies: identify the immediate prerequisite "
        "that must be satisfied before the rest of the high-level task can proceed. "
        "Pick the one prerequisite that is now actionable. Output exactly ONE next_plan.\n\n"
    ),
    'voyager': (
        "[Voyager] Pick the next sub-task that builds incrementally on the last "
        "plan's result, expanding the frontier of useful progress. Prefer novel "
        "areas / targets over repetition. Output exactly ONE next_plan.\n\n"
    ),
}


class PlanningBase:
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


def _format_action(action: Any) -> str:
    if not isinstance(action, dict) or not action.get('name'):
        return 'none'
    raw_args = action.get('args')
    if raw_args is None:
        raw_args = {k: v for k, v in action.items() if k != 'name'}
    args_str = ', '.join(f'{k}={v}' for k, v in (raw_args or {}).items())
    return f"{action['name']}({args_str})"


def _format_planning_user_content(information: Dict[str, Any]) -> str:
    ctx = information.get('context', {})
    agent = ctx.get('agent', 'unknown')
    role = ctx.get('role', 'unassigned')
    capabilities = ctx.get('capabilities', '')
    position = ctx.get('position', '?')
    carrying = ctx.get('carrying', 'nothing')
    teammates = ctx.get('teammates', [])

    last_plan = information.get('last_plan', '') or 'none'
    high_level_task = information.get('high_level_task', '') or 'none'
    last_action_str = _format_action(information.get('last_action'))

    observation = information.get('observation', {}) or {}
    world_state_belief = information.get('world_state_belief', {}) or {}
    memory = information.get('memory', []) or []
    messages = information.get('messages', []) or []

    lines: List[str] = []

    # ── URGENT (always first) ───────────────────────────────────────────────
    urgent_help = information.get('urgent_help_request')
    if urgent_help:
        from_agent = urgent_help.get('from', 'teammate')
        req_text = urgent_help.get('text', '')
        lines.extend([
            "== URGENT: INCOMING HELP REQUEST ==",
            f'{from_agent} asks: "{req_text}"',
            "You MUST plan a SendMessage this cycle:",
            '    message="yes" or "no" (exact lowercase, no extra text)',
            f'    send_to="{from_agent}"',
            '    message_type="help"',
            'Saying "yes" commits you to auto-navigate to the victim.',
            "====================================",
            "",
        ])

    active_req = information.get('active_help_request')
    if active_req:
        lines.extend([
            "== TEAM HELP-REQUEST LOCK ==",
            f"Agent {active_req.get('requester')} already has an active ask_help "
            f"for victim {active_req.get('victim_id')} at {active_req.get('victim_location')}.",
            "Do NOT propose a new ask_help this cycle — only one help request "
            "may be active across the team at a time.",
            (f"Already assigned to {active_req.get('accepted_by')}."
             if active_req.get('accepted_by') else
             "Reply 'yes' or 'no' to that request instead of starting your own."),
            "====================================",
            "",
        ])

    urgent_abandon = information.get('urgent_abandon')
    if urgent_abandon:
        lines.extend([
            "== URGENT: ABANDON CURRENT TASK ==",
            urgent_abandon,
            "All teammates have either explicitly refused or ignored your help request.",
            "You MUST stop waiting, pick a completely different objective, and continue your mission independently.",
            "====================================",
            "",
        ])

    last_validation_error = information.get('last_validation_error', '')
    if last_validation_error:
        lines.extend([
            "== LAST ACTION REJECTED BY VALIDATOR ==",
            last_validation_error,
            "IMPORTANT — do NOT replan your entire task because of this. Instead:",
            "  • If an obstacle is blocking the path or door: emit next_plan = "
            "\"Remove obstacle <obstacle_id> at [x, y]\" and resume the original plan once cleared.",
            "  • If you need to navigate to a location first: emit next_plan = "
            "\"Move to [x, y]\" and continue from there.",
            "  • If a partner capability issue: plan to ask for help via SendMessage.",
            "  • Only abandon the entire plan if the target no longer exists or the "
            "validator says the action is permanently impossible.",
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

    # ── MOST RECENT WORK ─────────────────────────────────────────────────────
    lines.extend([
        "== MOST RECENT WORK ==",
        f'High-level task (in-progress assignment — VERIFY against world before continuing; may already be complete): "{high_level_task}"',
        f'Last plan (sub-task that produced last_action): "{last_plan}"',
        f'Last action: {last_action_str}',
        "",
    ])

    # ── OBSERVATION ──────────────────────────────────────────────────────────
    lines.extend([
        "== OBSERVATION (vision range only — victims/teammates/obstacles right now) ==",
        to_toon({
            'victims': observation.get('victims', []),
            'teammates': observation.get('teammates', []),
            'obstacles': observation.get('obstacles', []),
        }),
        "",
    ])

    # ── WORLD STATE BELIEF ───────────────────────────────────────────────────
    lines.extend([
        "== WORLD STATE BELIEF (cumulative across the run; rescued/removed already pruned) ==",
        to_toon({
            'victims': world_state_belief.get('victims', {}),
            'obstacles': world_state_belief.get('obstacles', {}),
        }),
        "",
    ])

    # ── AREA COVERAGE ────────────────────────────────────────────────────────
    area_summaries = information.get('area_summaries') or []
    if area_summaries:
        lines.append("== AREA COVERAGE (door location + % searched) ==")
        for a in area_summaries:
            door_str = f"door={a['door']}" if a.get('door') else "door=unknown"
            pct = int(a.get('coverage', 0) * 100)
            lines.append(f"  {a['name']}: {door_str}, {pct}% searched ({a.get('status', '?')})")
        lines.append("")

    # ── TEAM ─────────────────────────────────────────────────────────────────
    lines.extend([
        "== TEAM (teammates' current_plan may be outdated — soft signal only) ==",
        to_toon(teammates) if teammates else "none",
        "",
    ])

    # ── MEMORY ───────────────────────────────────────────────────────────────
    lines.extend([
        "== MEMORY (past episodes, newest first; each carries motivation + outcome) ==",
        to_toon(memory) if memory else "none",
        "",
    ])

    # ── MESSAGES ─────────────────────────────────────────────────────────────
    lines.extend([
        "== MESSAGES (incoming only, newest first; [HELP REQUEST] entries are critical) ==",
        to_toon(messages) if messages else "none",
        "",
    ])

    # ── INSTRUCTIONS ─────────────────────────────────────────────────────────
    lines.extend([
        "== INSTRUCTIONS ==",
        PLANNING_INSTRUCTIONS.strip(),
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
