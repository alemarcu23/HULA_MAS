import json
import re
from typing import Dict, List, Any, Optional
from helpers.toon_utils import to_toon


REASONING_PROMPT_CORE = """
You are a Search and Rescue agent. Your job each cycle is to execute the
`current_plan` (already decided by the planner) by emitting ONE tool call —
or TWO tool calls in the single allowed pairing described below.

The plan is atomic and fully specified (it includes any required victim_id,
[x, y], obstacle_id, partner_id). You do NOT re-plan. If the plan looks
unreachable, choose the tool call that makes the most direct progress toward
it (e.g. MoveTo before Carry, RemoveObject if an obstacle is in the way).

Two-call pairing (the ONLY allowed multi-call shape):
- You MAY emit exactly TWO tool calls when the first is `MoveTo(x, y)` and the
  second is the colocated follow-up that the plan implies at that destination:
  `PickUp`, `SearchArea`, `RemoveObject`, `CarryObject`, or `Drop`.
- The second call must use the same target (matching coordinates / object_id)
  that the plan specifies. Navigation will complete first across multiple
  ticks; the second action will fire automatically on arrival without a fresh
  planner cycle.
- Never emit two MoveTos, never emit a SendMessage as the second call, never
  emit a follow-up before a non-MoveTo primary. In any other case emit ONE
  tool call.
- If the current_plan is itself a non-move atomic action (e.g. "Pick up
  victim_X at [x, y]" while you are already at [x, y]), emit ONE call — do
  not pad with a redundant MoveTo.

Rules:
- Return one tool call (or the allowed two-call pair) — no natural language,
  no explanations.
- Every tool call has a `task_completing` field. Set it to the exact
  `current_plan` text if this single action completes the plan; otherwise "N/A".
- Verify from OBSERVATION that the plan is actually completed before setting
  `task_completing` to anything other than "N/A".
- The action target MUST match the plan target (same object_id / coordinates).
- If current_plan is "Search area N for victims", you MUST call SearchArea(area=N).
  Do NOT substitute MoveTo or MoveToArea — those do not search the area and will
  miss victims. If an obstacle blocks the door or path, remove it first, then
  SearchArea.
- If the URGENT banner names an incoming help request, emit a SendMessage
  with `message_type="help"` and `message` exactly "yes" or "no".
- If the planner's `critique` is non-empty, treat the previous action as failed
  and pick a DIFFERENT action that addresses the critique.
- Loop check: if `recent_actions` shows the same action 2+ times, you are
  looping — pick a completely different action type.

Joint action requirements:
- CarryObjectTogether / RemoveObjectTogether REQUIRE `partner_id` — use an
  `object_id` from OBSERVATION.teammates.
- Both you and the partner must be adjacent (Chebyshev distance ≤ 1) to the
  target before issuing a cooperative action.
- If no partner is adjacent, send an `ask_help` message via SendMessage.

Capability coordination:
- You only know YOUR OWN capabilities (YOUR CAPABILITIES block below).
- You do NOT know teammates' strength / medical skill / vision range.
- If a coordination decision hinges on a teammate's capabilities, SendMessage
  them asking "What are your capabilities?" before assuming.
- If a teammate asks about yours, ALWAYS reply with a brief description.

Messaging:
- message_type="ask_help" to request, "help" to respond yes/no, "message" for
  general updates / capability sharing.
"""


# ── Strategy-specific system-prompt decorators ──────────────────────────────
# Each string is PREPENDED to REASONING_PROMPT_CORE when that strategy is
# active. Content adapted from the reference paper (ALFWorld → Search & Rescue).

_COT_ADDENDUM = (
    "[CoT] Solve the task step by step. Before choosing a tool, reason privately "
    "about the current world state, the active subtask, and the likely consequences "
    "of each candidate action. Then emit exactly one tool call.\n\n"
)

_REACT_ADDENDUM = (
    "[ReAct] Reason with a Thought → Action loop:\n"
    "  Thought: <brief reasoning about goal, observations, and constraints>\n"
    "Then call the single best action tool.\n\n"
)

_REFLEXION_ADDENDUM = (
    "[Reflexion] Before acting, reflect on what you have done and what failed. "
    "If a previous action failed (see critic_feedback.critique and reflexion block), "
    "you MUST try a completely different approach this turn — do not repeat the same "
    "action. Then call the single best action tool.\n\n"
)

_SELF_REFINE_ADDENDUM = (
    "[Self-Refine phase 1/2] Choose the best tool call for the active subtask. "
    "A second verification pass will check your call for syntactic and semantic "
    "correctness against the tool schema, so be precise with argument names and types.\n\n"
)

_SELF_REFLECTIVE_TOT_ADDENDUM = (
    "[Self-Reflective-ToT phase 1/2] Consider three candidate tool calls internally "
    "— each targeting a different reasonable sub-goal given the active subtask — then "
    "silently pick the one most likely to succeed and emit ONLY that final tool call.\n\n"
)


# ── Follow-up prompts for two-pass strategies ───────────────────────────────

_REFINE_FOLLOWUP_TEMPLATE = (
    "You are verifying a proposed tool call for a Search and Rescue agent.\n"
    "Proposed call: {tool_name}({args_json})\n"
    "Active subtask: {subtask}\n"
    "Agent position: {position}\n"
    "Available tool names: {tool_names}\n\n"
    "Check (a) the tool name is in the available list, (b) required arguments are "
    "present and of the correct type (coordinates are ints, object_ids are strings), "
    "(c) the call advances the subtask, (d) it is NOT a no-op (e.g. MoveTo to your "
    "own position is invalid).\n\n"
    "Respond with exactly one line in one of these two formats:\n"
    "  correct\n"
    "  error, revised: {{\"name\": \"<tool>\", \"args\": {{...}}}}\n"
)


def _format_reasoning_user_content(information: Dict[str, Any]) -> str:
    agent = information.get('agent_id', 'unknown')
    role = information.get('current_role', 'unassigned')
    capabilities = information.get('agent_capabilities', '')
    position = information.get('position', '?')
    carrying = information.get('carrying', 'nothing')
    current_plan = information.get('current_plan', '') or 'none'
    motivation = information.get('motivation', '') or ''
    critique = information.get('last_critique', '') or ''
    observation = information.get('observation', {}) or {}
    recent_actions = information.get('recent_actions', []) or []

    lines: List[str] = []

    # ── URGENT (mirrors Planning) ───────────────────────────────────────────
    urgent_help = information.get('urgent_help_request')
    if urgent_help:
        from_agent = urgent_help.get('from', 'teammate')
        req_text = urgent_help.get('text', '')
        lines.extend([
            "== URGENT: INCOMING HELP REQUEST ==",
            f'{from_agent} asks: "{req_text}"',
            "You MUST emit SendMessage this cycle:",
            '    message="yes" or "no" (exact lowercase, no extra text)',
            f'    send_to="{from_agent}"',
            '    message_type="help"',
            "====================================",
            "",
        ])

    my_acc = information.get('my_help_acceptance')
    if my_acc:
        lines.extend([
            "== YOU HAVE AN ACCEPTED HELP COMMITMENT ==",
            f"You said 'yes' to {my_acc.get('requester')}'s ask_help for "
            f"{my_acc.get('kind')} of {my_acc.get('target_id') or my_acc.get('victim_id')}.",
            "Your job is to fulfill this commitment — do NOT abandon it to "
            "explore or pursue a different task. If you are not already walking "
            "to the target or in a coop rendezvous, plan a MoveTo the target "
            "location and then CarryObjectTogether (if kind=carry) or "
            "RemoveObjectTogether (if kind=remove) with the requester.",
            "==========================================",
            "",
        ])

    active_req = information.get('active_help_request')
    if active_req:
        lines.extend([
            "== TEAM HELP-REQUEST LOCK ==",
            f"{active_req.get('requester')} has the active ask_help for victim "
            f"{active_req.get('victim_id')} at {active_req.get('victim_location')}.",
            "Do NOT emit a new SendMessage with message_type='ask_help'.",
            (f"Already assigned to {active_req.get('accepted_by')} — if you "
             f"previously said 'yes' you have lost the assignment and your "
             f"autonav has been cleared; resume your previous task."
             if active_req.get('accepted_by') else
             "Only the first 'yes' reply wins. Be decisive."),
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

    # ── LAST ACTION (most recent executed action, distinct from loop tail) ──
    last_action = information.get('last_action')
    if last_action:
        la_name = last_action.get('name', '?')
        la_args = last_action.get('args', {})
        la_outcome = last_action.get('outcome', '')
        lines.extend([
            "== LAST ACTION ==",
            f"Name: {la_name}",
            f"Args: {to_toon(la_args) if la_args else '{}'}",
            f"Outcome: {la_outcome}" if la_outcome else "Outcome: (not recorded)",
            "",
        ])

    # ── VALIDATION ERROR (from previous action, if any) ──────────────────────
    last_validation_error = information.get('last_validation_error', '')
    if last_validation_error:
        lines.extend([
            "== LAST ACTION REJECTED BY VALIDATOR ==",
            last_validation_error,
            "You must resolve this before the plan can proceed. Do NOT repeat the "
            "rejected action. Instead:",
            "  • If an obstacle blocks the path or door: call RemoveObject(object_id=<id>) "
            "or RemoveObjectTogether if your strength requires it.",
            "  • If you must navigate first: call MoveTo(x=<x>, y=<y>).",
            "  • If a capability issue: call SendMessage to request help.",
            "====================================",
            "",
        ])

    # ── CURRENT PLAN ─────────────────────────────────────────────────────────
    lines.extend([
        "== CURRENT PLAN (decided this tick by the planner — execute it with ONE tool call) ==",
        f'Plan:       "{current_plan}"',
        f'Motivation: "{motivation}"',
        f'Critique of last action (from planner): "{critique}"' if critique
            else 'Critique of last action: (none — last action succeeded or was a SendMessage)',
        "",
    ])

    # ── OBSERVATION ──────────────────────────────────────────────────────────
    lines.extend([
        "== OBSERVATION (vision range only) ==",
        to_toon({
            'victims': observation.get('victims', []),
            'teammates': observation.get('teammates', []),
            'obstacles': observation.get('obstacles', []),
        }),
        "",
    ])

    # ── AREA DOORS ────────────────────────────────────────────────────────────
    area_summaries = information.get('area_summaries') or []
    if area_summaries:
        lines.append("== AREA DOORS ==")
        for a in area_summaries:
            door_str = str(a['door']) if a.get('door') else "unknown"
            lines.append(f"  {a['name']}: door={door_str}")
        lines.append("")

    # ── INSTRUCTIONS: loop tail + reminder to emit one tool call ────────────
    lines.extend([
        "== RECENT ACTIONS (for loop detection) ==",
        to_toon(recent_actions) if recent_actions else "none",
        "",
        "Emit exactly one tool call now.",
    ])

    return '\n'.join(lines)


class FollowupRequest:
    """Returned by a reasoning strategy when it needs another LLM call.

    agent_sar._handle_reasoning_result interprets this by submitting the new
    LLM call and keeping the pipeline in the REASONING stage.
    """
    __slots__ = ('messages', 'tools', 'tool_choice')

    def __init__(self, messages: List[Dict[str, str]], tools=None, tool_choice: str = 'none'):
        self.messages = messages
        self.tools = tools
        self.tool_choice = tool_choice


class ActionCommit:
    """Returned by a reasoning strategy when it has settled on a final action."""
    __slots__ = ('name', 'args')

    def __init__(self, name: str, args: Dict[str, Any]):
        self.name = name
        self.args = args


# ── Base class ──────────────────────────────────────────────────────────────

class ReasoningBase:
    """Single-pass baseline: standard tool-call extraction, no prompt decoration.

    Subclasses override:
      - ``_decorate_system_prompt(core)`` to prepend strategy-specific instructions.
      - ``on_llm_result(message, agent)`` for multi-pass strategies; default is
        a no-op (return ``None`` → agent handles the tool call via the existing
        extraction path in _handle_reasoning_result).
    """

    # Number of phases this strategy runs per reasoning step. Overriden by
    # two-pass strategies. Used only for logging / metrics.
    num_phases: int = 1

    def __init__(self, plan: Any = None) -> None:
        self.plan = plan
        # per-reasoning-step FSM state, reset each time a new reasoning step starts
        self._phase: str = 'main'
        self._pending_action: Optional[Dict[str, Any]] = None

    def reset_phase(self) -> None:
        """Called by agent_sar at the start of each reasoning step."""
        self._phase = 'main'
        self._pending_action = None

    def _decorate_system_prompt(self, core: str) -> str:
        """Prepend strategy-specific instructions. Default: no decoration."""
        return core

    def get_reasoning_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        critic_feedback = information.get('critic_feedback', '')
        game_rules = information.get('game_rules', '')
        agent_capabilities = information.get('agent_capabilities', '')
        tools_available = information.get('tools_available', [])

        # Build system prompt: strategy decoration + core rules + game rules + capabilities
        decorated_core = self._decorate_system_prompt(REASONING_PROMPT_CORE)
        system_parts = [decorated_core]
        if game_rules:
            system_parts.append(game_rules)
        if agent_capabilities:
            system_parts.append(f"== YOUR CAPABILITIES ==\n{agent_capabilities}")
        if tools_available:
            system_parts.append(f"== AVAILABLE TOOLS ==\n" + ', '.join(tools_available))

        system_content = '\n\n'.join(system_parts)

        # Prepend a prominent warning when the critic flagged the last action as failed
        if isinstance(critic_feedback, dict) and critic_feedback.get('success') is False:
            critique_text = critic_feedback.get('critique', '')
            warning = (
                f"WARNING: Your last action FAILED. Critique: {critique_text}\n"
                f"You MUST choose a different action. Do NOT repeat the same action.\n\n"
            )
            system_content = warning + system_content

        return [
            {"role": "system", "content": system_content},
            # critic_feedback is already injected into the system prompt as a WARNING block;
            # including it in the user message too causes Qwen3 to pattern-match its
            # {reasoning/success/critique} structure and output that format instead of a tool call.
            {"role": "user",   "content": _format_reasoning_user_content(information)},
        ]

    # ── Multi-pass hook ───────────────────────────────────────────────────

    def on_llm_result(self, message, agent) -> Optional[Any]:
        """Hook for multi-pass strategies.

        Returns:
          None            — no special handling; let the agent run its default
                            tool_call extraction path.
          FollowupRequest — submit another LLM call and stay in REASONING stage.
          ActionCommit    — commit this (name, args) directly as the final action.
        """
        return None


# ── Concrete single-pass strategies ─────────────────────────────────────────

class ReasoningIO(ReasoningBase):
    """Baseline: no prompt decoration, single-pass tool call."""
    pass


class ReasoningCoT(ReasoningBase):
    def _decorate_system_prompt(self, core: str) -> str:
        return _COT_ADDENDUM + core


class ReasoningReAct(ReasoningBase):
    def _decorate_system_prompt(self, core: str) -> str:
        return _REACT_ADDENDUM + core


class ReasoningReflexion(ReasoningBase):
    def _decorate_system_prompt(self, core: str) -> str:
        return _REFLEXION_ADDENDUM + core


# ── Two-pass strategies ─────────────────────────────────────────────────────

def _extract_action_from_message(message, tools_by_name: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract {'name': ..., 'args': ...} from an LLM response message.

    Tries structured tool_calls first, then falls back to JSON extraction from
    free-form content. Returns None if nothing usable is found.
    """
    tool_calls = getattr(message, 'tool_calls', None)
    if tool_calls:
        tc = tool_calls[0]
        name = tc.function.name
        args_raw = tc.function.arguments
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except (json.JSONDecodeError, TypeError):
            args = {}
        if name in tools_by_name:
            return {'name': name, 'args': args}

    content = getattr(message, 'content', '') or ''
    if not content:
        return None

    # Fenced ```json block
    m = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict) and parsed.get('name') in tools_by_name:
                return {
                    'name': parsed['name'],
                    'args': parsed.get('args', parsed.get('arguments', {})),
                }
        except (json.JSONDecodeError, ValueError):
            pass

    # First { ... } span
    m = re.search(r'\{.*\}', content, re.DOTALL)
    if m:
        for loader in (json.loads, _try_ast_literal_eval):
            try:
                parsed = loader(m.group(0))
                if isinstance(parsed, dict) and parsed.get('name') in tools_by_name:
                    return {
                        'name': parsed['name'],
                        'args': parsed.get('args', parsed.get('arguments', {})),
                    }
            except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                continue
    return None


def _try_ast_literal_eval(text: str):
    import ast
    return ast.literal_eval(text)


class ReasoningSelfRefine(ReasoningBase):
    """Two-pass: main tool call → verify & optionally revise.

    Phase 'main':   submit standard reasoning prompt, extract action from response.
    Phase 'refine': submit a verifier prompt asking the LLM to either confirm or
                    patch the tool call. If the verifier outputs
                    'error, revised: {...}', replace the pending action.
    """
    num_phases = 2

    def _decorate_system_prompt(self, core: str) -> str:
        return _SELF_REFINE_ADDENDUM + core

    def on_llm_result(self, message, agent) -> Optional[Any]:
        if self._phase == 'main':
            extracted = _extract_action_from_message(message, agent.tools_by_name)
            if extracted is None:
                return None  # let default path handle retries
            self._pending_action = extracted

            subtask = (
                agent._pipeline_context.get('planned_task')
                or agent._current_task
                or ''
            )
            position = agent.WORLD_STATE.get('agent', {}).get('location')
            verify_user = _REFINE_FOLLOWUP_TEMPLATE.format(
                tool_name=extracted['name'],
                args_json=json.dumps(extracted['args']),
                subtask=subtask,
                position=position,
                tool_names=', '.join(sorted(agent.tools_by_name.keys())),
            )
            self._phase = 'refine'
            return FollowupRequest(
                messages=[
                    {"role": "system", "content": "You are a strict verifier of tool calls."},
                    {"role": "user", "content": verify_user},
                ],
                tools=None,
                tool_choice='none',
            )

        if self._phase == 'refine':
            content = (getattr(message, 'content', '') or '').strip()
            pending = self._pending_action or {}
            revised = _parse_refine_verdict(content, agent.tools_by_name)
            final = revised or pending
            self._phase = 'done'
            self._pending_action = None
            if not final.get('name'):
                return None
            return ActionCommit(final['name'], final.get('args', {}))

        return None


class ReasoningSelfReflectiveToT(ReasoningBase):
    """Two-pass: candidate-ensemble + internal vote, then a verify/refine pass.

    Phase 'main':   prompt asks the model to consider 3 candidates and emit the
                    best one (internal voting — keeps latency at a single LLM
                    call for this phase).
    Phase 'refine': same verify-&-revise pass as Self-Refine.
    """
    num_phases = 2

    def _decorate_system_prompt(self, core: str) -> str:
        return _SELF_REFLECTIVE_TOT_ADDENDUM + core

    def on_llm_result(self, message, agent) -> Optional[Any]:
        if self._phase == 'main':
            extracted = _extract_action_from_message(message, agent.tools_by_name)
            if extracted is None:
                return None
            self._pending_action = extracted

            subtask = (
                agent._pipeline_context.get('planned_task')
                or agent._current_task
                or ''
            )
            position = agent.WORLD_STATE.get('agent', {}).get('location')
            verify_user = _REFINE_FOLLOWUP_TEMPLATE.format(
                tool_name=extracted['name'],
                args_json=json.dumps(extracted['args']),
                subtask=subtask,
                position=position,
                tool_names=', '.join(sorted(agent.tools_by_name.keys())),
            )
            self._phase = 'refine'
            return FollowupRequest(
                messages=[
                    {"role": "system", "content": "You are a strict verifier of tool calls emitted by a Tree-of-Thought reasoner."},
                    {"role": "user", "content": verify_user},
                ],
                tools=None,
                tool_choice='none',
            )

        if self._phase == 'refine':
            content = (getattr(message, 'content', '') or '').strip()
            pending = self._pending_action or {}
            revised = _parse_refine_verdict(content, agent.tools_by_name)
            final = revised or pending
            self._phase = 'done'
            self._pending_action = None
            if not final.get('name'):
                return None
            return ActionCommit(final['name'], final.get('args', {}))

        return None


def _parse_refine_verdict(text: str, tools_by_name: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse the output of a refine/verify LLM call.

    Expected formats:
      'correct'                         → return None (keep pending)
      'error, revised: {"name": ..., "args": ...}'  → return the revised dict
    """
    if not text:
        return None
    lowered = text.lower().lstrip()
    if lowered.startswith('correct'):
        return None
    # Extract a JSON dict from anything after 'revised:'
    m = re.search(r'revised:\s*(\{.*\})', text, re.DOTALL)
    if not m:
        return None
    for loader in (json.loads, _try_ast_literal_eval):
        try:
            parsed = loader(m.group(1))
            if isinstance(parsed, dict) and parsed.get('name') in tools_by_name:
                return {
                    'name': parsed['name'],
                    'args': parsed.get('args', parsed.get('arguments', {})),
                }
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
    return None


# ── Registry ────────────────────────────────────────────────────────────────

REASONING_STRATEGY_REGISTRY: Dict[str, type] = {
    'io':                   ReasoningIO,
    'cot':                  ReasoningCoT,
    'react':                ReasoningReAct,
    'reflexion':            ReasoningReflexion,
    'self_refine':          ReasoningSelfRefine,
    'self_reflective_tot':  ReasoningSelfReflectiveToT,
}


def build_reasoning_strategy(name: str) -> ReasoningBase:
    """Instantiate a reasoning strategy by string key. Falls back to ReAct."""
    cls = REASONING_STRATEGY_REGISTRY.get(name, ReasoningReAct)
    return cls()
