import json
import re
from typing import Dict, List, Any, Optional
from helpers.toon_utils import to_toon


REASONING_PROMPT_CORE = """
You are a Search and Rescue agent. Your goal is to find and rescue as many victims as possible.

You are given a subtask. Return exactly one tool call to advance or complete it.
- Every tool call has a `task_completing` field. Set it to the exact subtask text if this action completes the subtask. Otherwise set it to "N/A".
- Before marking a task completed, verify from your observation that it is actually done.
- If your subtask involves sending a message, use SendMessage with the appropriate `message_type`.

Core rules:
- Return exactly one tool call — no natural language, no explanations.
- NEVER call MoveTo with coordinates matching your own `your_position` — that is a no-op.
- If `recent_actions` shows the same action 2+ times, you are looping — choose a completely different action type.
- If `critic_feedback` reports failure, you MUST try a DIFFERENT action per the `critique` field.
- Your action target must match the current subtask target (same object_id or location).

Joint action requirements:
- CarryObjectTogether / RemoveObjectTogether REQUIRE `partner_id` — use an `object_id` from observation.teammates.
- Both you and your partner must be adjacent (Chebyshev distance ≤ 1) to the target before calling a cooperative action.
- If no partner is adjacent, send an ask_help message via SendMessage and wait.

Teammate capabilities — you do NOT know what your teammates can do:
- You only know YOUR OWN capabilities (listed below under YOUR CAPABILITIES).
- You do NOT know your teammate's strength, medical skill, or vision range.
- If you need to coordinate on a task that depends on a teammate's capabilities (e.g. "can they carry this victim alone?"),
  send them a message asking: "What are your capabilities?" before assuming they can or cannot help.
- If a teammate asks about your capabilities, ALWAYS reply using SendMessage with a brief description
  (e.g. "I can carry mildly injured victims alone and have medium vision.").
- When you know a teammate's capabilities from a previous message, use that information to divide tasks efficiently.

Messaging:
- message_type="ask_help" to request assistance; "help" to respond; "message" for general updates or capability sharing.
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
        observation = information.get('observation', {})
        task_decomposition = information.get('task_decomposition', '')
        memory = information.get('memory', '') or 'none'
        critic_feedback = information.get('critic_feedback', '')
        last_critique = information.get('last_critique', '')
        recent_actions = information.get('recent_actions', [])
        game_rules = information.get('game_rules', '')
        agent_capabilities = information.get('agent_capabilities', '')
        role_prompt = information.get('role_prompt', '')
        tools_available = information.get('tools_available', [])

        your_position = observation.get('agent', {}).get('location')

        info_dict: Dict[str, Any] = {
            "current_subtask": task_decomposition,
            "your_position": your_position,
            "observation": observation,
            "memory": memory,
            "recent_actions": recent_actions,
        }
        # critic_feedback is already injected into the system prompt as a WARNING block;
        # including it in the user message too causes Qwen3 to pattern-match its
        # {reasoning/success/critique} structure and output that format instead of a tool call.
        if last_critique:
            info_dict["last_critique"] = last_critique

        # Build system prompt: strategy decoration + core rules + game rules + capabilities
        decorated_core = self._decorate_system_prompt(REASONING_PROMPT_CORE)
        system_parts = [decorated_core]
        if game_rules:
            system_parts.append(game_rules)
        if agent_capabilities:
            system_parts.append(f"== YOUR CAPABILITIES ==\n{agent_capabilities}")
        if role_prompt:
            system_parts.append(f"== YOUR ROLE ==\n{role_prompt}")
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
            {"role": "user",   "content": to_toon(info_dict)},
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
