import json
from typing import Dict, List, Any
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

Messaging:
- message_type="ask_help" to request assistance; "help" to respond; "message" for general updates.
"""


class ReasoningBase:
    def __init__(self, plan):
        self.plan = plan


class ReasoningIO(ReasoningBase):
    def get_reasoning_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        observation = information.get('observation', {})
        task_decomposition = information.get('task_decomposition', '')
        memory = information.get('memory', '') or 'none'
        critic_feedback = information.get('critic_feedback', '')
        recent_actions = information.get('recent_actions', [])
        game_rules = information.get('game_rules', '')
        agent_capabilities = information.get('agent_capabilities', '')
        tools_available = information.get('tools_available', [])

        your_position = observation.get('agent', {}).get('location')

        info_dict: Dict[str, Any] = {
            "current_subtask": task_decomposition,
            "your_position": your_position,
            "observation": observation,
            "memory": memory,
            "critic_feedback": critic_feedback,
            "recent_actions": recent_actions,
        }

        # Build system prompt: core rules + game rules + capabilities
        system_parts = [REASONING_PROMPT_CORE]
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

        print("Generating reasoning prompt with information:", json.dumps(info_dict, indent=2, default=str))
        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": to_toon(info_dict)},
        ]
