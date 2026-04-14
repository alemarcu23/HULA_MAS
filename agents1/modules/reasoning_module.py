import json
from typing import Dict, List, Any
from helpers.toon_utils import to_toon


REASONING_PROMPT = """
You are a Search and Rescue agent. Your goal is to find and rescue as many victims as possible.

You are given a subtask. Return exactly one tool call to advance or complete it.
- Every tool call has a `task_completing` field. Set it to the exact subtask text if this action completes the subtask. Otherwise set it to "N/A".
- Before marking a task completed, verify from your observation that it is actually done.
- If your subtask involves sending a message, use SendMessage with the appropriate `message_type` ("ask_help", "help", or "message").

Core rules:
- Do not output natural language, explanations, reasoning, or multiple options.
- Return exactly one tool call.
- Choose an action that directly advances the current subtask.
- NEVER select MoveTo with coordinates matching your `your_position` — that is a no-op and wastes a tick.
- Do not repeat any of your last 3 actions (visible in `recent_actions`) if a different valid action is available.
- If `recent_actions` shows the same action repeated 2+ times, you are in a loop — choose a completely different action type.
- If `critic_feedback` reports failure, you MUST try a DIFFERENT action. Read the `critique` field for what to do instead.
- Your action target must match your current_subtask target (same object ID or location).
- Use only information that is available in the current observation and memory.
- Do not assume a task is complete unless the observation confirms it.

Joint action mechanics:
- CarryObjectTogether / RemoveObjectTogether require specifying a partner_id — use a teammate ID from the 'teammates' list in your observation.
- The specified partner must be adjacent (within 1 block) to the target object.
- If no partner is adjacent, send an ask_help message and wait (Idle) until they arrive.
- If a teammate sent you an ask_help message, navigate to their location to assist.
- Once a cooperative carry succeeds, both agents are automatically moved to the drop zone — no further action needed.

Messaging:
- If the subtask involves communication, use `SendMessage`.
- Use `message_type="ask_help"` to request assistance, `"help"` to respond to assistance, and `"message"` otherwise.

Navigation: your current position is given in `your_position` (x, y). Use it to compute relative distances when choosing MoveTo coordinates.

Input fields reference:
- current_subtask: The specific task you must complete right now
- your_position: Your current [x, y] coordinates on the grid — NEVER use these as a MoveTo target
- observation.agent: Your state (location, carrying)
- observation.victims: Victims within your vision range
- observation.obstacles: Obstacles within your vision range
- observation.teammates: Other agents and their positions
- observation.known: All objects discovered so far (global memory)
- observation.area_exploration: Coverage % per area
- memory: Your recent actions, messages, and critic feedback (most recent last)
- critic_feedback: Evaluation of your LAST action (success/failure + critique) — if success=false, follow the critique
- recent_actions: Your last 3 actions — if all identical, you are looping; choose something different

Output exactly one tool call with all required fields.
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

        your_position = observation.get('agent', {}).get('location')

        info_dict: Dict[str, Any] = {
            "current_subtask": task_decomposition,
            "your_position": your_position,
            "observation": observation,
            "memory": memory,
            "critic_feedback": critic_feedback,
            "recent_actions": recent_actions,
        }

        system_content = REASONING_PROMPT
        # Prepend a prominent warning when the critic flagged the last action as failed
        if isinstance(critic_feedback, dict) and critic_feedback.get('success') is False:
            critique_text = critic_feedback.get('critique', '')
            warning = (
                f"WARNING: Your last action FAILED. Critique: {critique_text}\n"
                f"You MUST choose a different action. Do NOT repeat the same action.\n\n"
            )
            system_content = warning + REASONING_PROMPT

        print("Generating reasoning prompt with information:", json.dumps(info_dict, indent=2))
        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": to_toon(info_dict)},
        ]
