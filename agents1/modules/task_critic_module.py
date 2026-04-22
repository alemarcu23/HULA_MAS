from typing import Dict, List, Any
from helpers.toon_utils import to_toon, to_toon_sectioned


CRITIC_PROMPT_BASE = """
You are an assistant that evaluates progress in a search and rescue mission.
Assess whether the last action advanced the current subtask. Exceeding requirements counts as success.

When the action fails, your critique MUST be actionable: state what went wrong, suggest a specific next action to try, and note any preconditions to verify first.

Special rule: A MoveTo where the target coordinates match the agent's current position is ALWAYS a no-op failure, even if the move technically executed. Mark it success=false with critique explaining the agent must choose a different destination.

Respond with valid JSON only:
{"reasoning": "what happened and why", "success": true/false, "critique": "actionable next step if failed, empty string if succeeded"}

Examples:

INPUT:
Position: [3,5], Carrying: mildly_injured_woman
Last action: CarryObject(object_id="mildly_injured_woman")
Task: Pick up mild victim at [3, 5]
RESPONSE:
{"reasoning": "At [3,5], now carrying victim. CarryObject succeeded.", "success": true, "critique": ""}

INPUT:
Position: [5,10], Carrying: None, Nearby victims: critically_injured_man at [5,10]
Last action: CarryObjectTogether(object_id="critically_injured_man")
Task: Carry critical victim cooperatively
RESPONSE:
{"reasoning": "CarryObjectTogether failed — check partner_id was provided and teammate is adjacent.", "success": false, "critique": "Ensure partner_id is set to a teammate object_id from observation.teammates, and both agents are adjacent to the victim before retrying."}

INPUT:
Position: [5,3], Carrying: None
Last action: MoveTo(x=5, y=3)
Task: Navigate to victim at [8, 7]
RESPONSE:
{"reasoning": "MoveTo target (5,3) equals current position — no-op.", "success": false, "critique": "You moved to your own position. Choose a different destination toward [8, 7]."}
"""


class CriticBase:
    def __init__(self, plan):
        self.plan = plan

    def get_critic_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        observation = information.get('observation', {})
        current_task = information.get('current_task', '')
        all_observations = information.get('all_observations', '')
        last_action = information.get('last_action', {})
        game_rules = information.get('game_rules', '')
        agent_capabilities = information.get('agent_capabilities', '')

        # Inject game rules + capabilities into the system prompt so the critic
        # can judge capability-specific failures (e.g. low-medical solo carry).
        system_content = CRITIC_PROMPT_BASE
        if game_rules:
            system_content += f"\n\n{game_rules}"
        if agent_capabilities:
            system_content += f"\n\n== AGENT CAPABILITIES ==\n{agent_capabilities}"

        info_dict: Dict[str, Any] = {
            "current_task": current_task,
            "last_action": last_action,
            "observation": observation,
            "all_observations": all_observations,
        }
        print(to_toon(info_dict))

        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": to_toon(info_dict)},
        ]
