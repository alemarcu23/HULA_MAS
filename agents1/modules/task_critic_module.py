from typing import Dict, List, Any
from helpers.toon_utils import to_toon


CRITIC_PROMPT = """
You are an assistant that evaluates progress in a search and rescue mission.
Assess whether the last action advanced the current subtask. Exceeding requirements counts as success.

When the action fails, your critique MUST be actionable: state what went wrong, suggest a specific next action to try, and note any preconditions to verify first.

Respond with valid JSON only:
{"reasoning": "what happened and why", "success": true/false, "critique": "actionable next step if failed, empty string if succeeded"}

Examples:

INPUT:
Position: [3,5], Carrying: mildly_injured_woman, Nearby: none
Last action: CarryObject(object_id="mildly_injured_woman")
Task: Pick up victim at location [3, 5]
RESPONSE:
{"reasoning": "At [3, 5], carrying the victim, no objects nearby. CarryObject succeeded.", "success": true, "critique": ""}

INPUT:
Position: [3,5], Carrying: None, Nearby: mildly_injured_woman [3, 7]
Last action: CarryObject(object_id="mildly_injured_woman")
Task: Pick up victim at location [3, 7]
RESPONSE:
{"reasoning": "You are not carrying the victim that you included int the CarryObject action.", "success": false, "critique": "You are too far away from the victim. Move closer"}

INPUT:
Position: [3,5], Carrying: None, Nearby: mildly_injured_woman
Last action: CarryObject(object_id="mildly_injured_woman")
Task: Pick up victim at location [3, 5]
RESPONSE:
{"reasoning": "Victim still on ground despite CarryObject attempt. Action failed.", "success": false, "critique": "CarryObject failed — check if this victim requires cooperative carry (CarryObjectTogether) due to severity. Retry or ask teammate for help."}

INPUT:
Position: [15,5], Carrying: None, Nearby: stone_1 at [15,4]
Last action: RemoveObject(object_id="stone_1")
Task: Remove obstacle blocking the door of Area 3
RESPONSE:
{"reasoning": "stone_1 still in nearby objects after RemoveObject — removal failed.", "success": false, "critique": "Solo removal failed. Try RemoveObjectTogether with a teammate, or check if you need to be adjacent to the object first."}

INPUT:
Position: [5,10], Carrying: None, Nearby: critically_injured_man at [5,10]
Last action: CarryObjectTogether(object_id="critically_injured_man")
Task: Carry critical victim at [5, 10] to drop zone cooperatively
RESPONSE:
{"reasoning": "CarryObjectTogether failed — no teammate adjacent. Critical victims need two agents.", "success": false, "critique": "Send ask_help message via SendMessage with victim ID and your location. Wait for teammate to arrive before retrying CarryObjectTogether."}

INPUT:
Position: [9,5], Carrying: None, Nearby: none
Last action: MoveTo(x=9, y=4)
Task: Explore Area 2
RESPONSE:
{"reasoning": "Agent moved to [9,5] but task is to explore Area 2. Movement happened but area exploration hasn't started yet — this is progress toward the task, not completion.", "success": false, "critique": "You are near Area 2 entrance at [9, 4]. Enter the area and search for victims inside."}
"""


class CriticBase:
    def __init__(self, plan):
        self.plan = plan

    def get_critic_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        observation = information.get('observation', {})
        current_task = information.get('current_task', '')
        all_observations = information.get('all_observations', '')
        last_action = information.get('last_action', {})

        info_dict: Dict[str, Any] = {
            "current_task": current_task,
            "last_action": last_action,
            "observation": observation,
            "all_observations": all_observations,
        }
        print(to_toon(info_dict))

        return [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user",   "content": to_toon(info_dict)},
        ]