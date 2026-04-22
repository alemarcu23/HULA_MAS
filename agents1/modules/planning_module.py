import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
from helpers.toon_utils import to_toon

try:
    from engine.parsing_utils import load_few_shot
except ImportError:
    def load_few_shot(key):
        return []

logger = logging.getLogger('Planning')

TASK_DECOMPOSITION_PROMPT = """
You are an expert planner for a search and rescue agent.
Break the given task into 3-7 atomic subtasks. Each subtask must be verifiable as complete or incomplete from a single observation (e.g., "am I at location X?", "am I carrying victim Y?").

Include target IDs and coordinates from the observed objects when available.

DAG conditional syntax (use when outcome is uncertain):
- Write subtasks as numbered steps.
- To add a conditional branch, end the step with ". If <condition>, <alternative action>."
  Example: "3. Remove stone_1 blocking the door. If stone_1 requires cooperative removal, send ask_help to teammate."
- The TaskGraph parser will split conditional text into a node + condition_action.

Respond with ONLY a numbered list, one subtask per line. No explanations or preamble.
Example:
1. Navigate to area 1 entrance at [3, 4].
2. Remove any obstacle blocking the door. If obstacle is a big rock, send ask_help and wait for teammate.
3. Search area 1 for victims.
4. Carry mildly_injured_woman to drop zone.
5. Navigate to drop zone.
6. Drop the victim at the drop zone.
"""

PLANNER_PROMPT = """
You are the task planner for a search-and-rescue agent.

Your job is to output the single best NEXT task for this agent only.
The overall mission is to rescue all victims from all areas and deliver them to the drop zone.

Rules:
- Output exactly ONE single-sentence task.
- The task must be immediately actionable by this agent.
- Do NOT output explanations, reasoning, lists, or multiple options.
- Do NOT assign tasks that are already completed or currently being executed by another agent.
- Include object IDs, agent IDs, area names, and [x, y] coordinates whenever known.

Output style:
- Return exactly one sentence.
- Mention the target and location when known.

Good examples:
- "Go to [8, 6] and help rescuebot0 carry victim_3 together."
- "Deliver victim_2 to the drop zone."
- "Pick up victim_1 at [2, 5] alone."
- "Send ask_help for victim_3 at [8, 6] and wait adjacent to it."
- "Remove the rock blocking Area 2 at [4, 7]."
- "Explore Area 1."

Bad outputs:
- "I think the best plan is to first explore and then maybe rescue someone."
- "Help teammate, or explore Area 2 if that fails."
- "The next action depends on what the other agents do."

Anti-self-assignment rule:
- NEVER assign a cooperative task where the only named partner is yourself (agent_id).
- Cooperative tasks (CarryObjectTogether, RemoveObjectTogether) must name a DIFFERENT agent as the partner.

Teammate capabilities — you do NOT know what your teammates can do:
- You only know YOUR OWN capabilities (see agent_capabilities field).
- You do NOT know your teammate's vision range, strength, or medical skill.
- Before assigning a cooperative task to a teammate, or before relying on them for a specific role,
  you should ASK them about their capabilities via SendMessage.
- Example: send a message like "What are your capabilities? Can you carry critically injured victims alone?"
- If you have received a teammate's capability info in messages, use it to coordinate better.
- If you need help with something you cannot do alone and don't know if your teammate can, ask first.

Input fields reference:
- agent_id: YOUR agent identifier — never assign cooperative tasks with yourself as partner
- position: Your current [x, y] location
- current_task: The high-level mission task assigned to you
- game_rules: Official rules for carrying victims and removing obstacles — always follow these
- critic_feedback: Whether your last action succeeded or failed — if failed, adjust the task
- previous_tasks: Your recent task history — avoid repeating failed tasks
- nearby_objects: Victims and obstacles within your vision range
- observed_objects: ALL objects ever discovered across the map
- rescued_victims: Victims already delivered to the drop zone — do not re-rescue these
- area_exploration: Coverage % per area — prioritize areas with lowest coverage
- messages: Recent messages from teammates — respond to ask_help requests first, and to capability questions
- agent_capabilities: YOUR physical capabilities — only assign tasks feasible for you
"""

_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'to', 'at', 'for', 'and', 'or', 'is', 'in', 'of',
    'it', 'its', 'this', 'that', 'be', 'by', 'on', 'with', 'from',
})


class Planning:
    def __init__(self, mode: str = 'simple') -> None:
        self.mode = mode
        self.task_decomposition: List[SubTask] = []
        self.task_graph: Optional[TaskGraph] = None
        self.current_task = ''

    def set_current_task(self, task: str) -> None:
        self.current_task = task

    def set_task_decomposition(self, decomposition: str) -> None:
        self.task_decomposition = decomposition

    def set_manual_task_decomposition(self, decomposition: List[str]) -> None:
        """Set plan from a list of task description strings."""
        self.task_decomposition = [SubTask(desc) for desc in decomposition]
        if self.task_decomposition:
            self.task_decomposition[0].status = TaskStatus.ACTIVE
        if self.mode == 'dag':
            self.task_graph = TaskGraph.from_task_list(decomposition)
                
    def get_planning_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        print("Generating planning prompt with information:", json.dumps(information, indent=2, default=str))

        # Inject capability-aware game rules into the system prompt
        game_rules = information.get('game_rules', '')
        system_content = PLANNER_PROMPT
        if game_rules:
            system_content = f"{PLANNER_PROMPT}\n\n{game_rules}"

        messages = [{"role": "system", "content": system_content}]

        # Load few-shot examples for next-task planning
        try:
            examples = load_few_shot('planning_next_task')
            for ex in examples:
                if 'user' in ex and 'assistant' in ex:
                    messages.append({"role": "user", "content": ex['user'].strip()})
                    messages.append({"role": "assistant", "content": ex['assistant'].strip()})
        except Exception:
            pass

        messages.append({"role": "user", "content": to_toon(information)})
        return messages

    def get_task_decomposition_prompt(
        self, information: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        world_state = information.get('world_state', {})
        memory = information.get('memory', '') or 'none'
        feedback = information.get('feedback', '') or 'none'

        info_dict: Dict[str, Any] = {
            "task": self.current_task,
            "world_state": world_state,
            "memory": memory,
            "feedback": feedback,
        }

        return [
            {"role": "system", "content": TASK_DECOMPOSITION_PROMPT},
            {"role": "user",   "content": to_toon(info_dict)},
        ]
    
    def advance_task(self, task_completing: str = '') -> None:
        """Advance the active task if task_completing matches it."""
        if not task_completing:
            return
        if self.mode == 'dag':
            self._advance_dag(task_completing)
        else:
            self._advance_simple(task_completing)

    def _advance_simple(self, task_completing: str) -> None:
        active = next(
            (st for st in self.task_decomposition if st.status == TaskStatus.ACTIVE),
            None,
        )
        if active is None:
            return
        if _is_task_match(task_completing, active.description):
            active.status = TaskStatus.COMPLETED
            for st in self.task_decomposition:
                if st.status == TaskStatus.PENDING:
                    st.status = TaskStatus.ACTIVE
                    break
            logger.info("Task completed: '%s' (matched '%s')", active.description, task_completing)

    def _advance_dag(self, task_completing: str) -> None:
        if not self.task_graph:
            return
        node = self.task_graph.get_current_task()
        if node is None:
            return
        if _is_task_match(task_completing, node.description):
            self.task_graph.advance()



class TaskStatus(Enum):
    PENDING = 'pending'
    ACTIVE = 'active'
    COMPLETED = 'completed'
    SKIPPED = 'skipped'

@dataclass
class SubTask:
    description: str
    status: TaskStatus = TaskStatus.PENDING

    def __str__(self) -> str:
        return self.description

@dataclass
class TaskNode:
    """A single node in the task graph."""
    id: int
    description: str
    status: TaskStatus = TaskStatus.PENDING
    is_condition: bool = False
    condition_action: str = ''
    next_id: Optional[int] = None

    def full_description(self) -> str:
        if self.is_condition and self.condition_action:
            return f"{self.description}. If so, {self.condition_action}"
        return self.description


_COND_INLINE = re.compile(r'^(.*?)\.\s*[Ii]f\s+(.+?),\s*(.+)$')
_COND_SUBBULLET = re.compile(r'\n\s*-\s*[Ii]f\s+(.+?),\s*(.+)$')


class TaskGraph:
    """Directed acyclic graph of tasks with conditional branching support."""

    def __init__(self) -> None:
        self._nodes: Dict[int, TaskNode] = {}
        self._head_id: Optional[int] = None

    @classmethod
    def from_task_list(cls, tasks: List[str]) -> 'TaskGraph':
        graph = cls()
        if not tasks:
            return graph

        nodes: List[TaskNode] = []
        for idx, raw in enumerate(tasks):
            node_id = idx + 1
            m = _COND_SUBBULLET.search(raw)
            if m:
                desc = raw[:m.start()].strip().rstrip('.')
                nodes.append(TaskNode(
                    id=node_id, description=desc,
                    is_condition=True, condition_action=m.group(2).strip(),
                ))
                continue
            m = _COND_INLINE.match(raw.strip())
            if m:
                nodes.append(TaskNode(
                    id=node_id, description=m.group(1).strip(),
                    is_condition=True, condition_action=m.group(3).strip(),
                ))
                continue
            nodes.append(TaskNode(id=node_id, description=raw.strip()))

        for i in range(len(nodes) - 1):
            nodes[i].next_id = nodes[i + 1].id
        for node in nodes:
            graph._nodes[node.id] = node
        if nodes:
            graph._head_id = nodes[0].id
            nodes[0].status = TaskStatus.ACTIVE

        return graph

    def get_current_task(self) -> Optional[TaskNode]:
        if self._head_id is None:
            return None
        return self._nodes.get(self._head_id)

    def advance(self) -> None:
        """Mark current task completed, remove it, activate next."""
        node = self._nodes.get(self._head_id)
        if node is None:
            return
        node.status = TaskStatus.COMPLETED
        next_id = node.next_id
        del self._nodes[node.id]
        self._head_id = next_id
        if next_id is not None and next_id in self._nodes:
            self._nodes[next_id].status = TaskStatus.ACTIVE
        else:
            self._head_id = None
        logger.info("DAG task completed: '%s' | next: %s", node.description, next_id)

    def is_empty(self) -> bool:
        return len(self._nodes) == 0

    def __repr__(self) -> str:
        parts = []
        for nid, node in sorted(self._nodes.items()):
            marker = '>' if node.status == TaskStatus.ACTIVE else ' '
            cond = ' [COND]' if node.is_condition else ''
            parts.append(f"  {marker} {node.id}. {node.description}{cond} ({node.status.value})")
        return "TaskGraph:\n" + "\n".join(parts) if parts else "TaskGraph: (empty)"


def _is_task_match(task_completing: str, task_description: str) -> bool:
    """Check if task_completing keywords overlap >=50% with task description."""
    tc_words = set(task_completing.lower().split()) - _STOP_WORDS
    td_words = set(task_description.lower().split()) - _STOP_WORDS
    if not tc_words:
        return False
    return len(tc_words & td_words) / len(tc_words) >= 0.5

