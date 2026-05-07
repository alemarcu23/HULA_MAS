import json
import logging
import re
import ast
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

CRITIC_PLAN_PROMPT = """
You are a combined evaluator and task planner for a search-and-rescue agent.
## STEP 1: EVALUATE LAST ACTION
Assess whether the `last_action` advanced the goal.
*   SUCCESS: The action achieved or exceeded its intent. (Note: If `last_action` is empty or null, treat it as a success).
*   FAILURE: The action did not achieve its intent. 
       Rule: A `MoveTo` action where the target coordinates exactly match the agent's current position is ALWAYS a no-op failure. However, this might be because something is blocking the path. You should check for this next instead of trying the same plan again.
*   CRITIQUE: If an action fails, your critique MUST be actionable. State what went wrong, suggest a specific fix, and note preconditions to verify (e.g., "Check partner_id", "Choose new destination").\=

Examples:
INPUT: Position [3,5], Carrying mildly_injured_woman, Last action: CarryObject(object_id="mildly_injured_woman"), Task: Pick up mild victim at [3, 5]
OUT: success=true, critique=""

INPUT: Position [5,3], Last action: MoveTo(x=5, y=3), Task: Navigate to victim at [8, 7]
OUT: success=false, critique="MoveTo target (5,3) equals current position — no-op. Choose a different destination toward [8, 7]."

INPUT: Position [5,10], Carrying: None, Last action: CarryObjectTogether(object_id="critically_injured_man"), Task: Carry critical victim cooperatively
OUT: success=false, critique="CarryObjectTogether failed — check that teammate is adjacent."

## STEP 2: PLAN NEXT TASK

Based on the evaluation above and the world state, output the single best NEXT task for yourself based on your role. IF the critique from STEP 1 indicates a failure, the next plan should
be different from the previous one and address the failure. For example, if the last action was a failed MoveTo, check for obstacles in the observations and plan to remove them or choose a different area to search.
If you are a searching agent, you should focus on exploring new areas, especially those with low coverage. If you are a rescuing agent, you should focus on picking up or delivering victims. 
If you are a heavy_duty agent, you should focus on removing obstacles or assisting teammates.
You are free to ignore your role for the next task, but you should choose actions that fit the current situation, your agent's capabilities and your team's needs.

Rules:
- Do NOT assign tasks that are already completed or currently being executed by another agent.
- Include object IDs, agent IDs, area names, and [x, y] coordinates whenever known.
- Victims listed in history.rescued_victims are already safe — NEVER plan to rescue, carry, or interact with them.
- If the same task appears 3 or more times in history.previous_tasks without progress, switch to a completely different task: explore a new room or area not yet covered (use history.area_coverage to find uncovered areas).

## INPUT DATA REFERENCE
*   `agent_id`: Your name.
*   `position`: Your current [x,y] location.
*   `current_task`: Your high-level mission goal.
*   `critic_feedback`: Feedback on your last action. If it failed, adjust your next task to fix it.
*   `previous_tasks`: Your recent history (avoid repeating past tasks).
*   `observed_objects`: Known map states.
*   `rescued_victims`: Victims already saved (do not re-rescue).
*   `area_exploration`: Coverage % (prioritize lowest coverage areas).

Good examples:
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


# ── Strategy-specific prompt addenda ────────────────────────────────────────
# Each strategy prepends a short instruction block to the base prompt,
# adapted from the reference paper (ALFWorld → Search & Rescue).

_STRATEGY_ADDENDA: Dict[str, str] = {
    'io': "",  # baseline
    'deps': (
        "[DEPS] You are a helper AI agent. Generate a sequence of sub-goals "
        "(actions) for the search-and-rescue task below. Reason about each "
        "sub-goal and the tool call it requires. Prefer decompositions that "
        "thread through explicit multi-hop intermediate states (e.g. reach "
        "door → search area → pick victim → reach drop zone → drop).\n\n"
    ),
    'td': (
        "[TD] Produce a plan with explicit temporal dependencies: each "
        "subtask must list which prior subtasks must be completed first. "
        "This avoids parallel conflicts and out-of-order execution.\n\n"
    ),
    'voyager': (
        "[Voyager] You are a helpful assistant that generates sub-goals to "
        "complete the search-and-rescue task. Criteria:\n"
        "  1) Return a list of sub-goals executable in order.\n"
        "  2) For each sub-goal give a brief reasoning and the tool to call.\n"
        "Keep the list minimal and executable.\n\n"
    ),
}


# ── Strategy classes ────────────────────────────────────────────────────────

class PlanningBase:
    """Base planning strategy — no prompt decoration, IO-style behavior.

    Subclasses override ``prefix`` to prepend strategy-specific instructions
    to the base planning/decomposition prompt.
    """

    prefix: str = _STRATEGY_ADDENDA['io']

    def decorate_system_prompt(self, base_prompt: str) -> str:
        return (self.prefix + base_prompt) if self.prefix else base_prompt

    @staticmethod
    def parse_decomposition_dicts(text: str) -> List[Dict[str, Any]]:
        """Extract `{...}` dicts from the LLM output (reference-paper parser)."""
        dict_strings = re.findall(r"\{[^{}]*\}", text)
        dicts: List[Dict[str, Any]] = []
        for ds in dict_strings:
            try:
                parsed = ast.literal_eval(ds)
                if isinstance(parsed, dict):
                    dicts.append(parsed)
            except (ValueError, SyntaxError):
                continue
        return dicts


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


_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'to', 'at', 'for', 'and', 'or', 'is', 'in', 'of',
    'it', 'its', 'this', 'that', 'be', 'by', 'on', 'with', 'from',
})


class Planning:
    def __init__(self, mode: str = 'simple', strategy: str = 'io') -> None:
        self.mode = mode
        self.strategy_name = strategy
        self.strategy = build_planning_strategy(strategy)
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

    def get_active_subtask_description(self) -> Optional[str]:
        """Return the current ACTIVE subtask description, or None."""
        if self.mode == 'dag' and self.task_graph is not None:
            node = self.task_graph.get_current_task()
            return node.description if node is not None else None
        active = next(
            (st for st in self.task_decomposition if st.status == TaskStatus.ACTIVE),
            None,
        )
        return active.description if active is not None else None

    def is_fully_completed(self) -> bool:
        """True iff no ACTIVE or PENDING subtask remains (plan drained)."""
        if self.mode == 'dag':
            if self.task_graph is None:
                return True
            return self.task_graph.is_empty()
        if not self.task_decomposition:
            return True
        return not any(
            st.status in (TaskStatus.ACTIVE, TaskStatus.PENDING)
            for st in self.task_decomposition
        )

    def advance_active_task(self) -> Optional[str]:
        """Force-advance the current ACTIVE subtask (used by critic-gated flow).

        Unlike ``advance_task(task_completing)``, this does not require a
        string match — it unconditionally marks the active node complete and
        activates the next one. Returns the description of the task that was
        advanced, or None if nothing was active.
        """
        if self.mode == 'dag':
            if self.task_graph is None:
                return None
            node = self.task_graph.get_current_task()
            if node is None:
                return None
            desc = node.description
            self.task_graph.advance()
            return desc
        active = next(
            (st for st in self.task_decomposition if st.status == TaskStatus.ACTIVE),
            None,
        )
        if active is None:
            return None
        desc = active.description
        active.status = TaskStatus.COMPLETED
        for st in self.task_decomposition:
            if st.status == TaskStatus.PENDING:
                st.status = TaskStatus.ACTIVE
                break
        logger.info("Critic-gated advance: '%s'", desc)
        return desc

    def get_planning_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        # Use CRITIC_PLAN_PROMPT as the base — it handles both evaluation and planning
        # in one call. Strategy decorator prepends any strategy-specific prefix.
        system_content = self.strategy.decorate_system_prompt(CRITIC_PLAN_PROMPT)

        messages = [{"role": "system", "content": system_content}]

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

        system_content = self.strategy.decorate_system_prompt(TASK_DECOMPOSITION_PROMPT)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        try:
            examples = load_few_shot('planning_decompose')
            for ex in examples:
                if 'user' in ex and 'assistant' in ex:
                    messages.append({"role": "user", "content": ex['user'].strip()})
                    messages.append({"role": "assistant", "content": ex['assistant'].strip()})
        except Exception:
            pass

        messages.append({"role": "user", "content": to_toon(info_dict)})
        return messages

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
