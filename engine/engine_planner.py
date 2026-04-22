"""
EnginePlanner — Central coordinator agent for multi-agent task planning.

A minimal MATRX agent (invisible, idle-only) that coordinates rescue agents
via MATRX's native Message system. Uses the same module architecture as
rescue agents (planning, reasoning, communication, area_tracker — no critic).

The planner manages an iteration-based lifecycle:
    NEEDS_PLANNING → PLANNING_IN_PROGRESS → EXECUTING →
    NEEDS_SUMMARIZATION → SUMMARIZING → (loop)

All LLM calls are non-blocking: dispatched to the shared thread pool,
polled each tick in decide_on_action.
"""

import json
import logging
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

import yaml

from brains1.ArtificialBrain import ArtificialAgentBrain
from matrx.messages.message import Message

from agents1.async_model_prompting import call_llm_sync, _get_executor
from agents1.modules.area_tracker import AreaExplorationTracker
from agents1.modules.communication_module import CommunicationModule
from agents1.modules.planning_module import Planning
from agents1.modules.reasoning_module import ReasoningIO
from engine.iteration_data import IterationData
from engine.parsing_utils import parse_json_response, load_few_shot
from helpers.toon_utils import to_toon
from memory.base_memory import BaseMemory
from worlds1.environment_info import EnvironmentInformation

logger = logging.getLogger('EnginePlanner')

# Load iteration-level LLM prompts (task generation, summarization, Q&A).
_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), 'prompts_engine_planner.yaml')
with open(_PROMPTS_FILE, 'r') as _f:
    PROMPTS = yaml.safe_load(_f)

# Phase constants for the iteration state machine.
NEEDS_PLANNING = 'needs_planning'
PLANNING_IN_PROGRESS = 'planning_in_progress'
EXECUTING = 'executing'
NEEDS_SUMMARIZATION = 'needs_summarization'
SUMMARIZING = 'summarizing'


class EnginePlanner(ArtificialAgentBrain):
    """Central coordinator agent that assigns tasks to rescue agents.

    Registered as a MATRX agent with full visibility, separate team ("planner"),
    and no physical actions (always idles). Communicates exclusively via Messages.

    Args:
        llm_model:           Ollama model name for LLM calls.
        ticks_per_iteration: MATRX ticks per planning iteration.
        max_iterations:      Maximum planning iterations before stopping.
        score_file:          Path to score.json for termination checks.
        include_human:       Whether a human agent is present.
        api_base:            Ollama base URL.
        manual_plans_file:   Optional YAML file to override LLM task generation.
        planning_mode:       'simple' or 'dag' (task decomposition strategy).
        env_info:            EnvironmentInformation for area tracker.
    """

    def __init__(
        self,
        llm_model: str = 'qwen3:8b',
        ticks_per_iteration: int = 1200,
        max_iterations: int = 50,
        score_file: str = 'logs/score.json',
        include_human: bool = False,
        api_base: Optional[str] = None,
        manual_plans_file: Optional[str] = None,
        planning_mode: str = 'simple',
        env_info: Optional[EnvironmentInformation] = None,
    ) -> None:
        super().__init__()

        # Store config for use in initialize() (agent_id not yet available)
        self._config = {
            'llm_model': llm_model,
            'ticks_per_iteration': ticks_per_iteration,
            'max_iterations': max_iterations,
            'score_file': score_file,
            'include_human': include_human,
            'api_base': api_base,
            'manual_plans_file': manual_plans_file,
            'planning_mode': planning_mode,
            'env_info': env_info or EnvironmentInformation(),
        }

    # ------------------------------------------------------------------
    # MATRX lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Called by MATRX once after agent_id is assigned."""
        cfg = self._config
        self._llm_model = cfg['llm_model']
        self._api_base = cfg['api_base']
        self._ticks_per_iteration = cfg['ticks_per_iteration']
        self.max_iterations = cfg['max_iterations']
        self.score_file = cfg['score_file']
        self._include_human = cfg['include_human']

        # ── Modules (same types as rescue agents, planner-specific prompts) ──
        self.planner_module = Planning(mode=cfg['planning_mode'])
        self.reasoning = ReasoningIO('EMPTY')
        self.communication = CommunicationModule(
            agent_id=self.agent_id,
            strategy='always_respond',
            llm_model=self._llm_model,
            api_base=self._api_base,
        )
        self.area_tracker = AreaExplorationTracker(
            cfg['env_info'].get_area_cells()
        )
        self.memory = BaseMemory(maxlen=200)

        # ── Iteration state machine ──────────────────────────────────────
        self._phase = NEEDS_PLANNING
        self._iteration = 0
        self._ticks_in_iteration = 0
        self._total_ticks = 0
        self.iteration_history: deque = deque(maxlen=self.max_iterations)
        self._iteration_data: Optional[IterationData] = None
        self._last_summary = ''
        self._checkpoint_interval = 1000  # Save metrics every N ticks
        self._last_checkpoint_tick = 0

        # ── Async LLM futures ────────────────────────────────────────────
        self._planning_future = None
        self._summary_future = None
        self._prefetch_future = None
        self._pending_answers: list = []  # list of (question_msg, Future)

        # ── Agent discovery ──────────────────────────────────────────────
        self._rescue_agent_ids: List[str] = []
        self._rescue_agent_caps: Dict[str, Dict] = {}

        # ── Manual plans override ────────────────────────────────────────
        self._manual_data: Optional[dict] = None
        self._manual_iteration: int = 0
        if cfg['manual_plans_file']:
            with open(cfg['manual_plans_file'], 'r') as mf:
                self._manual_data = yaml.safe_load(mf)
            logger.info('[Planner] Loaded manual plans from %s', cfg['manual_plans_file'])

        # ── World state cache (updated each tick from filter_observations) ─
        self._world_state: Dict = {}

        # Track whether simulation should stop
        self._should_stop = False

        logger.info('[%s] EnginePlanner initialized (model=%s, ticks/iter=%d)',
                     self.agent_id, self._llm_model, self._ticks_per_iteration)

    def filter_observations(self, state):
        """Full visibility — return state unmodified."""
        return state

    def decide_on_action(self, state):
        """Called every tick by MATRX. Runs the iteration state machine."""
        self._total_ticks += 1
        # Cache world state for LLM prompts
        self._world_state = state

        # Discover rescue agents on first tick
        if not self._rescue_agent_ids:
            self._discover_agents(state)

        # Process incoming messages (questions, status reports)
        self._process_incoming_messages()

        # Update area tracker with all rescue agents' positions
        self._update_area_tracker(state)

        # Run the iteration state machine
        if not self._should_stop:
            self._run_state_machine(state)

        # Periodic checkpoint for HPC resilience
        if self._total_ticks - self._last_checkpoint_tick >= self._checkpoint_interval:
            self._save_checkpoint()
            self._last_checkpoint_tick = self._total_ticks

        return None, {}  # Always Idle

    def _save_checkpoint(self) -> None:
        """Save iteration history to disk for crash resilience."""
        if not self.iteration_history:
            return
        log_dir = os.path.dirname(self.score_file)
        os.makedirs(log_dir, exist_ok=True)
        checkpoint = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'total_ticks': self._total_ticks,
            'current_iteration': self._iteration,
            'phase': self._phase,
            'iterations': [{
                'iteration': d.iteration,
                'task_assignments': d.task_assignments,
                'summary': d.summary,
                'score': d.score,
                'block_hit_rate': d.block_hit_rate,
            } for d in self.iteration_history],
        }
        path = os.path.join(log_dir, 'iteration_history.json')
        try:
            with open(path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except Exception as e:
            logger.warning('Failed to save checkpoint: %s', e)

    # ------------------------------------------------------------------
    # Agent discovery
    # ------------------------------------------------------------------

    def _discover_agents(self, state) -> None:
        """Scan state for rescue agents (IDs starting with 'rescuebot')."""
        for obj_id in state.keys():
            if isinstance(obj_id, str) and obj_id.startswith('rescuebot'):
                self._rescue_agent_ids.append(obj_id)
                # Cache capabilities if available
                caps = state.get(obj_id, {}).get('capabilities')
                if caps:
                    self._rescue_agent_caps[obj_id] = caps
        if self._rescue_agent_ids:
            logger.info('[Planner] Discovered agents: %s', self._rescue_agent_ids)

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def _process_incoming_messages(self) -> None:
        """Process messages from rescue agents (questions, status reports)."""
        # Let communication module handle general message bookkeeping
        self.communication.process_messages(self.received_messages)

        # Check for planner-protocol messages
        for msg in self.received_messages:
            content = msg.content if hasattr(msg, 'content') else msg
            if not isinstance(content, dict):
                continue

            msg_type = content.get('type', '')

            if msg_type == 'planner_question':
                self._handle_agent_question(msg)
            elif msg_type == 'task_status':
                self._handle_task_status(content)

        # Harvest completed answer futures and send responses
        still_pending = []
        for question_msg, future in self._pending_answers:
            if future.done():
                try:
                    answer_text = future.result()
                except Exception as e:
                    answer_text = f'Error processing question: {e}'
                    logger.warning('[Planner] Answer future raised: %s', e)

                q_content = question_msg.content if hasattr(question_msg, 'content') else question_msg
                from_id = question_msg.from_id if hasattr(question_msg, 'from_id') else q_content.get('from', '')

                self.send_message(Message(
                    content={'type': 'planner_answer', 'answer': answer_text},
                    from_id=self.agent_id,
                    to_id=from_id,
                ))
                logger.info('[Planner] Answered %s: %s', from_id, answer_text[:80])
            else:
                still_pending.append((question_msg, future))
        self._pending_answers = still_pending

    def _handle_agent_question(self, msg) -> None:
        """Submit an LLM call to answer an agent's question."""
        content = msg.content if hasattr(msg, 'content') else msg
        from_id = msg.from_id if hasattr(msg, 'from_id') else content.get('from', '')
        question = content.get('question', '')

        logger.info('[Planner] Question from %s: %s', from_id, question[:80])
        future = _get_executor().submit(
            self._answer_question_sync, from_id, question
        )
        self._pending_answers.append((msg, future))

    def _handle_task_status(self, content: dict) -> None:
        """Record agent task status in memory."""
        self.memory.update('task_status', {
            'agent': content.get('agent_id', '?'),
            'task': content.get('task', ''),
            'status': content.get('status', ''),
            'details': content.get('details', ''),
        })

    # ------------------------------------------------------------------
    # Area tracker (global view)
    # ------------------------------------------------------------------

    def _update_area_tracker(self, state) -> None:
        """Update area tracker using all rescue agents' positions."""
        for aid in self._rescue_agent_ids:
            agent_data = state.get(aid, {})
            loc = agent_data.get('location')
            if loc:
                caps = self._rescue_agent_caps.get(aid, {})
                vision_str = caps.get('vision', 'medium') if caps else 'medium'
                vision = {'low': 1, 'medium': 2, 'high': 3}.get(vision_str, 2)
                self.area_tracker.update(tuple(loc), vision_radius=vision)

    # ------------------------------------------------------------------
    # Iteration state machine
    # ------------------------------------------------------------------

    def _run_state_machine(self, state) -> None:
        """Advance the iteration lifecycle (non-blocking)."""

        if self._phase == NEEDS_PLANNING:
            self._on_needs_planning()

        if self._phase == PLANNING_IN_PROGRESS:
            self._on_planning_in_progress()

        if self._phase == EXECUTING:
            self._ticks_in_iteration += 1
            if self._ticks_in_iteration >= self._ticks_per_iteration:
                self._phase = NEEDS_SUMMARIZATION

        if self._phase == NEEDS_SUMMARIZATION:
            self._on_needs_summarization()

        if self._phase == SUMMARIZING:
            self._on_summarizing()

    def _on_needs_planning(self) -> None:
        """Submit task generation to background thread."""
        print(f"\n{'=' * 60}")
        print(f'[Planner] Planning iteration {self._iteration}')
        print(f"{'=' * 60}")

        self._iteration_data = IterationData(iteration=self._iteration)
        self._ticks_in_iteration = 0

        # Use prefetch if available
        if self._prefetch_future is not None and self._prefetch_future.done():
            prefetch = self._prefetch_future
            self._prefetch_future = None
            try:
                result = prefetch.result()
                if result and 'tasks' in result:
                    logger.info('[Planner] Used prefetched tasks')
                    # Wrap in a resolved future
                    import concurrent.futures
                    f = concurrent.futures.Future()
                    f.set_result(result)
                    self._planning_future = f
                    self._phase = PLANNING_IN_PROGRESS
                    return
            except Exception as e:
                logger.error('[Planner] Prefetch future raised: %s', e)

        if self._prefetch_future is not None:
            # Prefetch exists but not done yet — use it
            self._planning_future = self._prefetch_future
            self._prefetch_future = None
            self._phase = PLANNING_IN_PROGRESS
            return

        # No prefetch: submit fresh task generation
        self._planning_future = _get_executor().submit(
            self._generate_tasks_sync
        )
        self._phase = PLANNING_IN_PROGRESS

    def _on_planning_in_progress(self) -> None:
        """Poll planning future, distribute tasks via messages when done."""
        if self._planning_future is None or not self._planning_future.done():
            return

        try:
            task_assignments = self._planning_future.result()
        except Exception as e:
            print(f'[Planner] Planning error: {e}')
            task_assignments = {
                'tasks': {aid: 'explore area 1 and find victims'
                          for aid in self._rescue_agent_ids}
            }
        self._planning_future = None
        self._iteration_data.task_assignments = task_assignments

        # Send task_assignment messages to each rescue agent
        agent_tasks = task_assignments.get('tasks', {})
        agent_plans = task_assignments.get('plans', {})
        all_assignments = agent_tasks  # shared context for agents

        for aid in self._rescue_agent_ids:
            task = agent_tasks.get(aid, 'Explore nearest area')
            msg_content = {
                'type': 'task_assignment',
                'iteration': self._iteration,
                'task': task,
                'task_assignments': all_assignments,
            }
            # Include manual plan if provided
            plan = agent_plans.get(aid)
            if plan:
                msg_content['plan'] = plan

            self.send_message(Message(
                content=msg_content,
                from_id=self.agent_id,
                to_id=aid,
            ))
            print(f'[Planner] Assigned to {aid}: {task}')

        # Record initial task results
        for aid in self._rescue_agent_ids:
            self._iteration_data.task_results.append({
                'agent_id': aid,
                'task': agent_tasks.get(aid, 'Explore nearest area'),
                'result': {'status': 'executing'},
            })

        self.memory.update('iteration_tasks', {
            'iteration': self._iteration,
            'assignments': agent_tasks,
        })

        self._phase = EXECUTING

    def _on_needs_summarization(self) -> None:
        """Submit summarization + prefetch to background threads."""
        # Update task results
        for result in self._iteration_data.task_results:
            result['result'] = {
                'status': 'ticks_exhausted',
                'ticks_executed': self._ticks_in_iteration,
            }

        # Submit summarization
        world_state_summary = to_toon(self._world_state) if self._world_state else ''
        self._summary_future = _get_executor().submit(
            self._summarize_sync, self._iteration_data, world_state_summary
        )

        # Pre-fetch NEXT iteration's tasks in background
        self._prefetch_future = _get_executor().submit(
            self._generate_tasks_sync
        )

        # Update _last_summary when done (callback runs on pool thread)
        def _on_summary_done(fut):
            try:
                self._last_summary = fut.result()
            except Exception:
                pass
        self._summary_future.add_done_callback(_on_summary_done)

        self._phase = SUMMARIZING

    def _on_summarizing(self) -> None:
        """Poll summary future, check termination, advance iteration."""
        if self._summary_future is None or not self._summary_future.done():
            return

        try:
            summary = self._summary_future.result()
        except Exception as e:
            print(f'[Planner] Summary error: {e}')
            summary = 'Summary unavailable'
        self._summary_future = None

        self._iteration_data.summary = summary
        print(f'[Planner] Iteration {self._iteration} summary: {summary[:200]}')

        # Termination check
        should_continue = self._check_termination(self._iteration_data)
        self._iteration_data.continue_simulation = should_continue
        self.iteration_history.append(self._iteration_data)

        if not should_continue:
            print(f'[Planner] Stopping at iteration {self._iteration}')
            self._should_stop = True
            return

        self._iteration += 1
        self._phase = NEEDS_PLANNING

    # ------------------------------------------------------------------
    # LLM call implementations (synchronous, run in background threads)
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_location_from_id(task_value):
        """Remove any '@...' location suffix from target_id."""
        if not isinstance(task_value, dict):
            return task_value
        raw_id = task_value.get('target_id')
        if isinstance(raw_id, str) and '@' in raw_id:
            task_value['target_id'] = raw_id.split('@')[0].strip()
        return task_value

    def _generate_tasks_sync(self) -> Dict[str, Any]:
        """Generate task assignments via LLM. Runs in background thread."""
        agent_ids = self._rescue_agent_ids

        # Use manual plans if configured
        if self._manual_data is not None:
            return self._build_manual_tasks(agent_ids)

        # Build formatted agent listing with capabilities
        agent_lines = []
        for aid in agent_ids:
            caps = self._rescue_agent_caps.get(aid)
            if caps:
                cap_str = ', '.join(f'{k}: {v}' for k, v in caps.items())
                agent_lines.append(f'  - {aid} ({cap_str})')
            else:
                agent_lines.append(f'  - {aid}')
        agent_ids_formatted = '\n'.join(agent_lines)
        agent_tasks_schema = ', '.join(f'"{aid}": "task for {aid}"' for aid in agent_ids)

        json_schema = f'{{"tasks": {{{agent_tasks_schema}}}"'
        if self._include_human:
            json_schema += ', "human_task": "suggested task for human"'

        num_agents = len(agent_ids)
        # generate_tasks_system has no format placeholders — use as-is
        system_prompt = PROMPTS['generate_tasks_system']

        human_line = '- Human' if self._include_human else ''

        user_prompt = PROMPTS['generate_tasks_user'].format(
            world_state_summary=to_toon(self._world_state),
            json_schema=json_schema,
            agent_ids_formatted=agent_ids_formatted,
            human_line=human_line,
            previous_summary=(
                self._last_summary if self._last_summary
                else 'First iteration.'
            ),
        )

        response = call_llm_sync(
            llm_model=self._llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_base=self._api_base,
            few_shot_messages=load_few_shot('generate_tasks'),
        )

        result = parse_json_response(response)
        print('LLM task generation response:', result)

        if result:
            if 'tasks' in result:
                result['tasks'] = {
                    aid: self._strip_location_from_id(task)
                    for aid, task in result['tasks'].items()
                }
                # Prevent an agent being assigned a cooperative task that names
                # only itself — replace the self-reference with another agent's ID.
                other_ids = self._rescue_agent_ids
                for aid, task in result['tasks'].items():
                    if aid in task:
                        partners = [o for o in other_ids if o != aid]
                        if partners:
                            result['tasks'][aid] = task.replace(aid, partners[0])
                return result

        # Fallback if LLM fails
        logger.warning('LLM task generation failed, using fallback')
        fallback = {
            'tasks': {aid: 'explore area 1 and find victims' for aid in agent_ids},
        }
        if self._include_human:
            fallback['human_task'] = 'Please explore areas and rescue victims'
        return fallback

    def _summarize_sync(self, iteration_data: IterationData,
                        world_state_summary: str) -> str:
        """Summarize iteration results via LLM. Runs in background thread."""
        score_info = self._read_score_info()

        task_results_text = ''
        for result in iteration_data.task_results:
            task_results_text += (
                f"- {result.get('agent_id', '?')}: task='{result.get('task', '?')}' "
                f"status={result.get('result', {}).get('status', '?')}\n"
            )
        if not task_results_text:
            task_results_text = 'No task results recorded this iteration.'

        system_prompt = PROMPTS['summarize_system'].strip()
        user_prompt = PROMPTS['summarize_user'].format(
            task_assignments_json=to_toon(iteration_data.task_assignments),
            task_results_text=task_results_text,
            world_state_summary=world_state_summary,
            score_info=score_info,
        )

        response = call_llm_sync(
            llm_model=self._llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_base=self._api_base,
            few_shot_messages=load_few_shot('summarize'),
        )

        if response:
            return response

        # Fallback
        parts = [
            f"{r.get('agent_id', '?')}: "
            f"{r.get('task', '?')} -> "
            f"{r.get('result', {}).get('status', '?')}"
            for r in iteration_data.task_results
        ]
        return '; '.join(parts) if parts else 'No results'

    def _answer_question_sync(self, agent_id: str, question: str) -> str:
        """Answer an agent question via LLM. Runs in background thread."""
        current_tasks = {}
        if self.iteration_history:
            current_tasks = self.iteration_history[-1].task_assignments
        elif self._rescue_agent_ids:
            current_tasks = {aid: '(pending)' for aid in self._rescue_agent_ids}

        # Include agent capabilities if known
        caps = self._rescue_agent_caps.get(agent_id)
        agent_capabilities = (
            ', '.join(f'{k}: {v}' for k, v in caps.items()) if caps else 'unknown'
        )

        system_prompt = PROMPTS['answer_question_system']
        user_prompt = PROMPTS['answer_question_user'].format(
            agent_id=agent_id,
            agent_capabilities=agent_capabilities,
            question=question,
            world_state=to_toon(self._world_state),
            current_tasks=json.dumps(current_tasks, default=str),
        )

        response = call_llm_sync(
            llm_model=self._llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_base=self._api_base,
            few_shot_messages=load_few_shot('planner_answer'),
            max_token_num=2000,
            temperature=0.2,
        )
        return response or 'Unable to answer at this time.'

    # ------------------------------------------------------------------
    # Manual plans override
    # ------------------------------------------------------------------

    def _build_manual_tasks(self, agent_ids: list) -> Dict[str, Any]:
        """Build task assignments from the manual plans YAML file."""
        tasks: Dict[str, str] = {}
        plans: Dict[str, str] = {}

        iterations = self._manual_data.get('iterations', [])
        if iterations:
            idx = min(self._manual_iteration, len(iterations) - 1)
            entry = iterations[idx]
            self._manual_iteration += 1
            for aid in agent_ids:
                if aid in entry:
                    agent_entry = entry[aid]
                    tasks[aid] = agent_entry.get('task', 'explore nearest area')
                    if 'plan' in agent_entry:
                        plans[aid] = agent_entry['plan']

        for aid in agent_ids:
            if aid not in plans:
                agent_plan = self._manual_data.get('agent_plans', {}).get(aid)
                if agent_plan:
                    plans[aid] = agent_plan
            if aid not in tasks:
                tasks[aid] = 'explore nearest area'

        result: Dict[str, Any] = {'tasks': tasks}
        if plans:
            result['plans'] = plans
        logger.info('[Planner] Manual tasks: %s', tasks)
        return result

    # ------------------------------------------------------------------
    # Termination & scoring
    # ------------------------------------------------------------------

    def _read_score_info(self) -> str:
        """Read current score data from score.json."""
        try:
            with open(self.score_file, 'r') as f:
                score_data = json.load(f)
                return (
                    f"Victims rescued: {score_data.get('victims_rescued', 0)}/"
                    f"{score_data.get('total_victims', 8)}, "
                    f"Score: {score_data.get('score', 0)}, "
                    f"Block hit rate: {score_data.get('block_hit_rate', 0.0):.2f}"
                )
        except (FileNotFoundError, json.JSONDecodeError):
            return 'Score data unavailable'

    def _check_termination(self, iteration_data: IterationData) -> bool:
        """Check if simulation should continue (rule-based, no LLM)."""
        try:
            with open(self.score_file, 'r') as f:
                score_data = json.load(f)
                block_hit_rate = score_data.get('block_hit_rate', 0.0)
                iteration_data.block_hit_rate = block_hit_rate
                iteration_data.score = score_data.get('score', 0)
                logger.info('  Score: %s, Block hit rate: %.2f',
                           iteration_data.score, block_hit_rate)
                if block_hit_rate >= 1.0:
                    logger.info('  All victims rescued (block_hit_rate == 1.0)')
                    return False
        except FileNotFoundError:
            logger.warning('  Score file %s not found, continuing...', self.score_file)
        except json.JSONDecodeError:
            logger.warning('  Score file %s invalid JSON, continuing...', self.score_file)

        if iteration_data.iteration >= self.max_iterations - 1:
            logger.info('  Max iterations (%d) reached', self.max_iterations)
            return False

        return True
