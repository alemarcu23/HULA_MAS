"""
SimulationMetrics — post-simulation aggregator.

Pulls data from AgentMetricsTracker instances, agent internals (memory,
communication, area tracking), the EnginePlanner, and the score file to
produce a single comprehensive JSON report.
"""

import json
import os
import re
import time
from itertools import combinations
from typing import Any, Dict, List, Optional

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def _strip_thinking(obj):
    """Recursively strip <think>…</think> blocks from all strings in a nested structure."""
    if isinstance(obj, str):
        return _THINK_RE.sub('', obj).strip()
    if isinstance(obj, dict):
        return {k: _strip_thinking(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_thinking(v) for v in obj]
    return obj

from metrics.agent_metrics import AgentMetricsTracker


def _gini_coefficient(values: List[int]) -> float:
    """Compute the Gini coefficient for a list of non-negative integers."""
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumsum = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumsum += v
        weighted_sum += (i + 1) * v
    mean = cumsum / n
    if mean == 0:
        return 0.0
    return (2.0 * weighted_sum) / (n * cumsum) - (n + 1) / n


class SimulationMetrics:

    def __init__(self) -> None:
        self._agents: List[Any] = []

    def register(self, agent: Any) -> None:
        self._agents.append(agent)

    def aggregate(
        self,
        agents: Optional[List[Any]] = None,
        planner: Any = None,
        score_file: Optional[str] = None,
        start_time: Optional[float] = None,
        config: Optional[Dict] = None,
        iteration_history: Optional[List] = None,
    ) -> Dict[str, Any]:
        agent_list = agents if agents is not None else self._agents
        wall_clock = time.time() - start_time if start_time else 0.0

        # Read score
        score_data = {}
        if score_file and os.path.exists(score_file):
            with open(score_file) as f:
                score_data = json.load(f)

        result: Dict[str, Any] = {}

        # ── Experiment metadata ──────────────────────────────────────────
        result['experiment_metadata'] = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'wall_clock_seconds': round(wall_clock, 2),
            'config': config or {},
            'num_agents': len(agent_list),
        }

        # ── Task performance ─────────────────────────────────────────────
        result['task_performance'] = {
            'victims_rescued': score_data.get('victims_rescued', 0),
            'total_victims': score_data.get('total_victims', 0),
            'score': score_data.get('score', 0),
            'block_hit_rate': score_data.get('block_hit_rate', 0.0),
        }

        # Count total victims found across all agents
        all_victims_found = set()
        for agent in agent_list:
            tracker = self._get_tracker(agent)
            if tracker:
                for v in tracker.victims_found:
                    all_victims_found.add(v['victim_id'])
        result['task_performance']['victims_found'] = len(all_victims_found)

        completion_ticks = max(
            (getattr(a, '_tick_count', 0) for a in agent_list), default=0
        )
        result['task_performance']['completion_ticks'] = completion_ticks

        # ── Spatial coordination ─────────────────────────────────────────
        per_agent_cells: Dict[str, set] = {}
        per_agent_areas: Dict[str, List] = {}
        for agent in agent_list:
            aid = self._get_agent_id(agent)
            tracker = self._get_tracker(agent)
            if tracker:
                per_agent_cells[aid] = tracker.cells_visited
            if hasattr(agent, 'area_tracker'):
                per_agent_areas[aid] = agent.area_tracker.get_all_summaries()

        all_cells = [c for c in per_agent_cells.values()]
        union_cells = set().union(*all_cells) if all_cells else set()
        total_visits = sum(len(c) for c in all_cells)
        # Redundant visits = total individual visits minus unique cells visited.
        # This correctly captures any pairwise or higher-order overlap, unlike
        # N-way set intersection which is zero whenever any agent didn't visit a cell.
        redundant_visits = total_visits - len(union_cells)

        pairwise_overlap = {
            f'{a}___{b}': len(per_agent_cells[a] & per_agent_cells[b])
            for a, b in combinations(sorted(per_agent_cells), 2)
        }

        result['spatial_coordination'] = {
            'areas_covered_per_agent': {
                aid: summaries for aid, summaries in per_agent_areas.items()
            },
            'total_unique_cells': len(union_cells),
            'redundant_visits': redundant_visits,
            'overlap_ratio': round(redundant_visits / total_visits, 3) if total_visits else 0.0,
            'pairwise_overlap': pairwise_overlap,
            'per_agent_unique_cells': {
                aid: len(cells) for aid, cells in per_agent_cells.items()
            },
        }

        # Count obstacles removed across all agents
        _REMOVE_ACTIONS = {'RemoveObject', 'RemoveObjectTogether'}
        obstacles_removed = 0
        for agent in agent_list:
            tracker = self._get_tracker(agent)
            if tracker:
                obstacles_removed += sum(
                    1 for a in tracker.action_log
                    if a.get('action_name') in _REMOVE_ACTIONS
                )
        result['task_performance']['obstacles_removed'] = obstacles_removed
        result['task_performance']['cells_explored'] = len(union_cells)

        # ── Communication ────────────────────────────────────────────────
        total_messages = 0
        messages_per_agent: Dict[str, Dict] = {}
        messages_by_type: Dict[str, int] = {}

        for agent in agent_list:
            aid = self._get_agent_id(agent)
            tracker = self._get_tracker(agent)
            if not tracker:
                continue
            sent = len(tracker.messages_sent)
            received = len(tracker.messages_received)
            total_messages += sent
            messages_per_agent[aid] = {'sent': sent, 'received': received}
            for m in tracker.messages_sent:
                mtype = m.get('message_type', 'unknown')
                messages_by_type[mtype] = messages_by_type.get(mtype, 0) + 1

        result['communication'] = {
            'total_messages': total_messages,
            'messages_per_agent': messages_per_agent,
            'messages_by_type': messages_by_type,
        }

        # ── Help seeking ─────────────────────────────────────────────────
        total_help = 0
        help_per_agent: Dict[str, Dict] = {}
        total_responses = 0

        for agent in agent_list:
            aid = self._get_agent_id(agent)
            tracker = self._get_tracker(agent)
            if not tracker:
                continue
            total_help += tracker.help_requests_sent
            total_responses += tracker.help_responses_sent
            help_per_agent[aid] = {
                'sent': tracker.help_requests_sent,
                'received': tracker.help_requests_received,
                'responses_sent': tracker.help_responses_sent,
            }

        _ACCEPT_KW = {'accept', 'will help', 'on my way', 'heading', 'coming', 'assist', 'yes'}
        _REFUSE_KW = {'cannot', "can't", 'unable', 'busy', 'occupied', 'not able', 'no'}
        help_accepted = 0
        help_refused = 0
        for agent in agent_list:
            tracker = self._get_tracker(agent)
            if not tracker:
                continue
            for m in tracker.messages_sent:
                if m.get('message_type') == 'help':
                    text_lower = m.get('text', '').lower()
                    if any(kw in text_lower for kw in _ACCEPT_KW):
                        help_accepted += 1
                    elif any(kw in text_lower for kw in _REFUSE_KW):
                        help_refused += 1

        result['help_seeking'] = {
            'total_help_requests': total_help,
            'per_agent': help_per_agent,
            'help_response_rate': round(total_responses / total_help, 3) if total_help else 0.0,
            'help_accepted': help_accepted,
            'help_refused': help_refused,
        }

        # ── Agent efficiency ─────────────────────────────────────────────
        efficiency: Dict[str, Dict] = {}
        actions_per_agent: Dict[str, int] = {}

        for agent in agent_list:
            aid = self._get_agent_id(agent)
            tracker = self._get_tracker(agent)
            if not tracker:
                continue
            total_actions = len(tracker.action_log)
            total_ticks = getattr(agent, '_tick_count', 0) or (tracker.idle_ticks + tracker.llm_wait_ticks + total_actions)
            actions_per_agent[aid] = total_actions

            action_counts: Dict[str, int] = {}
            for a in tracker.action_log:
                aname = a.get('action_name', 'unknown')
                action_counts[aname] = action_counts.get(aname, 0) + 1

            efficiency[aid] = {
                'action_counts_by_type': action_counts,
                'total_actions': total_actions,
                'idle_ticks': tracker.idle_ticks,
                'llm_wait_ticks': tracker.llm_wait_ticks,
                'idle_ratio': round(tracker.idle_ticks / total_ticks, 3) if total_ticks else 0.0,
                'unique_cells_visited': len(tracker.cells_visited),
                'cooperative_action_count': len(tracker.cooperative_actions),
                'validation_failures': tracker.validation_failures,
                'llm_calls': tracker.llm_call_count,
                'avg_llm_latency_s': round(
                    sum(tracker.llm_latencies) / len(tracker.llm_latencies), 3
                ) if tracker.llm_latencies else 0.0,
            }

        result['agent_efficiency'] = {'per_agent': efficiency}

        # ── Task allocation balance ──────────────────────────────────────
        action_counts_list = list(actions_per_agent.values())
        result['task_allocation_balance'] = {
            'actions_per_agent': actions_per_agent,
            'gini_coefficient': round(_gini_coefficient(action_counts_list), 3),
        }

        # ── Per-victim timeline ──────────────────────────────────────────
        victim_timeline: Dict[str, Dict] = {}
        for agent in agent_list:
            aid = self._get_agent_id(agent)
            tracker = self._get_tracker(agent)
            if not tracker:
                continue
            for v in tracker.victims_found:
                vid = v['victim_id']
                if vid not in victim_timeline or v['tick'] < victim_timeline[vid]['found_tick']:
                    victim_timeline[vid] = {
                        'victim_id': vid,
                        'found_tick': v['tick'],
                        'found_by': aid,
                        'severity': v['severity'],
                        'location': v['location'],
                    }

        sorted_timeline = sorted(victim_timeline.values(), key=lambda x: x['found_tick'])
        result['additional_suggested_metrics'] = {
            'per_victim_timeline': sorted_timeline,
            'time_to_first_victim_found': sorted_timeline[0]['found_tick'] if sorted_timeline else None,
        }

        # ── Agent memory dumps ───────────────────────────────────────────
        memory_dumps: Dict[str, Dict] = {}
        shared_memory_dumped = False
        for agent in agent_list:
            aid = self._get_agent_id(agent)
            dump: Dict[str, Any] = {}

            if hasattr(agent, 'memory'):
                try:
                    dump['full_memory'] = agent.memory.retrieve_all()
                except Exception:
                    dump['full_memory'] = []

            if hasattr(agent, 'communication') and hasattr(agent.communication, 'all_messages_raw'):
                dump['all_messages_sent_and_received'] = agent.communication.all_messages_raw
            else:
                dump['all_messages_sent_and_received'] = []

            if hasattr(agent, 'area_tracker'):
                dump['area_exploration_final'] = agent.area_tracker.get_all_summaries()

            if hasattr(agent, 'WORLD_STATE_GLOBAL'):
                dump['world_state_global'] = agent.WORLD_STATE_GLOBAL

            if not shared_memory_dumped and hasattr(agent, 'shared_memory') and agent.shared_memory:
                dump['shared_memory'] = agent.shared_memory.retrieve_all()
                shared_memory_dumped = True

            memory_dumps[aid] = dump

        result['agent_memory_dumps'] = memory_dumps

        # ── Iteration history ────────────────────────────────────────────
        if iteration_history:
            result['iteration_history'] = [
                {
                    'iteration': d.iteration,
                    'task_assignments': d.task_assignments,
                    'summary': d.summary,
                    'score': d.score,
                    'block_hit_rate': getattr(d, 'block_hit_rate', 0.0),
                } if hasattr(d, 'iteration') else d
                for d in iteration_history
            ]
        else:
            result['iteration_history'] = []

        return result

    def save(self, path: str, results: Optional[Dict] = None) -> None:
        if results is None:
            results = {}
        results = _strip_thinking(results)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_tracker(agent: Any) -> Optional[AgentMetricsTracker]:
        return getattr(agent, 'metrics', None)

    @staticmethod
    def _get_agent_id(agent: Any) -> str:
        return getattr(agent, 'agent_id', str(id(agent)))
