"""
SimulationMetrics — post-simulation aggregator.

Pulls data from AgentMetricsTracker instances, agent internals (memory,
communication, area tracking), the EnginePlanner, and the score file to
produce a single comprehensive JSON report.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

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
        if len(all_cells) >= 2:
            overlap = set.intersection(*all_cells)
        else:
            overlap = set()

        result['spatial_coordination'] = {
            'areas_covered_per_agent': {
                aid: summaries for aid, summaries in per_agent_areas.items()
            },
            'total_unique_cells': len(union_cells),
            'overlap_cells_count': len(overlap),
            'overlap_ratio': round(len(overlap) / len(union_cells), 3) if union_cells else 0.0,
            'per_agent_unique_cells': {
                aid: len(cells) for aid, cells in per_agent_cells.items()
            },
        }

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

        result['help_seeking'] = {
            'total_help_requests': total_help,
            'per_agent': help_per_agent,
            'help_response_rate': round(total_responses / total_help, 3) if total_help else 0.0,
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
            'location_heatmaps': {
                self._get_agent_id(a): self._get_tracker(a).location_trace
                for a in agent_list if self._get_tracker(a)
            },
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
