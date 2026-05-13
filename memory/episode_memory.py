"""
Structured episode memory for SAR agents.

An episode groups everything that belongs to one agent decision cycle:
  - Context: task, role, messages received/sent, critic feedback, planned task
  - Action: what the LLM decided (name + args)
  - Outcome: MATRX feedback from the following tick (previous_action_result)

Episodes are closed at the start of the *next* pipeline cycle, at which point
previous_action_result (set by MATRX at the end of the previous EXECUTE tick)
is available to record success/failure.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeRecord:
    # ── Identity ────────────────────────────────────────────────────────────
    episode_id: str
    agent_id: str
    tick_open: int
    task: str
    role: str
    closed: bool = False

    # ── Context (COMM + PLANNING stages) ────────────────────────────────────
    received_messages: List[Dict[str, Any]] = field(default_factory=list)
    comm_sent: Optional[str] = None
    critic_feedback: Optional[Dict[str, Any]] = None
    planned_task: Optional[str] = None
    motivation: Optional[str] = None

    # ── Action (EXECUTE stage) ───────────────────────────────────────────────
    action_name: Optional[str] = None
    action_args: Optional[Dict[str, Any]] = None
    tick_action: Optional[int] = None
    validation_failure: Optional[str] = None

    # ── Loop signal (REASONING stage) ───────────────────────────────────────
    loop_warning: Optional[str] = None

    # ── Outcome (captured at start of NEXT cycle) ───────────────────────────
    tick_close: Optional[int] = None
    outcome_succeeded: Optional[bool] = None
    outcome_reason: Optional[str] = None

    # ── Collaboration (cooperative-carry help exchange) ──────────────────────
    collaboration: Optional[Dict[str, Any]] = None

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_id': self.episode_id,
            'agent_id': self.agent_id,
            'tick_open': self.tick_open,
            'tick_close': self.tick_close,
            'task': self.task,
            'role': self.role,
            'closed': self.closed,
            'received_messages': self.received_messages,
            'comm_sent': self.comm_sent,
            'critic_feedback': self.critic_feedback,
            'planned_task': self.planned_task,
            'motivation': self.motivation,
            'action_name': self.action_name,
            'action_args': self.action_args,
            'tick_action': self.tick_action,
            'validation_failure': self.validation_failure,
            'loop_warning': self.loop_warning,
            'outcome_succeeded': self.outcome_succeeded,
            'outcome_reason': self.outcome_reason,
            'collaboration': self.collaboration,
        }

    # ── Human-readable console output ────────────────────────────────────────

    def __str__(self) -> str:
        if not self.closed:
            status = 'OPEN'
        elif self.outcome_succeeded is True:
            status = 'OK'
        elif self.outcome_succeeded is False:
            status = 'FAIL'
        else:
            status = 'UNKNOWN'

        critic_ok = (
            self.critic_feedback is not None
            and self.critic_feedback.get('success', True)
        )
        critique_text = (self.critic_feedback or {}).get('critique', '')[:80]

        lines = [
            f'── Episode {self.episode_id} [{status}] ──────────────',
            f'  tick:     {self.tick_open} → {self.tick_close or "?"}',
            f'  agent:    {self.agent_id}  role={self.role}',
            f'  task:     {self.task}',
            f'  plan:     {self.planned_task or "—"}',
            f'  critic:   {"✓" if critic_ok else "✗"} {critique_text}',
            (
                f'  action:   {self.action_name}({json.dumps(self.action_args, default=str)})'
                if self.action_name
                else '  action:   —'
            ),
            f'  outcome:  {self.outcome_reason or "—"}',
        ]
        if self.loop_warning:
            lines.append(f'  ⚠ LOOP:   {self.loop_warning[:80]}')
        if self.validation_failure:
            lines.append(f'  ✗ VALID:  {self.validation_failure[:80]}')
        if self.comm_sent:
            lines.append(f'  msg_out:  {self.comm_sent[:80]}')
        if self.received_messages:
            lines.append(f'  msg_in:   {len(self.received_messages)} messages')
        return '\n'.join(lines)


class EpisodeMemory:
    """Per-agent episode store.

    Stores closed EpisodeRecords in a rolling deque.  One episode is open at a
    time (self._open_episode).  Episodes are closed — and the previous action
    result captured — at the beginning of each new pipeline cycle.
    """

    MAX_EPISODES: int = 20

    def __init__(self) -> None:
        self._episodes: deque = deque(maxlen=self.MAX_EPISODES)
        self._open_episode: Optional[EpisodeRecord] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open_episode(
        self,
        tick: int,
        agent_id: str,
        task: str,
        role: str,
    ) -> EpisodeRecord:
        episode_id = f'ep_{agent_id}_{tick}'
        self._open_episode = EpisodeRecord(
            episode_id=episode_id,
            agent_id=agent_id,
            tick_open=tick,
            task=task,
            role=role,
        )
        return self._open_episode

    def set_received_messages(self, messages: List[Dict[str, Any]]) -> None:
        if self._open_episode is not None:
            self._open_episode.received_messages = list(messages)

    def set_comm_sent(self, text: str) -> None:
        if self._open_episode is not None:
            self._open_episode.comm_sent = text

    def set_critic_result(self, critic_result: Dict[str, Any]) -> None:
        if self._open_episode is not None:
            self._open_episode.critic_feedback = critic_result

    def set_planned_task(self, task: str) -> None:
        if self._open_episode is not None:
            self._open_episode.planned_task = task

    def set_motivation(self, motivation: str) -> None:
        if self._open_episode is not None:
            self._open_episode.motivation = motivation

    def set_action(
        self,
        action_name: str,
        action_args: Dict[str, Any],
        tick_action: int,
    ) -> None:
        if self._open_episode is not None:
            self._open_episode.action_name = action_name
            self._open_episode.action_args = action_args
            self._open_episode.tick_action = tick_action

    def set_loop_warning(self, warning: str) -> None:
        if self._open_episode is not None:
            self._open_episode.loop_warning = warning

    def set_collaboration(self, collab: Dict[str, Any]) -> None:
        """Record a cooperative-carry collaboration on the open episode.

        Expected keys: requester, responder, victim_id, duration_ticks, outcome.
        """
        if self._open_episode is not None:
            self._open_episode.collaboration = dict(collab)

    def close_episode(
        self,
        tick: int,
        succeeded: Optional[bool],
        reason: Optional[str],
    ) -> Optional[EpisodeRecord]:
        ep = self._open_episode
        if ep is None or ep.closed:
            return ep
        ep.tick_close = tick
        ep.outcome_succeeded = succeeded
        ep.outcome_reason = reason
        ep.closed = True
        self._episodes.append(ep)
        print(str(ep))
        self._open_episode = None
        return ep

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_open_episode(self) -> Optional[EpisodeRecord]:
        return self._open_episode

    def get_closed_episodes(self, n: int = 5) -> List[EpisodeRecord]:
        episodes = list(self._episodes)
        recent = episodes[-n:] if n < len(episodes) else episodes
        return list(reversed(recent))

    # ── LLM serialization ─────────────────────────────────────────────────────

    def to_prompt_memory(self, n: int = 10) -> List[Dict[str, Any]]:
        """Compact per-episode summaries for the Planning prompt's MEMORY block.

        Newest first.  Each entry carries the planning context (task, planned_task,
        motivation, critique) plus the action taken and its MATRX outcome, so the
        planner can spot loops and successes at a glance.
        """
        episodes = self.get_closed_episodes(n)
        result = []
        for ep in episodes:
            outcome = (
                'OK' if ep.outcome_succeeded is True
                else 'FAIL' if ep.outcome_succeeded is False
                else 'UNKNOWN'
            )
            entry: Dict[str, Any] = {
                'tick': ep.tick_open,
                'task': ep.task,
                'planned_task': ep.planned_task or '',
                'motivation': ep.motivation or '',
                'action': ep.action_name or '',
                'action_args': ep.action_args or {},
                'outcome': outcome,
                'critique': (ep.critic_feedback or {}).get('critique', ''),
            }
            result.append(entry)
        return result

    # Backwards-compat alias for any caller still using the old name.
    def to_prompt_previous_tasks(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.to_prompt_memory(n=n)
