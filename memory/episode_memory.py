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
            'action_name': self.action_name,
            'action_args': self.action_args,
            'tick_action': self.tick_action,
            'validation_failure': self.validation_failure,
            'loop_warning': self.loop_warning,
            'outcome_succeeded': self.outcome_succeeded,
            'outcome_reason': self.outcome_reason,
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
        return episodes[-n:] if n < len(episodes) else episodes

    # ── LLM serialization ─────────────────────────────────────────────────────

    def to_prompt_previous_tasks(self, n: int = 5) -> List[Dict[str, Any]]:
        """Replacement for memory.retrieve_all()[-5:] in the planning prompt.

        Returns a list of compact dicts, one per closed episode, richest-first
        (most recent last, matching the old ordering convention).
        """
        episodes = self.get_closed_episodes(n)
        result = []
        for ep in episodes:
            entry: Dict[str, Any] = {
                'kind': 'episode',
                'tick': ep.tick_open,
                'task': ep.task,
                'planned_task': ep.planned_task,
                'action': ep.action_name,
                'outcome_succeeded': ep.outcome_succeeded,
                'outcome_reason': ep.outcome_reason,
            }
            if ep.critic_feedback:
                entry['critique'] = ep.critic_feedback.get('critique', '')
            if ep.loop_warning:
                entry['loop_warning'] = ep.loop_warning
            if ep.validation_failure:
                entry['validation_failure'] = ep.validation_failure
            result.append(entry)
        return result

    def to_prompt_memory(self, n: int = 15) -> List[Dict[str, Any]]:
        """Replacement for memory.retrieve_all()[-15:] in the reasoning prompt.

        Returns episode dicts shaped like the old flat memory entries so that
        existing to_toon() serializers see no type change.  The open (in-flight)
        episode is included last as a PENDING entry.
        """
        episodes = self.get_closed_episodes(n)
        result = []
        for ep in episodes:
            entry: Dict[str, Any] = {
                'kind': 'episode',
                'tick': ep.tick_open,
                'task': ep.task,
                'role': ep.role,
                'action': ep.action_name,
                'action_args': ep.action_args,
                'outcome': 'OK' if ep.outcome_succeeded else ('FAIL' if ep.outcome_succeeded is False else 'UNKNOWN'),
                'outcome_reason': ep.outcome_reason,
            }
            if ep.critic_feedback:
                entry['critique'] = ep.critic_feedback.get('critique', '')
                entry['critic_success'] = ep.critic_feedback.get('success')
            if ep.planned_task:
                entry['planned_task'] = ep.planned_task
            if ep.loop_warning:
                entry['loop_warning'] = ep.loop_warning
            if ep.validation_failure:
                entry['validation_failure'] = ep.validation_failure
            if ep.comm_sent:
                entry['comm_sent'] = ep.comm_sent
            if ep.received_messages:
                entry['received_messages'] = ep.received_messages
            result.append(entry)

        # Append the open episode as a PENDING entry if one exists
        open_ep = self._open_episode
        if open_ep is not None:
            pending: Dict[str, Any] = {
                'kind': 'episode',
                'tick': open_ep.tick_open,
                'task': open_ep.task,
                'role': open_ep.role,
                'outcome': 'PENDING',
            }
            if open_ep.critic_feedback:
                pending['critique'] = open_ep.critic_feedback.get('critique', '')
            if open_ep.planned_task:
                pending['planned_task'] = open_ep.planned_task
            if open_ep.comm_sent:
                pending['comm_sent'] = open_ep.comm_sent
            result.append(pending)

        return result[-n:]
