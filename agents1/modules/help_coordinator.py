"""Message-based help-request coordination.

Implements a deterministic single-winner protocol for cooperative-carry help
requests using only inter-agent messages — no SharedMemory writes.

Invariants:
  1. At most one ask_help is active across the team at any time. A new ask_help
     observed from a peer while we have our own pending overrides ours when
     its request_id is lexicographically smaller (supersede tie-break).
  2. Only one responder commits per request: the requester broadcasts
     ``help_assigned`` on the first ``"yes"`` reply it sees; other responders
     observe the assignment and abort their commitment.
  3. The requester suspends its own pipeline while waiting (caller checks
     ``is_requester_waiting``).
  4. Resolution (rescued, all_declined, timeout, superseded) is announced via
     ``help_complete`` / ``help_canceled`` broadcasts; receivers purge the
     exchange from their inbox and the requester records the collaboration.

Message types added by this module:
    help_assigned : {request_id, chosen_responder}
    help_canceled : {request_id, reason}            reason∈{all_declined,timeout,superseded}
    help_complete : {request_id, victim_id, requester, responder, duration_ticks}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from matrx.messages.message import Message

from agents1.modules.communication_module import extract_coords_from_text

HELP_TIMEOUT_TICKS = 3000   # absolute fallback only; replan-count logic fires first
OVERLAP_TIE_BREAK_TICKS = 2
IMPLICIT_DECLINE_REPLANS = 3  # replans without reply → implicit decline

MSG_ASK_HELP = 'ask_help'
MSG_HELP = 'help'
MSG_HELP_ASSIGNED = 'help_assigned'
MSG_HELP_CANCELED = 'help_canceled'
MSG_HELP_COMPLETE = 'help_complete'

_ACTION_SEND = 'send'
_ACTION_SUPPRESS = 'suppress'
_ACTION_REWRITE = 'rewrite'


@dataclass
class _MyRequest:
    request_id: str
    victim_id: str
    victim_location: Optional[Tuple[int, int]]
    tick_sent: int
    expected: int
    kind: str = 'carry'           # 'carry' | 'remove'
    target_id: str = ''           # victim_id for carry, obstacle_id for remove
    replies: Dict[str, str] = field(default_factory=dict)
    assigned_to: Optional[str] = None
    expected_peers: List[str] = field(default_factory=list)
    peer_replan_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class _KnownRequest:
    request_id: str
    requester: str
    victim_id: str
    victim_location: Optional[Tuple[int, int]]
    tick_received: int
    kind: str = 'carry'
    target_id: str = ''
    assigned_to: Optional[str] = None


@dataclass
class _MyAcceptance:
    request_id: str
    requester: str
    victim_id: str
    tick_accepted: int
    kind: str = 'carry'
    target_id: str = ''
    target_location: Optional[Tuple[int, int]] = None


@dataclass
class ResolutionEvent:
    kind: str                       # 'rescued' | 'canceled' | 'lost_assignment'
    request_id: str
    requester: str
    responder: Optional[str] = None
    victim_id: Optional[str] = None
    duration_ticks: Optional[int] = None
    reason: Optional[str] = None


class HelpCoordinator:
    """Pure local state — only inputs are received messages and outbound msgs."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._my_request: Optional[_MyRequest] = None
        self._my_acceptance: Optional[_MyAcceptance] = None
        self._known_active: Optional[_KnownRequest] = None
        self._completed_ids: Set[str] = set()      # request_ids already resolved
        self._processed_msg_count: int = 0

    # ── Public gates ──────────────────────────────────────────────────────

    def is_requester_waiting(self) -> bool:
        return self._my_request is not None

    def my_request_id(self) -> Optional[str]:
        return self._my_request.request_id if self._my_request else None

    def my_acceptance(self) -> Optional[Dict[str, Any]]:
        if not self._my_acceptance:
            return None
        return {
            'request_id': self._my_acceptance.request_id,
            'requester': self._my_acceptance.requester,
            'victim_id': self._my_acceptance.victim_id,
            'kind': self._my_acceptance.kind,
            'target_id': self._my_acceptance.target_id,
            'target_location': self._my_acceptance.target_location,
        }

    def is_responder_committed(self) -> bool:
        return self._my_acceptance is not None

    def clear_my_acceptance(self, reason: str = '') -> None:
        if self._my_acceptance is not None:
            print(f'[{self.agent_id}] Clearing _my_acceptance ({reason or "n/a"})')
            self._my_acceptance = None

    def broadcast_complete(self, tick: int) -> List[Message]:
        """Helper-side: emit MSG_HELP_COMPLETE for the currently accepted request.

        Used when this agent finished the coop action it accepted (carry or
        remove). Clears _my_acceptance and the corresponding _known_active.
        Returns the outbound messages to send.
        """
        outbound: List[Message] = []
        a = self._my_acceptance
        if a is None:
            return outbound
        duration = tick - a.tick_accepted
        outbound.append(self._build_msg(MSG_HELP_COMPLETE, {
            'request_id': a.request_id,
            'victim_id': a.victim_id,
            'requester': a.requester,
            'responder': self.agent_id,
            'duration_ticks': duration,
            'text': f'help for {a.requester} ({a.kind} {a.target_id or a.victim_id}) complete',
        }, to_id=None))
        self._completed_ids.add(a.request_id)
        if self._known_active and self._known_active.request_id == a.request_id:
            self._known_active = None
        self._my_acceptance = None
        return outbound

    def is_request_already_assigned(self, requester_id: str) -> bool:
        k = self._known_active
        return bool(k and k.requester == requester_id and k.assigned_to
                    and k.assigned_to != self.agent_id)

    def active_request_snapshot(self) -> Optional[Dict[str, Any]]:
        """Snapshot for prompts — only when the active request is NOT ours."""
        k = self._known_active
        if not k:
            return None
        return {
            'request_id': k.request_id,
            'requester': k.requester,
            'victim_id': k.victim_id,
            'victim_location': k.victim_location,
            'accepted_by': k.assigned_to,
        }

    # ── Outgoing message vetting ──────────────────────────────────────────

    def vet_outgoing(
        self,
        message_type: str,
        send_to: Any,
        text: str,
        tick: int,
        victim_id: str = '',
        victim_location: Optional[Tuple[int, int]] = None,
        expected_responders: int = 0,
        expected_peer_ids: Optional[List[str]] = None,
        kind: str = 'carry',
        target_id: str = '',
    ) -> Tuple[str, Any]:
        """Validate / rewrite a help-related outbound message.

        Returns one of:
            ('send',     {extra content fields to merge})
            ('rewrite',  {fields including coerced text})
            ('suppress', reason_str)
        For non-help message_types returns ('send', {}).
        """
        if message_type == MSG_ASK_HELP:
            return self._vet_ask_help(
                text, tick, victim_id, victim_location,
                expected_responders, expected_peer_ids,
                kind=kind, target_id=target_id,
            )
        if message_type == MSG_HELP and send_to and send_to != 'all':
            return self._vet_help_reply(send_to, text, tick)
        return _ACTION_SEND, {}

    def _vet_ask_help(self, text, tick, victim_id, victim_location, expected_responders,
                      expected_peer_ids=None, kind='carry', target_id=''):
        if self._my_request is not None:
            return _ACTION_SUPPRESS, 'already have an active ask_help'
        if self._known_active is not None and self._known_active.requester != self.agent_id:
            return _ACTION_SUPPRESS, f'team help-request slot held by {self._known_active.requester}'

        rid = f'{self.agent_id}:{tick}'
        if not victim_location:
            victim_location = extract_coords_from_text(text or '')
        peers = list(expected_peer_ids or [])
        # target_id defaults to victim_id when kind=='carry' (legacy behavior)
        resolved_target = target_id or victim_id or ''
        self._my_request = _MyRequest(
            request_id=rid,
            victim_id=victim_id or '',
            victim_location=victim_location,
            tick_sent=tick,
            expected=max(int(expected_responders or 0), len(peers)),
            expected_peers=peers,
            kind=kind or 'carry',
            target_id=resolved_target,
        )
        return _ACTION_REWRITE, {
            'request_id': rid,
            'victim_id': victim_id or '',
            'victim_location': list(victim_location) if victim_location else None,
            'kind': kind or 'carry',
            'target_id': resolved_target,
        }

    def _vet_help_reply(self, target, text, tick):
        stripped = (text or '').strip().lower()
        if stripped not in ('yes', 'no'):
            return _ACTION_SEND, {}

        k = self._known_active
        # Coerce yes→no if the request is already assigned to someone else
        if stripped == 'yes' and k and k.requester == target and k.assigned_to \
                and k.assigned_to != self.agent_id:
            return _ACTION_REWRITE, {'text': 'no', 'request_id': k.request_id}

        if stripped == 'yes' and k and k.requester == target:
            self._my_acceptance = _MyAcceptance(
                request_id=k.request_id,
                requester=k.requester,
                victim_id=k.victim_id,
                tick_accepted=tick,
                kind=k.kind,
                target_id=k.target_id or k.victim_id,
                target_location=k.victim_location,
            )
            return _ACTION_REWRITE, {'text': 'yes', 'request_id': k.request_id}

        rid = k.request_id if k and k.requester == target else ''
        return _ACTION_REWRITE, {'text': stripped, 'request_id': rid}

    # ── Inbox ingestion (the heart of the protocol) ───────────────────────

    def set_agent_id(self, agent_id: str) -> None:
        """Refresh agent_id (MATRX sets it after construction in some setups)."""
        if agent_id and agent_id != self.agent_id:
            self.agent_id = agent_id

    def ingest(
        self,
        received_messages: List[Any],
        tick: int,
        rescued_victim_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[Message], List[ResolutionEvent]]:
        """Walk new messages, mutate state, emit outbound msgs + events.

        ``rescued_victim_ids`` is the set of victim_ids known to be rescued
        (read by the agent from the existing ``rescued_victims`` list — not a
        new shared key). Used to detect when our own request was resolved by
        a successful coop-carry.
        """
        outbound: List[Message] = []
        events: List[ResolutionEvent] = []

        new_count = len(received_messages)
        new_msgs = received_messages[self._processed_msg_count:]
        self._processed_msg_count = new_count

        for raw in new_msgs:
            content = getattr(raw, 'content', raw)
            if not isinstance(content, dict):
                continue
            sender = getattr(raw, 'from_id', '') or content.get('from', '')
            mtype = content.get('message_type', '')

            if mtype == MSG_ASK_HELP and sender != self.agent_id:
                self._on_peer_ask_help(content, sender, tick, outbound)
            elif mtype == MSG_HELP and sender != self.agent_id:
                self._on_help_reply(content, sender, tick, outbound)
            elif mtype == MSG_HELP_ASSIGNED:
                self._on_help_assigned(content, sender, events)
            elif mtype == MSG_HELP_CANCELED:
                self._on_help_canceled(content, sender, events)
            elif mtype == MSG_HELP_COMPLETE:
                self._on_help_complete(content, sender, events)
            elif mtype == 'plan_update' and sender != self.agent_id:
                # Track how many times each peer replans while we have a pending request.
                if self._my_request is not None:
                    self._my_request.peer_replan_counts[sender] = (
                        self._my_request.peer_replan_counts.get(sender, 0) + 1
                    )

        # Detect own rescue via existing rescued_victims list (read-only).
        rescued_victim_ids = rescued_victim_ids or set()
        if self._my_request and self._my_request.victim_id \
                and self._my_request.victim_id in rescued_victim_ids:
            self._broadcast_complete(tick, outbound, events)
        if self._my_acceptance and self._my_acceptance.victim_id in rescued_victim_ids:
            self._my_acceptance = None

        # Replan-count-based implicit-decline check (primary timeout mechanism).
        self._check_implicit_all_declined(tick, outbound, events)

        # Absolute fallback timeout (fires only if plan_update messages stop entirely).
        if self._my_request and (tick - self._my_request.tick_sent) >= HELP_TIMEOUT_TICKS \
                and self._my_request.assigned_to is None:
            self._broadcast_cancel(tick, reason='timeout', outbound=outbound, events=events)

        # Tie-break: if we still hold our request AND a peer request is also
        # active with a smaller request_id, we supersede ours.
        if self._my_request and self._known_active \
                and self._known_active.requester != self.agent_id \
                and abs(self._known_active.tick_received - self._my_request.tick_sent) <= OVERLAP_TIE_BREAK_TICKS \
                and self._known_active.request_id < self._my_request.request_id:
            self._broadcast_cancel(tick, reason='superseded', outbound=outbound, events=events)

        return outbound, events

    # ── Message handlers ──────────────────────────────────────────────────

    def _on_peer_ask_help(self, content, sender, tick, outbound):
        rid = content.get('request_id') or f'{sender}:?'
        if rid in self._completed_ids:
            return
        loc = content.get('victim_location')
        if isinstance(loc, list) and len(loc) == 2:
            loc = (int(loc[0]), int(loc[1]))
        elif not loc:
            loc = extract_coords_from_text(content.get('text', ''))
        self._known_active = _KnownRequest(
            request_id=rid,
            requester=sender,
            victim_id=content.get('victim_id', ''),
            victim_location=loc if isinstance(loc, tuple) else None,
            tick_received=tick,
            kind=content.get('kind', 'carry') or 'carry',
            target_id=content.get('target_id') or content.get('victim_id', '') or '',
        )

    def _on_help_reply(self, content, sender, tick, outbound):
        # Only the requester acts on reply messages.
        if not self._my_request:
            return
        to = content.get('to') or ''
        # to-field may be missing; trust request_id match
        rid = content.get('request_id', '')
        if rid and rid != self._my_request.request_id:
            return
        text = (content.get('text') or '').strip().lower()
        if text not in ('yes', 'no'):
            return
        self._my_request.replies[sender] = text

        # First 'yes' wins — broadcast assignment
        if text == 'yes' and self._my_request.assigned_to is None:
            self._my_request.assigned_to = sender
            outbound.append(self._build_msg(MSG_HELP_ASSIGNED, {
                'request_id': self._my_request.request_id,
                'chosen_responder': sender,
                'text': f'help assigned to {sender}',
            }, to_id=None))
        # All-declined detection is handled by _check_implicit_all_declined (called
        # at the end of ingest) which covers both explicit NO replies and implicit
        # declines inferred from replan counts.

    def _on_help_assigned(self, content, sender, events):
        rid = content.get('request_id', '')
        chosen = content.get('chosen_responder', '')
        if not rid:
            return
        if self._known_active and self._known_active.request_id == rid:
            self._known_active.assigned_to = chosen
        # If we said yes and lost — emit event so the agent can abort autonav.
        if self._my_acceptance and self._my_acceptance.request_id == rid \
                and chosen != self.agent_id:
            events.append(ResolutionEvent(
                kind='lost_assignment',
                request_id=rid,
                requester=self._my_acceptance.requester,
                responder=chosen,
            ))
            self._my_acceptance = None
        # If we are the chosen one, keep _my_acceptance to track completion.

    def _on_help_canceled(self, content, sender, events):
        rid = content.get('request_id', '')
        reason = content.get('reason', 'unknown')
        if not rid:
            return
        self._completed_ids.add(rid)
        requester = sender or (self._known_active.requester if self._known_active else '')

        if self._known_active and self._known_active.request_id == rid:
            self._known_active = None
        if self._my_request and self._my_request.request_id == rid:
            self._my_request = None
        if self._my_acceptance and self._my_acceptance.request_id == rid:
            self._my_acceptance = None

        events.append(ResolutionEvent(
            kind='canceled', request_id=rid, requester=requester, reason=reason,
        ))

    def _on_help_complete(self, content, sender, events):
        rid = content.get('request_id', '')
        if not rid:
            return
        self._completed_ids.add(rid)
        requester = content.get('requester', '')
        responder = content.get('responder', '')
        victim_id = content.get('victim_id', '')
        duration = content.get('duration_ticks')

        if self._known_active and self._known_active.request_id == rid:
            self._known_active = None
        if self._my_request and self._my_request.request_id == rid:
            self._my_request = None
        if self._my_acceptance and self._my_acceptance.request_id == rid:
            self._my_acceptance = None

        events.append(ResolutionEvent(
            kind='rescued', request_id=rid, requester=requester,
            responder=responder, victim_id=victim_id, duration_ticks=duration,
        ))

    # ── Implicit-decline check ────────────────────────────────────────────

    def _check_implicit_all_declined(self, tick, outbound, events) -> None:
        """Cancel the request when all expected peers have replied or implicitly declined.

        Implicit decline: the peer has broadcast IMPLICIT_DECLINE_REPLANS plan_update
        messages since the ask_help was sent without ever replying 'yes'.
        Falls back to the count-based check when no expected_peers list is available.
        """
        r = self._my_request
        if not r or r.assigned_to is not None:
            return

        if r.expected_peers:
            implicit: List[str] = []
            explicit_no: List[str] = []
            for peer in r.expected_peers:
                reply = r.replies.get(peer)
                if reply == 'yes':
                    return  # at least one accepted — no cancel
                elif reply == 'no':
                    explicit_no.append(peer)
                elif r.peer_replan_counts.get(peer, 0) >= IMPLICIT_DECLINE_REPLANS:
                    implicit.append(peer)
                else:
                    return  # this peer hasn't been accounted for yet
            # Every peer is accounted for — determine cancel reason
            if implicit and not explicit_no:
                reason = 'all_ignored'
            else:
                reason = 'all_declined'
            self._broadcast_cancel(tick, reason=reason, outbound=outbound, events=events)
        else:
            # Fallback: count-based check (no peer-ID list available)
            expected = r.expected
            if (expected > 0
                    and len(r.replies) >= expected
                    and all(v == 'no' for v in r.replies.values())):
                self._broadcast_cancel(tick, reason='all_declined', outbound=outbound, events=events)

    # ── Outbound helpers ──────────────────────────────────────────────────

    def _broadcast_complete(self, tick, outbound, events):
        r = self._my_request
        if r is None:
            return
        duration = tick - r.tick_sent
        responder = r.assigned_to or ''
        outbound.append(self._build_msg(MSG_HELP_COMPLETE, {
            'request_id': r.request_id,
            'victim_id': r.victim_id,
            'requester': self.agent_id,
            'responder': responder,
            'duration_ticks': duration,
            'text': f'rescue of {r.victim_id} complete',
        }, to_id=None))
        events.append(ResolutionEvent(
            kind='rescued', request_id=r.request_id, requester=self.agent_id,
            responder=responder, victim_id=r.victim_id, duration_ticks=duration,
        ))
        self._completed_ids.add(r.request_id)
        self._my_request = None
        if self._my_acceptance and self._my_acceptance.request_id == r.request_id:
            self._my_acceptance = None

    _CANCEL_REASON_TEXT = {
        'all_declined': 'All teammates explicitly refused your help request.',
        'all_ignored': (
            f'All teammates received your request but did not respond after '
            f'replanning {IMPLICIT_DECLINE_REPLANS} times each. '
            'They have seen the request and chosen not to help.'
        ),
        'timeout': 'Help request timed out — no teammate responded in time.',
        'superseded': 'Help request was superseded by a higher-priority request.',
    }

    def _broadcast_cancel(self, tick, reason, outbound, events):
        r = self._my_request
        if r is None:
            return
        reason_text = self._CANCEL_REASON_TEXT.get(reason, f'help request canceled: {reason}')
        outbound.append(self._build_msg(MSG_HELP_CANCELED, {
            'request_id': r.request_id,
            'reason': reason,
            'text': reason_text,
        }, to_id=None))
        if events is not None:
            events.append(ResolutionEvent(
                kind='canceled', request_id=r.request_id,
                requester=self.agent_id, reason=reason,
            ))
        self._completed_ids.add(r.request_id)
        self._my_request = None

    def _build_msg(self, message_type, content_extra, to_id=None):
        content = {'message_type': message_type, 'from': self.agent_id}
        content.update(content_extra)
        return Message(content=content, from_id=self.agent_id, to_id=to_id)
