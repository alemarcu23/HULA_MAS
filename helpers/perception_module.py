import time
from typing import Any, Dict, List, Optional, Tuple
from helpers.perception_helpers import _serialize_agent, _serialize_nearby, _classify_type
from matrx.agents.agent_utils.state import State


class Perception:
    """Converts a filtered MATRX WorldState dict to an LLM-friendly dict
    that can be passed to to_toon() for token-efficient prompting."""

    def percept_state(
        self,
        state: Dict[str, Any],
        agent_id: str,
        teammates: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Serialize filtered state to a structured dict for TOON encoding.

        Args:
            state:     Filtered MATRX state dict (from filter_observations).
            agent_id:  The agent's own ID (used to extract self-info).
            teammates: Set of (agent_id, location) tuples for all known teammates.

        Returns:
            Dict ready to be passed to to_toon().
        """
        # Extract plain teammate IDs for object classification.
        teammate_ids: set = {t[0] for t in teammates} if teammates else set()

        victims, obstacles, walls = _serialize_nearby(state, agent_id, teammate_ids)
        result = {
            "agent": _serialize_agent(state, agent_id),
            "victims": victims,
            "obstacles": obstacles,
            "walls": walls,
            "teammates": [
                {"id": t[0], "x": int(t[1][0]), "y": int(t[1][1])}
                for t in teammates
            ] if teammates else [],
        }
        return result

    def init_global_state(self) -> None:
        """Initialise the persistent world-knowledge store.

        Must be called once before ``process_observations`` is used
        (e.g. from ``SearchRescueAgent.initialize()``).
        """
        self.WORLD_STATE_GLOBAL: Dict[str, Any] = {
            'victims':   [],   # [{'id', 'severity', 'location'}]
            'obstacles': [],   # [{'id', 'type', 'location'}]
        }

    def update_world_belief(self, state) -> Dict[str, Any]:
        """Update and return the persistent global world state.

        Call every tick with the filtered state so that objects seen
        at any point in the episode are remembered even after the agent
        moves away.
        """
        if not hasattr(self, 'WORLD_STATE_GLOBAL'):
            self.init_global_state()
        if state is not None:
            self.add_new_obs(state)
        return self.WORLD_STATE_GLOBAL

    def add_new_obs(self, state) -> None:
        """Merge *state* into ``WORLD_STATE_GLOBAL``."""
        if state is None:
            return

        partner_name = getattr(self, '_partner_name', None)
        include_human = getattr(self, '_include_human', False)

        # Populate teammate IDs so rescuebot/human entries are correctly
        # excluded from victim and obstacle lists in the global world belief.
        teammate_ids: set = {t[0] for t in getattr(self, 'teammates', set())}

        # --- Nearby objects ---
        skip_ids = {self.agent_id, 'World'} | teammate_ids
        if include_human and partner_name:
            skip_ids.add(partner_name)

        for obj_id, obj_data in state.items():
            if obj_id in skip_ids:
                continue
            if not isinstance(obj_data, dict):
                continue
            loc = obj_data.get('location')
            if loc is None:
                continue

            typ = _classify_type(obj_id, obj_data, teammate_ids)
            if typ is None:
                continue

            pos = [int(c) for c in loc]

            # --- Victims ---
            if typ == 'victim':
                img = str(obj_data.get('img_name', '')).lower()
                if 'critical' in img:
                    severity = 'critical'
                elif 'mild' in img:
                    severity = 'mild'
                else:
                    severity = 'healthy'
                existing = next((v for v in self.WORLD_STATE_GLOBAL['victims'] if v['id'] == obj_id), None)
                if existing is None:
                    self.WORLD_STATE_GLOBAL['victims'].append(
                        {'id': obj_id, 'severity': severity, 'location': pos}
                    )
                    print(
                        f'[Perception:{getattr(self, "agent_id", "?")}] '
                        f'NEW victim discovered: {obj_id}  '
                        f'severity={severity}  at={pos}'
                    )
                else:
                    existing['location'] = pos

            # --- Obstacles ---
            elif typ in ('rock', 'stone', 'tree'):
                existing = next((o for o in self.WORLD_STATE_GLOBAL['obstacles'] if o['id'] == obj_id), None)
                if existing is None:
                    self.WORLD_STATE_GLOBAL['obstacles'].append(
                        {'id': obj_id, 'type': typ, 'location': pos}
                    )
                    print(
                        f'[Perception:{getattr(self, "agent_id", "?")}] '
                        f'NEW obstacle discovered: {obj_id}  '
                        f'type={typ}  at={pos}'
                    )
                else:
                    existing['location'] = pos