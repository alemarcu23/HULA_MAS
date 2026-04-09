"""
Deterministic per-agent area-exploration tracker for the SAR grid world.

Precomputes inside cells for each room once (in WorldBuilder), then each
agent maintains its own belief of which cells it has personally observed.

Not shared across agents — each agent has its own tracker instance.
Inter-agent knowledge transfer happens via messaging, not via this module.

Usage:
    # WorldBuilder builds EnvironmentInformation which precomputes area cells.
    # Each agent receives env_info and creates its own tracker:
    #   self.area_tracker = AreaExplorationTracker(env_info.get_area_cells())

    # Inside agent, every tick:
    self.area_tracker.update(agent_location=(9, 3), vision_radius=1)

    # Query:
    self.area_tracker.get_area_summary("area 2")
    self.area_tracker.get_all_summaries()
    self.area_tracker.is_area_complete("area 2")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Geometry ────────────────────────────────────────────────────────────────

def compute_inside_cells(
    top_left: Tuple[int, int],
    width: int,
    height: int,
) -> Set[Tuple[int, int]]:
    """Return the set of interior cells for a room whose boundary is walls.

    Assumption (matching MATRX add_room):
        top_left is the outer top-left corner (a wall cell).
        width/height include the boundary walls on all four sides.
        Inside cells are everything strictly inside the wall ring.

    Example:
        top_left=(1,1), width=5, height=4
        walls: x in {1,5}, y in {1,4}
        inside: x in [2..4], y in [2..3]
        -> {(2,2),(3,2),(4,2),(2,3),(3,3),(4,3)}
    """
    x0, y0 = top_left
    cells: Set[Tuple[int, int]] = set()
    for x in range(x0 + 1, x0 + width - 1):
        for y in range(y0 + 1, y0 + height - 1):
            cells.add((x, y))
    return cells


def precompute_all_areas(
    areas_config: List[Dict[str, Any]],
) -> Dict[str, Set[Tuple[int, int]]]:
    """Precompute inside cells for every area in the config.

    Args:
        areas_config: List of area dicts with keys ``id``, ``pos``, ``w``, ``h``.

    Returns:
        Mapping from area name (e.g. ``"area 1"``) to its set of inside cells.
        Skips entries with ``id == "world_bounds"``.
    """
    result: Dict[str, Set[Tuple[int, int]]] = {}
    for cfg in areas_config:
        area_id = cfg["id"]
        if area_id == "world_bounds":
            continue
        name = f"area {area_id}"
        result[name] = compute_inside_cells(cfg["pos"], cfg["w"], cfg["h"])
    return result


# ── Per-area state ──────────────────────────────────────────────────────────

@dataclass
class _AreaState:
    """Mutable exploration state for a single area (internal)."""
    name: str
    inside_cells: Set[Tuple[int, int]]
    explored_cells: Set[Tuple[int, int]] = field(default_factory=set)

    @property
    def total(self) -> int:
        return len(self.inside_cells)

    @property
    def explored_count(self) -> int:
        return len(self.explored_cells)

    @property
    def unexplored_count(self) -> int:
        return self.total - self.explored_count

    @property
    def coverage(self) -> float:
        return self.explored_count / self.total if self.total else 1.0

    @property
    def remaining_cells(self) -> List[List[int]]:
        return sorted([list(c) for c in self.inside_cells - self.explored_cells])

    @property
    def is_complete(self) -> bool:
        return self.unexplored_count == 0

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "coverage": round(self.coverage, 2),
            "status": "complete" if self.is_complete else
                      "in_progress" if self.explored_count > 0 else
                      "not_started",
        }


# ── Per-agent tracker ──────────────────────────────────────────────────────

class AreaExplorationTracker:
    """Per-agent exploration tracker.  Each agent owns one instance.

    Args:
        area_cells: Precomputed mapping from area name to inside-cell sets,
                    as returned by ``precompute_all_areas()``.
    """

    def __init__(self, area_cells: Dict[str, Set[Tuple[int, int]]]) -> None:
        self._areas: Dict[str, _AreaState] = {
            name: _AreaState(name=name, inside_cells=frozenset(cells))
            for name, cells in area_cells.items()
        }

        # ── Initialisation summary ─────────────────────────────────────────
        # Printed once so you can verify the correct areas / cell counts were
        # loaded from the active world preset.
        print(f'[AreaTracker] Initialized {len(self._areas)} areas:')
        for aname, astate in sorted(self._areas.items()):
            if astate.inside_cells:
                xs = [x for x, _y in astate.inside_cells]
                ys = [y for _x, y in astate.inside_cells]
                print(
                    f'  {aname}: {astate.total} inside cells  '
                    f'x=[{min(xs)},{max(xs)}] y=[{min(ys)},{max(ys)}]'
                )
            else:
                print(
                    f'  {aname}: 0 inside cells '
                    f'(EMPTY — check room dimensions in the preset!)'
                )

    # ── Per-tick update ─────────────────────────────────────────────────────

    def update(
        self,
        agent_location: Tuple[int, int],
        vision_radius: int = 1,
    ) -> None:
        """Mark cells within the agent's Chebyshev vision radius as explored."""
        ax, ay = agent_location
        observed: Set[Tuple[int, int]] = {
            (ax + dx, ay + dy)
            for dx in range(-vision_radius, vision_radius + 1)
            for dy in range(-vision_radius, vision_radius + 1)
        }
        for area in self._areas.values():
            newly_seen = observed & area.inside_cells
            if newly_seen:
                was_started = area.explored_count > 0
                was_complete = area.is_complete
                area.explored_cells |= newly_seen
                # Print state-transition events so you can trace exploration progress.
                if not was_started and area.explored_count > 0:
                    print(
                        f'[AreaTracker] Started exploring {area.name} '
                        f'({area.explored_count}/{area.total} cells seen so far)'
                    )
                elif not was_complete and area.is_complete:
                    print(
                        f'[AreaTracker] Completed {area.name} '
                        f'(all {area.total} cells seen — 100% coverage)'
                    )

    # ── Queries ─────────────────────────────────────────────────────────────

    def get_area_summary(self, area_name: str) -> Optional[Dict[str, Any]]:
        """Return a compact summary dict for one area, or None if unknown."""
        area = self._areas.get(area_name)
        return area.summary() if area else None

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Return summaries for every registered area."""
        return [a.summary() for a in self._areas.values()]

    def is_area_complete(self, area_name: str) -> bool:
        """True when all inside cells of the area have been observed."""
        area = self._areas.get(area_name)
        return area.is_complete if area else False
