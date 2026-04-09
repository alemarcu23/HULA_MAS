"""
EnvironmentInformation — precomputed, read-only environment data passed to agents.

Created once in WorldBuilder, shared by reference with all agents.
Agents treat this as immutable ground truth about the world geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from agents1.modules.area_tracker import precompute_all_areas


# ── Raw area definitions (canonical source of truth) ────────────────────────

AREAS_RAW: List[Dict[str, Any]] = [
    # World Bounds
    {"id": "world_bounds", "pos": (0, 0), "w": 25, "h": 24, "door": None, "mat": None},

    # Row 1
    {"id": 1, "pos": (1, 1), "w": 5, "h": 4, "door": (3, 4), "mat": (3, 5), "enter": "North"},
    {"id": 2, "pos": (7, 1), "w": 5, "h": 4, "door": (9, 4), "mat": (9, 5), "enter": "North"},
    {"id": 3, "pos": (13, 1), "w": 5, "h": 4, "door": (15, 4), "mat": (15, 5), "enter": "North"},
    {"id": 4, "pos": (19, 1), "w": 5, "h": 4, "door": (21, 4), "mat": (21, 5), "enter": "North"},

    # Row 2
    {"id": 5, "pos": (1, 7), "w": 5, "h": 4, "door": (3, 7), "mat": (3, 6), "enter": "South"},
    {"id": 6, "pos": (7, 7), "w": 5, "h": 4, "door": (9, 7), "mat": (9, 6), "enter": "South"},
    {"id": 7, "pos": (13, 7), "w": 5, "h": 4, "door": (15, 7), "mat": (15, 6), "enter": "South"},

    # Row 3
    {"id": 8, "pos": (1, 13), "w": 5, "h": 4, "door": (3, 16), "mat": (3, 17), "enter": "North"},
    {"id": 9, "pos": (7, 13), "w": 5, "h": 4, "door": (9, 16), "mat": (9, 17), "enter": "North"},
    {"id": 10, "pos": (13, 13), "w": 5, "h": 4, "door": (15, 16), "mat": (15, 17), "enter": "North"},

    # Row 4
    {"id": 11, "pos": (1, 19), "w": 5, "h": 4, "door": (3, 19), "mat": (3, 18), "enter": "South"},
    {"id": 12, "pos": (7, 19), "w": 5, "h": 4, "door": (9, 19), "mat": (9, 18), "enter": "South"},
    {"id": 13, "pos": (13, 19), "w": 5, "h": 4, "door": (15, 19), "mat": (15, 18), "enter": "South"},
    {"id": 14, "pos": (19, 19), "w": 5, "h": 4, "door": (21, 19), "mat": (21, 18), "enter": "South"},
]


# ── Per-area metadata ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class AreaMetadata:
    """Static metadata for a single area."""
    id: int
    name: str
    pos: Tuple[int, int]
    width: int
    height: int
    inside_cells: frozenset[Tuple[int, int]]
    door: Optional[Tuple[int, int]] = None
    doormat: Optional[Tuple[int, int]] = None
    enter_direction: Optional[str] = None


# ── Environment information ─────────────────────────────────────────────────

@dataclass(frozen=True)
class EnvironmentInformation:
    """Static environment data computed at world-build time.

    Attributes:
        areas: Mapping from area integer ID to its full metadata.
        drop_zone: Top-left coordinate of the drop zone.
        drop_zone_height: Vertical extent of the drop zone.
        grid_size: (width, height) of the grid world.
        num_victims: Total number of victims in the world.
    """
    areas: Dict[int, AreaMetadata] = field(default_factory=dict)
    drop_zone: Tuple[int, int] = (23, 8)
    drop_zone_height: int = 8
    grid_size: Tuple[int, int] = (25, 24)
    num_victims: int = 8

    @staticmethod
    def build(areas_raw: Optional[List[Dict[str, Any]]] = None,
              drop_zone: Tuple[int, int] = (23, 8),
              drop_zone_height: int = 8,
              grid_size: Tuple[int, int] = (25, 24),
              num_victims: int = 8) -> EnvironmentInformation:
        """Construct from raw area config (defaults to AREAS_RAW)."""
        if areas_raw is None:
            areas_raw = AREAS_RAW

        area_cells = precompute_all_areas(areas_raw)
        areas: Dict[int, AreaMetadata] = {}

        for cfg in areas_raw:
            area_id = cfg["id"]
            if area_id == "world_bounds":
                continue
            name = f"area {area_id}"
            areas[area_id] = AreaMetadata(
                id=area_id,
                name=name,
                pos=cfg["pos"],
                width=cfg["w"],
                height=cfg["h"],
                inside_cells=frozenset(area_cells[name]),
                door=cfg.get("door"),
                doormat=cfg.get("mat"),
                enter_direction=cfg.get("enter"),
            )

        return EnvironmentInformation(
            areas=areas,
            drop_zone=drop_zone,
            drop_zone_height=drop_zone_height,
            grid_size=grid_size,
            num_victims=num_victims,
        )

    # ── Convenience lookups ─────────────────────────────────────────────────

    def get_door(self, area_id: int) -> Optional[Tuple[int, int]]:
        """Return door location for an area, or None if area doesn't exist."""
        area = self.areas.get(area_id)
        return area.door if area else None

    def get_enter_direction(self, area_id: int) -> Optional[str]:
        """Return enter direction ('North'/'South') for an area."""
        area = self.areas.get(area_id)
        return area.enter_direction if area else None

    def get_area_cells(self) -> Dict[str, frozenset[Tuple[int, int]]]:
        """Return mapping from area name to inside cells (for AreaExplorationTracker)."""
        return {area.name: area.inside_cells for area in self.areas.values()}
