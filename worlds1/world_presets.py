"""
World presets and dynamic world generation for SAR-env.

Defines WorldPreset and related data structures, preset factories,
room packing algorithm, and auto-decoration generation.

Usage from main.py:
    from worlds1.world_presets import get_preset
    preset = get_preset('random', seed=42, num_rooms=5, num_victims=6)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Victim / obstacle pools ──────────────────────────────────────────────────

VICTIM_POOL: List[Tuple[str, str]] = [
    ('critically injured girl', '/images/critically injured girl.svg'),
    ('critically injured elderly woman', '/images/critically injured elderly woman.svg'),
    ('critically injured man', '/images/critically injured man.svg'),
    ('critically injured dog', '/images/critically injured dog.svg'),
    ('mildly injured boy', '/images/mildly injured boy.svg'),
    ('mildly injured elderly man', '/images/mildly injured elderly man.svg'),
    ('mildly injured woman', '/images/mildly injured woman.svg'),
    ('mildly injured cat', '/images/mildly injured cat.svg'),
]

OBSTACLE_POOL: List[Tuple[str, str]] = [
    ('rock', '/images/stone.svg'),
    ('stone', '/images/stone-small.svg'),
    ('tree', '/images/tree-fallen2.svg'),
]


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class VictimDef:
    name: str
    img: str
    location: Tuple[int, int]
    area: str


@dataclass
class ObstacleDef:
    name: str
    img: str
    location: Tuple[int, int]


@dataclass
class DropZoneDef:
    location: Tuple[int, int]
    height: int


@dataclass
class RoomDef:
    id: int
    pos: Tuple[int, int]          # top-left corner (x, y)
    width: int                    # including walls, 4-10
    height: int                   # including walls, 4-10
    door: Tuple[int, int]
    doormat: Tuple[int, int]
    enter_direction: str          # 'North' or 'South'
    victims: List[VictimDef] = field(default_factory=list)
    obstacles: List[ObstacleDef] = field(default_factory=list)


@dataclass
class WorldPreset:
    name: str
    grid_width: int               # max 30
    grid_height: int              # max 30
    rooms: List[RoomDef]
    drop_zone: DropZoneDef
    ghost_victims: List[Tuple[str, str]]    # (name, img) for drop zone placeholders
    decorative_overrides: Optional[Dict] = None  # None = auto-generate
    seed: Optional[int] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _interior_cells(room: RoomDef) -> List[Tuple[int, int]]:
    """Return all cells strictly inside the room walls."""
    x0, y0 = room.pos
    cells = []
    for x in range(x0 + 1, x0 + room.width - 1):
        for y in range(y0 + 1, y0 + room.height - 1):
            cells.append((x, y))
    return cells


def _compute_door_bottom(x0: int, y0: int, w: int, h: int) -> Tuple[Tuple[int, int], Tuple[int, int], str]:
    """Door on bottom wall, enter from North (agent walks south to enter)."""
    door_x = x0 + w // 2
    door_y = y0 + h - 1
    return (door_x, door_y), (door_x, door_y + 1), 'North'


def _compute_door_top(x0: int, y0: int, w: int, h: int) -> Tuple[Tuple[int, int], Tuple[int, int], str]:
    """Door on top wall, enter from South (agent walks north to enter)."""
    door_x = x0 + w // 2
    door_y = y0
    return (door_x, door_y), (door_x, door_y - 1), 'South'


def _rects_overlap(ax, ay, aw, ah, bx, by, bw, bh, margin: int = 0) -> bool:
    """Check if two axis-aligned rectangles overlap (with optional margin)."""
    return not (ax + aw + margin <= bx or bx + bw + margin <= ax or
                ay + ah + margin <= by or by + bh + margin <= ay)


def _place_victims_in_rooms(rooms: List[RoomDef], num_victims: int, rng: random.Random):
    """Distribute victims randomly across rooms at random interior cells."""
    if not rooms:
        return
    pool = list(VICTIM_POOL)
    rng.shuffle(pool)
    victims_to_place = pool[:num_victims]

    for i, (name, img) in enumerate(victims_to_place):
        room = rooms[i % len(rooms)]
        interior = _interior_cells(room)
        # Avoid placing on door cell or where another victim already is
        used = {v.location for v in room.victims}
        available = [c for c in interior if c not in used and c != room.door]
        if not available:
            available = interior  # fallback
        loc = rng.choice(available)
        room.victims.append(VictimDef(
            name=name, img=img, location=loc, area=f'area {room.id}'
        ))


def _place_obstacles_at_doors(rooms: List[RoomDef], rng: random.Random, probability: float = 0.5):
    """Place obstacles at some room doors randomly."""
    for room in rooms:
        if rng.random() < probability:
            obs_name, obs_img = rng.choice(OBSTACLE_POOL)
            room.obstacles.append(ObstacleDef(
                name=obs_name, img=obs_img, location=[room.door[0], room.door[1]-1]
            ))


def _collect_all_victims(rooms: List[RoomDef]) -> List[Tuple[str, str]]:
    """Gather (name, img) for all victims across rooms — used for ghost blocks."""
    result = []
    for room in rooms:
        for v in room.victims:
            result.append((v.name, v.img))
    return result


# ── Converter to AREAS_RAW format ────────────────────────────────────────────

def to_areas_raw(preset: WorldPreset) -> List[Dict[str, Any]]:
    """Convert a WorldPreset into the AREAS_RAW format used by EnvironmentInformation.build()."""
    result = [{"id": "world_bounds", "pos": (0, 0),
               "w": preset.grid_width, "h": preset.grid_height,
               "door": None, "mat": None}]
    for room in preset.rooms:
        result.append({
            "id": room.id,
            "pos": room.pos,
            "w": room.width,
            "h": room.height,
            "door": room.door,
            "mat": room.doormat,
            "enter": room.enter_direction,
        })
    return result


# ── Auto decoration generators ───────────────────────────────────────────────

def generate_roof_tiles(rooms: List[RoomDef]) -> List[Tuple[int, int]]:
    """Generate roof tile positions from room wall cells."""
    tiles = []
    for room in rooms:
        x0, y0 = room.pos
        for x in range(x0, x0 + room.width):
            for y in range(y0, y0 + room.height):
                if x == x0 or x == x0 + room.width - 1 or y == y0 or y == y0 + room.height - 1:
                    tiles.append((x, y))
    return tiles


def generate_street_tiles(rooms: List[RoomDef], drop_zone: DropZoneDef,
                          grid_w: int, grid_h: int) -> List[Tuple[int, int]]:
    """Generate street tiles: corridor at mid-height, vertical paths from doormats to corridor,
    horizontal path from corridor to drop zone."""
    tiles_set = set()
    corridor_y = grid_h // 2

    # Horizontal corridor spanning the full width
    for x in range(1, grid_w - 1):
        tiles_set.add((x, corridor_y))

    # Vertical paths from each doormat to the corridor
    for room in rooms:
        mat_x, mat_y = room.doormat
        y_start, y_end = min(mat_y, corridor_y), max(mat_y, corridor_y)
        for y in range(y_start, y_end + 1):
            tiles_set.add((mat_x, y))

    # Path from corridor to drop zone
    dz_x, dz_y = drop_zone.location
    for y in range(min(corridor_y, dz_y), max(corridor_y, dz_y + drop_zone.height) + 1):
        tiles_set.add((dz_x - 1, y))

    # Remove tiles that are inside rooms (walls/interior)
    room_cells = set()
    for room in rooms:
        x0, y0 = room.pos
        for x in range(x0, x0 + room.width):
            for y in range(y0, y0 + room.height):
                room_cells.add((x, y))

    tiles_set -= room_cells
    return list(tiles_set)


def generate_area_signs(rooms: List[RoomDef]) -> List[Tuple[Tuple[int, int], int]]:
    """Return (location, area_id) pairs for area signs placed near each room's door."""
    signs = []
    for room in rooms:
        # Place sign near doormat
        mat_x, mat_y = room.doormat
        signs.append(((mat_x, mat_y), room.id))
    return signs


# ── Preset: static (14-room world with randomised victims and obstacles) ──────

def preset_static(seed=None, **kwargs) -> WorldPreset:
    """14-room world with 1–2 random victims per room (≥5 critical) and
    obstacles at 10/14 room entrances (≥4 big rocks). Seeded for reproducibility."""
    rng = random.Random(seed)

    rooms = [
        # Row 1 — North entry (door on bottom wall)
        RoomDef(1,  (1,  1),  5, 4, (3,  4),  (3,  5),  'North'),
        RoomDef(2,  (7,  1),  5, 4, (9,  4),  (9,  5),  'North'),
        RoomDef(3,  (13, 1),  5, 4, (15, 4),  (15, 5),  'North'),
        RoomDef(4,  (19, 1),  5, 4, (21, 4),  (21, 5),  'North'),
        # Row 2 — South entry (door on top wall)
        RoomDef(5,  (1,  7),  5, 4, (3,  7),  (3,  6),  'South'),
        RoomDef(6,  (7,  7),  5, 4, (9,  7),  (9,  6),  'South'),
        RoomDef(7,  (13, 7),  5, 4, (15, 7),  (15, 6),  'South'),
        # Row 3 — North entry
        RoomDef(8,  (1,  13), 5, 4, (3,  16), (3,  17), 'North'),
        RoomDef(9,  (7,  13), 5, 4, (9,  16), (9,  17), 'North'),
        RoomDef(10, (13, 13), 5, 4, (15, 16), (15, 17), 'North'),
        # Row 4 — South entry
        RoomDef(11, (1,  19), 5, 4, (3,  19), (3,  18), 'South'),
        RoomDef(12, (7,  19), 5, 4, (9,  19), (9,  18), 'South'),
        RoomDef(13, (13, 19), 5, 4, (15, 19), (15, 18), 'South'),
        RoomDef(14, (19, 19), 5, 4, (21, 19), (21, 18), 'South'),
    ]

    # ── Victim placement ──────────────────────────────────────────────────────
    # Each room gets 1 or 2 victims at random non-overlapping interior cells.
    # obs_potential is the cell just inside the door (direction depends on entry
    # side) — excluded from victim placement to avoid blocking the obstacle spot.
    # Guarantee ≥5 critical: the first victim in each of rooms[0..4] is always
    # drawn from CRITICAL. All other slots favour MILD 70/30 to produce more
    # yellow (mildly injured) victims across the map.

    _CRITICAL = [
        ('critically injured girl',          '/images/critically injured girl.svg'),
        ('critically injured elderly woman', '/images/critically injured elderly woman.svg'),
        ('critically injured man',           '/images/critically injured man.svg'),
        ('critically injured dog',           '/images/critically injured dog.svg'),
    ]
    _MILD = [
        ('mildly injured boy',         '/images/mildly injured boy.svg'),
        ('mildly injured elderly man', '/images/mildly injured elderly man.svg'),
        ('mildly injured woman',       '/images/mildly injured woman.svg'),
        ('mildly injured cat',         '/images/mildly injured cat.svg'),
    ]

    for idx, room in enumerate(rooms):
        interior = _interior_cells(room)
        # South rooms: door is on top wall, obstacle goes one cell down (into room).
        # North rooms: door is on bottom wall, obstacle goes one cell up (into room).
        obs_dy = 1 if room.enter_direction == 'South' else -1
        obs_potential = (room.door[0], room.door[1] + obs_dy)
        available = [c for c in interior if c != room.door and c != obs_potential]
        n_victims = rng.randint(1, 2)
        used: set = set()
        for j in range(n_victims):
            avail = [c for c in available if c not in used]
            if not avail:
                break
            loc = rng.choice(avail)
            used.add(loc)
            if idx < 5 and j == 0:
                pool = _CRITICAL
            else:
                pool = rng.choices([_CRITICAL, _MILD], weights=[1, 3], k=1)[0]
            name, img = rng.choice(pool)
            room.victims.append(VictimDef(name=name, img=img,
                                          location=loc, area=f'area {room.id}'))

    # ── Obstacle placement ────────────────────────────────────────────────────
    # 10 of 14 rooms receive one obstacle just inside their door entrance.
    # South-entry rooms: obstacle at door[1]+1 (one cell below the top door).
    # North-entry rooms: obstacle at door[1]-1 (one cell above the bottom door).
    # Exactly 4 rocks are guaranteed by building the type list before shuffling.

    _OBS_NON_ROCK = [
        ('stone', '/images/stone-small.svg'),
        ('tree',  '/images/tree-fallen2.svg'),
    ]
    room_indices = list(range(len(rooms)))
    rng.shuffle(room_indices)
    obs_type_list = (
        [('rock', '/images/stone.svg')] * 4 +
        [rng.choice(_OBS_NON_ROCK) for _ in range(6)]
    )
    rng.shuffle(obs_type_list)
    for i, room_idx in enumerate(room_indices[:10]):
        room = rooms[room_idx]
        obs_name, obs_img = obs_type_list[i]
        obs_dy = 1 if room.enter_direction == 'South' else -1
        room.obstacles.append(ObstacleDef(
            name=obs_name, img=obs_img,
            location=[room.door[0], room.door[1] + obs_dy],
        ))

    # ── Ghost victims and drop zone ───────────────────────────────────────────
    # Fixed 8 ghost blocks (one per victim type) keeps the drop zone compact.
    # Grid expanded to 30×30; drop zone at x=28 (clear of all rooms ≤x=23).
    ghost_victims = list(VICTIM_POOL)
    total_victims = sum(len(r.victims) for r in rooms)

    # ── Decorative overrides (unchanged — covers all 14 rooms visually) ───────
    decorative_overrides = {
        'roof_tiles': [
            (1,1),(2,1),(3,1),(4,1),(5,1),(1,2),(1,3),(1,4),(2,4),(4,4),(5,4),(5,3),(5,2),(7,1),(8,1),(9,1),
            (10,1),(11,1),(7,2),(7,3),(7,4),(8,4),(11,2),(11,3),(11,4),(10,4),(16,4),(17,4),(17,3),(17,2),
            (13,1),(14,1),(15,1),(16,1),(17,1),(13,2),(13,3),(13,4),(14,4),
            (19,1),(20,1),(21,1),(22,1),(23,1),(19,2),(19,3),(19,4),(20,4),(22,4),(23,4),(23,3),(23,2),
            (1,7),(1,8),(1,9),(1,10),(2,10),(3,10),(4,10),(5,10),(5,9),(5,8),(5,7),(4,7),(2,7),
            (13,7),(13,8),(13,9),(13,10),(14,10),(15,10),(16,10),(17,10),(17,9),(17,8),(17,7),(16,7),(14,7),
            (1,13),(2,13),(3,13),(4,13),(5,13),(1,14),(1,15),(1,16),(2,16),(4,16),(5,16),(5,15),(5,14),
            (7,13),(8,13),(9,13),(10,13),(11,13),(7,14),(7,15),(7,16),(8,16),(10,16),(11,16),(11,15),(11,14),
            (13,13),(14,13),(15,13),(16,13),(17,13),(13,14),(13,15),(13,16),(14,16),(17,14),(17,15),(17,16),(16,16),
            (1,19),(2,19),(4,19),(5,19),(1,20),(1,21),(1,22),(2,22),(3,22),(4,22),(5,22),(5,21),(5,20),
            (7,19),(8,19),(4,19),(5,19),(7,20),(7,21),(7,22),(8,22),(9,22),(10,22),(11,22),(11,21),(11,20),(11,19),
            (13,19),(14,19),(16,19),(17,19),(13,20),(13,21),(13,22),(14,22),(15,22),(16,22),(17,22),(17,21),(17,20),
            (19,19),(20,19),(22,19),(23,19),(19,20),(19,21),(19,22),(20,22),(21,22),(22,22),(23,22),(23,21),(23,20),
            (7,7),(7,8),(7,9),(7,10),(8,10),(9,10),(10,10),(11,10),(11,9),(11,8),(11,7),(10,7),(8,7),(10,19),
        ],
        'street_tiles': [
            (11,5),(13,5),(14,5),(13,6),(14,6),(12,5),(15,5),(15,6),(16,5),(16,6),(17,5),(17,6),(18,5),
            (8,6),(7,6),(6,6),(5,6),(4,6),(3,6),(2,6),(1,6),(20,9),(21,9),(21,14),(20,14),(19,14),(9,6),
            (1,5),(2,5),(3,5),(4,5),(5,5),(22,11),(22,12),(19,18),(18,18),(17,18),(16,18),(15,18),(13,17),
            (11,17),(10,17),(8,18),(7,18),(6,18),(5,18),(4,18),(3,18),(2,18),(1,18),(12,17),(18,6),
        ],
        'street_tiles_alt': [
            (21,10),(21,11),(21,12),(21,13),(19,15),(19,16),
        ],
        'plants': [
            (12,3),(12,4),(18,1),(18,2),(18,3),(18,4),(6,19),(6,20),(6,21),(18,19),
        ],
        'decorative_objects': [
            {'pos': (1, 12), 'name': 'plant', 'img': '/images/tree.svg', 'size': 3},
            {'pos': (21, 7), 'name': 'heli', 'img': '/images/helicopter.svg', 'size': 3, 'traversable': False},
            {'pos': (21, 16), 'name': 'ambulance', 'img': '/images/ambulance.svg', 'size': 2.3, 'traversable': False},
        ],
        'keyboard_sign': (12, 0),
        'area_signs': [
            ((3,1), '01', 0.5), ((9,1), '02', 0.55), ((15,1), '03', 0.55), ((21,1), '04', 0.55),
            ((3,10), '05', 0.55), ((9,10), '06', 0.55), ((15,10), '07', 0.55),
            ((3,13), '08', 0.55), ((9,13), '09', 0.55), ((15,13), '10', 0.55),
            ((3,22), '11', 0.45), ((9,22), '12', 0.55), ((15,22), '13', 0.55), ((21,22), '14', 0.55),
        ],
    }

    return WorldPreset(
        name='static',
        grid_width=25,
        grid_height=24,
        rooms=rooms,
        drop_zone=DropZoneDef((23, 8), 8),
        ghost_victims=ghost_victims,
        decorative_overrides=decorative_overrides,
        seed=seed,
    )


# ── Preset: 2 small houses ───────────────────────────────────────────────────

def preset_2_houses(seed=None, num_victims=3, **kwargs) -> WorldPreset:
    """Two small 5x4 houses with configurable victims."""
    """Reproduce the exact current hardcoded world layout."""
    rooms = [
        RoomDef(1, (1, 1), 5, 4, (3, 4), (3, 5), 'North'),
        RoomDef(2, (7, 1), 5, 4, (9, 4), (9, 5), 'North'),
    ]

    # Exact current victim placements
    rooms[0].victims = [VictimDef('mildly injured boy', '/images/mildly injured boy.svg', (2, 2), 'area 1')]
    rooms[1].victims = [
        VictimDef('critically injured dog', '/images/critically injured dog.svg', (10, 3), 'area 2'),
        VictimDef('critically injured girl', '/images/critically injured girl.svg', (10, 2), 'area 2'),
    ]

    ghost_victims = [
        ('critically injured girl', '/images/critically injured girl.svg'),
        ('critically injured dog', '/images/critically injured dog.svg'),
        ('mildly injured boy', '/images/mildly injured boy.svg'),
    ]

    rng = random.Random(seed)

    _place_obstacles_at_doors(rooms, rng, probability=1)

    # Exact current decorative object coordinates
    decorative_overrides = {
        'roof_tiles': [
            (1,1),(2,1),(3,1),(4,1),(5,1),(1,2),(1,3),(1,4),(2,4),(4,4),(5,4),(5,3),(5,2),(7,1),(8,1),(9,1),
            (10,1),(11,1),(7,2),(7,3),(7,4),(8,4),(11,2),(11,3),(11,4),(10,4),

        ],
        'street_tiles': [
            (11,5),(13,5),(14,5),(13,6),(14,6),(12,5),(15,5),(15,6),(16,5),(16,6),(17,5),(17,6),(18,5),
            (8,6),(7,6),(6,6),(5,6),(4,6),(3,6),(2,6),(1,6),(20,9),(21,9),(21,14),(20,14),(19,14),(9,6),
            (1,5),(2,5),(3,5),(4,5),(5,5),(22,11),(22,12),(19,18),(18,18),(17,18),(16,18),(15,18),(13,17),
            (11,17),(10,17),(8,18),(7,18),(6,18),(5,18),(4,18),(3,18),(2,18),(1,18),(12,17),(18,6),
        ],
        'street_tiles_alt': [
            (21,10),(21,11),(21,12),(21,13),(19,15),(19,16),
        ],
        'plants': [
            (12,3),(12,4),(18,1),(18,2),(18,3),(18,4),(6,19),(6,20),(6,21),(18,19),
        ],
        'decorative_objects': [
            {'pos': (1, 12), 'name': 'plant', 'img': '/images/tree.svg', 'size': 3},
            {'pos': (21, 7), 'name': 'heli', 'img': '/images/helicopter.svg', 'size': 3, 'traversable': False},
            {'pos': (21, 16), 'name': 'ambulance', 'img': '/images/ambulance.svg', 'size': 2.3, 'traversable': False},
        ],
        'keyboard_sign': (12, 0),
        'area_signs': [
            ((3,1), '01', 0.5), ((9,1), '02', 0.55),
        ],
    }

    return WorldPreset(
        name='preset2',
        grid_width=25,
        grid_height=24,
        rooms=rooms,
        drop_zone=DropZoneDef((23, 8), 3),
        ghost_victims=ghost_victims,
        decorative_overrides=decorative_overrides,
        seed=None,
    )

# ── Preset: 2 big houses (6x6 + 10x4) ──────────────────────────────────────

def preset_2_big_houses(seed=None, num_victims=5, **kwargs) -> WorldPreset:
    """One 6x6 room and one 10x4 room (10 wide, 4 tall)."""
    rng = random.Random(seed)

    rooms = [
        RoomDef(1, (1, 1), 6, 6, (4, 6), (4, 7), 'North'),
        RoomDef(2, (9, 1), 10, 4, (14, 4), (14, 5), 'North'),
    ]

    _place_victims_in_rooms(rooms, num_victims, rng)
    _place_obstacles_at_doors(rooms, rng, probability=0.5)

    ghost_victims = _collect_all_victims(rooms)
    dz_height = max(len(ghost_victims), 1)

    return WorldPreset(
        name='preset3',
        grid_width=22,
        grid_height=12,
        rooms=rooms,
        drop_zone=DropZoneDef((20, 1), dz_height),
        ghost_victims=ghost_victims,
        seed=seed,
    )


# ── Preset: random ───────────────────────────────────────────────────────────

def preset_random(seed=None, num_rooms=4, num_victims=5, **kwargs) -> WorldPreset:
    """Fully randomized world layout.

    Args:
        seed: Random seed for reproducibility.
        num_rooms: Number of rooms to generate (1-10).
        num_victims: Number of victims to distribute across rooms.
    """
    rng = random.Random(seed)

    # Grid size scales with room count, capped at 30x30
    base = max(18, 10 + num_rooms * 4)
    grid_w = min(30, base)
    grid_h = min(30, base)

    # Reserve rightmost 3 columns for drop zone + agent start area
    playable_w = grid_w - 3

    rooms: List[RoomDef] = []
    for i in range(num_rooms):
        placed = False
        for _ in range(200):
            w = rng.randint(4, 10)
            h = rng.randint(4, 10)
            # Keep within playable area (leave 1-cell border for world bounds wall)
            x = rng.randint(1, max(1, playable_w - w))
            y = rng.randint(1, max(1, grid_h - h - 2))

            # Check overlap with existing rooms (2-cell margin)
            overlap = False
            for existing in rooms:
                ex, ey = existing.pos
                if _rects_overlap(x, y, w, h, ex, ey, existing.width, existing.height, margin=2):
                    overlap = True
                    break
            if overlap:
                continue

            # Pick door on top or bottom wall
            if rng.random() < 0.5:
                door, doormat, enter_dir = _compute_door_bottom(x, y, w, h)
            else:
                door, doormat, enter_dir = _compute_door_top(x, y, w, h)

            # Ensure doormat is within grid bounds
            if not (0 < doormat[0] < grid_w - 1 and 0 < doormat[1] < grid_h - 1):
                continue

            rooms.append(RoomDef(
                id=i + 1, pos=(x, y), width=w, height=h,
                door=door, doormat=doormat, enter_direction=enter_dir,
            ))
            placed = True
            break

        if not placed:
            print(f"[WorldPresets] Warning: Could not place room {i + 1} after 200 attempts")

    _place_victims_in_rooms(rooms, num_victims, rng)
    _place_obstacles_at_doors(rooms, rng, probability=0.5)

    ghost_victims = _collect_all_victims(rooms)
    dz_height = max(len(ghost_victims), 1)

    # Drop zone on the right edge
    dz_x = grid_w - 2
    dz_y = max(1, (grid_h - dz_height) // 2)

    return WorldPreset(
        name='random',
        grid_width=grid_w,
        grid_height=grid_h,
        rooms=rooms,
        drop_zone=DropZoneDef((dz_x, dz_y), dz_height),
        ghost_victims=ghost_victims,
        seed=seed,
    )


# ── Preset registry ──────────────────────────────────────────────────────────

PRESET_REGISTRY = {
    'static': preset_static,
    'preset1': preset_static,
    'preset2': preset_2_houses,
    'preset3': preset_2_big_houses,
    'random': preset_random,
}


def get_preset(name: str, seed=None, **kwargs) -> WorldPreset:
    """Resolve a preset by name. Passes seed and any extra kwargs to the factory."""
    if name not in PRESET_REGISTRY:
        raise ValueError(f"Unknown world preset '{name}'. Available: {list(PRESET_REGISTRY.keys())}")
    return PRESET_REGISTRY[name](seed=seed, **kwargs)
