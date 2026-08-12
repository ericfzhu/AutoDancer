"""Curriculum task definitions and fixed reward values."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    max_turns: int
    regular_floors: int
    max_zone: int
    enemy_count: tuple[int, int]
    start_floor: int = 1
    include_boss: bool = False


TASKS: dict[str, TaskSpec] = {
    "navigation": TaskSpec("navigation", 96, 1, 1, (0, 0)),
    "single_enemy": TaskSpec("single_enemy", 96, 1, 1, (1, 1)),
    "mixed_room": TaskSpec("mixed_room", 160, 1, 1, (3, 5)),
    "floor": TaskSpec("floor", 384, 1, 1, (4, 8)),
    "zone": TaskSpec("zone", 1800, 3, 1, (5, 10), include_boss=True),
    "all_zones": TaskSpec("all_zones", 8000, 3, 4, (6, 12), include_boss=True),
}


REWARD_VALUES: dict[str, float] = {
    "success": 1.0,
    "failure": -1.0,
    "enemy_damage": 0.05,
    "enemy_kill": 0.20,
    "player_damage": -0.25,
    "reveal": 0.02,
    "turn": -0.001,
}

