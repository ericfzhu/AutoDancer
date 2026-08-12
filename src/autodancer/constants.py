"""Stable identifiers shared by the simulator, live adapter, and trace files."""

from enum import IntEnum

GRID_SIZE = 21
GRID_CHANNELS = 7
PLAYER_FEATURES = 16
INVENTORY_SLOTS = 8
INVENTORY_FEATURES = 3
RGB_SIZE = 256


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5
    ITEM_1 = 6
    ITEM_2 = 7
    THROW = 8
    SPELL_1 = 9
    SPELL_2 = 10


ACTION_COUNT = len(Action)


class GridChannel(IntEnum):
    TERRAIN = 0
    ACTOR = 1
    HEALTH = 2
    ITEM = 3
    TRAP = 4
    VISIBILITY = 5
    STATUS = 6


class Terrain(IntEnum):
    UNKNOWN = 0
    FLOOR = 1
    WALL = 2
    STAIRS = 3


class ActorKind(IntEnum):
    NONE = 0
    PLAYER = 1
    GREEN_SLIME = 2
    BLUE_SLIME = 3
    SKELETON = 4
    BAT = 5
    ARMADILLO = 6
    WARLOCK = 7
    BLADEMASTER = 8
    BOSS = 9


class ItemKind(IntEnum):
    NONE = 0
    GOLD = 1
    FOOD = 2
    BOMB = 3


class TrapKind(IntEnum):
    NONE = 0
    SPIKE = 1


class StatusFlag(IntEnum):
    NONE = 0
    BOMB = 1
    EXIT_LOCKED = 2


DIRECTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (0, -1),
    Action.RIGHT: (1, 0),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
}

