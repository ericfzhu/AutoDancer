"""Stable identifiers shared by the Lua mod and Python live adapter."""

from enum import IntEnum

GRID_SIZE = 21
GRID_CHANNELS = 19
MAP_SIZE = 65
MAP_CHANNELS = 5
PLAYER_FEATURES = 20
INVENTORY_SLOTS = 13
INVENTORY_FEATURES = 8
TYPE_VOCAB_SIZE = 4096
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
    TERRAIN_CLASS = 0
    TERRAIN_TYPE = 1
    ACTOR_CLASS = 2
    ACTOR_TYPE = 3
    HEALTH = 4
    MAX_HEALTH = 5
    ITEM_CLASS = 6
    ITEM_TYPE = 7
    TRAP = 8
    VISIBILITY = 9
    STATUS = 10
    FACING = 11
    BEAT_DELAY = 12
    BEAT_INTERVAL = 13
    FROZEN_TURNS = 14
    CONFUSED_TURNS = 15
    CHARGE_STATE = 16
    CHARGE_DIRECTION = 17
    SHIELD_DIRECTION = 18


class MapChannel(IntEnum):
    TERRAIN_CLASS = 0
    REVEAL_STATE = 1
    VISIT_COUNT = 2
    VISIT_RECENCY = 3
    PLAYER = 4


class PlayerFeature(IntEnum):
    HEALTH = 0
    MAX_HEALTH = 1
    GOLD = 2
    GROOVE = 3
    X = 4
    Y = 5
    ZONE = 6
    FLOOR = 7
    TURN = 8
    BOMBS = 9
    WEAPON_DAMAGE = 10
    VISIBLE_ENEMIES = 11
    ON_STAIRS = 12
    TASK = 13
    WON = 14
    DEAD = 15
    MUSIC_ELAPSED_DS = 16
    MUSIC_LENGTH_DS = 17
    MUSIC_REMAINING_DS = 18
    SONG_END_REACHED = 19


class InventoryFeature(IntEnum):
    ITEM_CLASS = 0
    ITEM_TYPE = 1
    QUANTITY = 2
    WEAPON_DAMAGE = 3
    COOLDOWN_TURNS = 4
    COOLDOWN_KILLS = 5
    READY = 6
    ACTIVE = 7


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
    ZOMBIE = 10
    SLIME_3 = 11
    MONKEY = 12
    SKELETON_2 = 13
    DRAGON = 14
    OTHER = 15


class ItemKind(IntEnum):
    NONE = 0
    GOLD = 1
    FOOD = 2
    BOMB = 3
    DIAMOND = 4
    DAGGER = 5
    SHOVEL = 6
    OTHER = 7
    BROADSWORD = 8


class TrapKind(IntEnum):
    NONE = 0
    SPIKE = 1
    BOUNCE_RIGHT = 2
    BOUNCE_UP = 3
    BOUNCE_LEFT = 4
    BOUNCE_DOWN = 5
    TEMPO_DOWN = 6
    TRAPDOOR = 7


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
