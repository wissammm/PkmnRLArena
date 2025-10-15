from dataclasses import dataclass


@dataclass(frozen=True)
class GenParamsSize:
    """
    Counter for pkmn team generation params
    """
    ID : int = 1
    LEVEL : int = 1
    MOVES : int = 4
    HP_PERCENT : int = 1
    ITEM : int = 1
    PARTY_SIZE : int = 6
    PKMN : int = ID + LEVEL + MOVES + HP_PERCENT + ITEM
    TEAM : int = PARTY_SIZE * PKMN


@dataclass(frozen=True)
class Ranges:
    """
    Gives min & max values for pokemon values, be it generaion params or stats params.
    """
    ITEM_1 = [178, 225]
    ITEM_2 = [132, 175]
    ALL_ITEMS = list(range(ITEM_1[1], ITEM_1[0], -1)) + list(
        range(ITEM_2[1], ITEM_2[0], -1)
    )
    SPECIES_ID = [0, 411]
    TYPE = [0, 17]
    STATS = [0, 550]
    HP = [0, 550]
    MOVE_ID = [0, 354]
    POWER = [0, 256]
    ACCURACY = [0, 100]
    PP = [0, 40]
    LEVEL = [1, 100]
    FRIENDSHIP = [0, 255]


"""
ID for each pkmn type
Note:
Mystery type must be a special event type & therefore mustn't be counted when summing types.
This is why Ranges.TYPES[1] = 17
"""
TYPES_ID = {
    255: "NONE",
    0: "NORM",
    1: "FIGHT",
    2: "FLY",
    3: "PSN",
    4: "GRND",
    5: "ROCK",
    6: "BUG",
    7: "GHOST",
    8: "STEEL",
    9: "MYSTERY",
    10: "FIRE",
    11: "WTR",
    12: "GRASS",
    13: "ELEK",
    14: "PSY",
    15: "ICE",
    16: "DRAGON",
    17: "DARK",
    18: "NUMBER_OF_MON_TYPES",
}
