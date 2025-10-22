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

type_chart = {
    "NORM":   {"NORM":1, "FIRE":1, "WTR":1, "ELEK":1, "GRASS":1, "ICE":1, "FIGHT":2, "PSN":1, "GRND":1, "FLY":1, "PSY":1, "BUG":1, "ROCK":0.5, "GHOST":0, "DRAGON":1, "DARK":1, "STEEL":0.5},
    "FIRE":     {"NORM":1, "FIRE":0.5, "WTR":0.5, "ELEK":1, "GRASS":2, "ICE":2, "FIGHT":1, "PSN":1, "GRND":1, "FLY":1, "PSY":1, "BUG":2, "ROCK":0.5, "GHOST":1, "DRAGON":0.5, "DARK":1, "STEEL":2},
    "WTR":    {"NORM":1, "FIRE":2, "WTR":0.5, "ELEK":1, "GRASS":0.5, "ICE":1, "FIGHT":1, "PSN":1, "GRND":2, "FLY":1, "PSY":1, "BUG":1, "ROCK":2, "GHOST":1, "DRAGON":0.5, "DARK":1, "STEEL":1},
    "ELEK": {"NORM":1, "FIRE":1, "WTR":2, "ELEK":0.5, "GRASS":0.5, "ICE":1, "FIGHT":1, "PSN":1, "GRND":0, "FLY":2, "PSY":1, "BUG":1, "ROCK":1, "GHOST":1, "DRAGON":0.5, "DARK":1, "STEEL":1},
    "GRASS":    {"NORM":1, "FIRE":0.5, "WTR":2, "ELEK":1, "GRASS":0.5, "ICE":1, "FIGHT":1, "PSN":0.5, "GRND":2, "FLY":0.5, "PSY":1, "BUG":0.5, "ROCK":2, "GHOST":1, "DRAGON":0.5, "DARK":1, "STEEL":0.5},
    "ICE":      {"NORM":1, "FIRE":0.5, "WTR":0.5, "ELEK":1, "GRASS":2, "ICE":0.5, "FIGHT":1, "PSN":1, "GRND":2, "FLY":2, "PSY":1, "BUG":1, "ROCK":1, "GHOST":1, "DRAGON":2, "DARK":1, "STEEL":0.5},
    "FIGHT": {"NORM":2, "FIRE":1, "WTR":1, "ELEK":1, "GRASS":1, "ICE":2, "FIGHT":1, "PSN":0.5, "GRND":1, "FLY":0.5, "PSY":0.5, "BUG":0.5, "ROCK":2, "GHOST":0, "DRAGON":1, "DARK":2, "STEEL":2},
    "PSN":   {"NORM":1, "FIRE":1, "WTR":1, "ELEK":1, "GRASS":2, "ICE":1, "FIGHT":1, "PSN":0.5, "GRND":0.5, "FLY":1, "PSY":1, "BUG":1, "ROCK":0.5, "GHOST":0.5, "DRAGON":1, "DARK":1, "STEEL":0},
    "GRND":   {"NORM":1, "FIRE":2, "WTR":1, "ELEK":2, "GRASS":0.5, "ICE":1, "FIGHT":1, "PSN":2, "GRND":1, "FLY":0, "PSY":1, "BUG":0.5, "ROCK":2, "GHOST":1, "DRAGON":1, "DARK":1, "STEEL":2},
    "FLY":   {"NORM":1, "FIRE":1, "WTR":1, "ELEK":0.5, "GRASS":2, "ICE":1, "FIGHT":2, "PSN":1, "GRND":1, "FLY":1, "PSY":1, "BUG":2, "ROCK":0.5, "GHOST":1, "DRAGON":1, "DARK":1, "STEEL":0.5},
    "PSY":  {"NORM":1, "FIRE":1, "WTR":1, "ELEK":1, "GRASS":1, "ICE":1, "FIGHT":2, "PSN":2, "GRND":1, "FLY":1, "PSY":0.5, "BUG":1, "ROCK":1, "GHOST":1, "DRAGON":1, "DARK":0, "STEEL":0.5},
    "BUG":      {"NORM":1, "FIRE":0.5, "WTR":1, "ELEK":1, "GRASS":2, "ICE":1, "FIGHT":0.5, "PSN":0.5, "GRND":1, "FLY":0.5, "PSY":2, "BUG":1, "ROCK":1, "GHOST":0.5, "DRAGON":1, "DARK":2, "STEEL":0.5},
    "ROCK":     {"NORM":1, "FIRE":2, "WTR":1, "ELEK":1, "GRASS":1, "ICE":2, "FIGHT":0.5, "PSN":1, "GRND":0.5, "FLY":2, "PSY":1, "BUG":2, "ROCK":1, "GHOST":1, "DRAGON":1, "DARK":1, "STEEL":0.5},
    "GHOST":    {"NORM":0, "FIRE":1, "WTR":1, "ELEK":1, "GRASS":1, "ICE":1, "FIGHT":1, "PSN":1, "GRND":1, "FLY":1, "PSY":2, "BUG":1, "ROCK":1, "GHOST":2, "DRAGON":1, "DARK":0.5, "STEEL":1},
    "DRAGON":   {"NORM":1, "FIRE":1, "WTR":1, "ELEK":1, "GRASS":1, "ICE":1, "FIGHT":1, "PSN":1, "GRND":1, "FLY":1, "PSY":1, "BUG":1, "ROCK":1, "GHOST":1, "DRAGON":2, "DARK":1, "STEEL":0.5},
    "DARK":     {"NORM":1, "FIRE":1, "WTR":1, "ELEK":1, "GRASS":1, "ICE":1, "FIGHT":0.5, "PSN":1, "GRND":1, "FLY":1, "PSY":2, "BUG":1, "ROCK":1, "GHOST":2, "DRAGON":1, "DARK":0.5, "STEEL":1},
    "STEEL":    {"NORM":1, "FIRE":0.5, "WTR":0.5, "ELEK":0.5, "GRASS":1, "ICE":2, "FIGHT":1, "PSN":1, "GRND":1, "FLY":1, "PSY":1, "BUG":1, "ROCK":2, "GHOST":1, "DRAGON":1, "DARK":1, "STEEL":0.5}
}

def effectiveness(attack_type: str, defense_type: str) -> float:
    return type_chart[attack_type][defense_type]