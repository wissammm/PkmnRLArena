from enum import Enum

class TurnType(Enum):
    """Enumeration for different turn types"""

    CREATE_TEAM = 0  # Initial team creation
    GENERAL = 1  # Both players act simultaneously
    PLAYER = 2  # Only player acts
    ENEMY = 3  # Only enemy acts
    DONE = 4  # Battle is finished
