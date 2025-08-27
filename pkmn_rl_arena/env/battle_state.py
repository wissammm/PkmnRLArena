from enum import Enum
from dataclasses import dataclass

class TurnType(Enum):
    """Enumeration for different turn types"""

    NOT_STARTED ="not_started"
    CREATE_TEAM = "create_team"  # Initial team creation
    GENERAL = "general"  # Both players act simultaneously
    PLAYER = "player"  # Only player acts
    ENEMY = "enemy"  # Only enemy acts
    DONE = "done"  # Battle is finished


@dataclass
class BattleState:
    """Represents the current state of the battle"""

    current_step: int = 0
    episode_done: bool = False
    current_turn: TurnType = TurnType.NOT_STARTED
    waiting_for_action: bool = False
    episode_steps: int = 0
