from enum import Enum
from dataclasses import dataclass
from typing import List

class TurnType(Enum):
    """Enumeration for different turn types"""
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
    current_turn: TurnType = TurnType.CREATE_TEAM
    waiting_for_action: bool = False
    episode_steps: int = 0

    def is_battle_done(self) -> bool:
        """Check if battle is finished"""
        return self.current_turn == TurnType.DONE

    def get_current_turn(self) -> TurnType:
        """Get current turn type"""
        return self.current_turn

    def get_required_agents(self) -> List[str]:
        """Get list of agents required for current turn"""
        if self.current_turn == TurnType.GENERAL:
            return ["player", "enemy"]
        elif self.current_turn == TurnType.PLAYER:
            return ["player"]
        elif self.current_turn == TurnType.ENEMY:
            return ["enemy"]
        else:
            return []
