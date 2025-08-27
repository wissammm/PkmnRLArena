from .battle_core import BattleCore
from .battle_state import TurnType

from typing import Dict, List

class ActionManager:
    """
    Manages action validation and execution.
    """

    def __init__(self, battle_core: BattleCore):
        self.battle_core = battle_core
        self.action_space_size = 10  # Actions 0-9

    @staticmethod
    def is_valid_action(action: int) -> bool:
        """Check if action is valid (simplified version)"""
        return 0 <= action <= 9

    def write_actions(self, turn_type: TurnType, actions: Dict[str, int]):
        """Write actions based on turn type"""
        if turn_type == TurnType.PLAYER and "player" in actions:
            self.battle_core.write_action("player", actions["player"])
        elif turn_type == TurnType.ENEMY and "enemy" in actions:
            self.battle_core.write_action("enemy", actions["enemy"])
        elif turn_type == TurnType.GENERAL:
            # Handle simultaneous actions
            if "player" in actions:
                self.battle_core.write_action("player", actions["player"])
            if "enemy" in actions:
                self.battle_core.write_action("enemy", actions["enemy"])

    def get_legal_actions(self, agent: str) -> List[int]:
        if agent == "player":
            legal_moves = self.battle_core.gba.read_u16_list(
                self.battle_core.addrs["legalMoveActionsPlayer"], 4
            )
            legal_switches = self.battle_core.gba.read_u16_list(
                self.battle_core.addrs["legalSwitchActionsPlayer"], 6
            )
        elif agent == "enemy":
            legal_moves = self.battle_core.gba.read_u16_list(
                self.battle_core.addrs["legalMoveActionsEnemy"], 4
            )
            legal_switches = self.battle_core.gba.read_u16_list(
                self.battle_core.addrs["legalSwitchActionsEnemy"], 6
            )
        else:
            raise ValueError(f"Unknown agent: {agent}")

        valid_moves = [i for i, move in enumerate(legal_moves) if move]
        valid_switches = [
            i + 4 for i, switch in enumerate(legal_switches) if switch
        ]  # Offset switches by 4

        # Combine moves and switches into a single list of legal actions
        return valid_moves + valid_switches
