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

    @staticmethod
    def check_agent_match_turntype(agent, turn_type) -> bool:
        return (
            agent == "player" and turn_type in [TurnType.PLAYER, TurnType.GENERAL]
        ) or (agent == "enemy" and turn_type in [TurnType.ENEMY, TurnType.GENERAL])

    def write_actions(self, turn_type: TurnType, actions: Dict[str, int])-> Dict[str,bool]:
        """Write actions based on turn type"""
        action_written = {"player": False, "enemy" : False}
        for agent , action in actions.items():
            if not self.check_agent_match_turntype(agent, turn_type):
                raise ValueError(f"Error : write_actions : invalid agent, expected \"player\" or \"enemy\", got {agent}")
            if not self.is_valid_action(action):
                continue
            self.battle_core.write_action(agent, actions[agent])
            action_written[agent] = True

        return action_written


    def get_legal_actions(self, agent: str) -> List[int]:
        match agent:
            case "player":
                legal_moves = self.battle_core.gba.read_u16_list(
                    self.battle_core.addrs["legalMoveActionsPlayer"], 4
                )
                legal_switches = self.battle_core.gba.read_u16_list(
                    self.battle_core.addrs["legalSwitchActionsPlayer"], 6
                )
            case "enemy":
                legal_moves = self.battle_core.gba.read_u16_list(
                    self.battle_core.addrs["legalMoveActionsEnemy"], 4
                )
                legal_switches = self.battle_core.gba.read_u16_list(
                    self.battle_core.addrs["legalSwitchActionsEnemy"], 6
                )
            case _:
                raise ValueError(f"Unknown agent: {agent}")

        valid_moves = [i for i, move in enumerate(legal_moves) if move]
        valid_switches = [
            i + 4 for i, switch in enumerate(legal_switches) if switch
        ]  # Offset switches by 4

        # Combine moves and switches into a single list of legal actions
        return valid_moves + valid_switches
