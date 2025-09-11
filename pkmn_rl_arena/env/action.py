from pkmn_rl_arena.logging import logger
from .battle_core import BattleCore, TurnType

import numpy as np
import numpy.typing as npt
from typing import Dict

ACTION_SPACE_SIZE = 9


class ActionManager:
    """
    Manages action validation and execution.
    """

    def __init__(self, battle_core: BattleCore):
        self.battle_core = battle_core
        self.action_space_size = 10  # Actions 0-9

    def is_valid_action(self, agent, action: int) -> bool:
        """Check if action is valid."""
        return 0 <= action <= 9 and self.get_action_mask(agent)[action] == 1

    @staticmethod
    def check_agent_match_turntype(agent, turn_type) -> bool:
        return (
            agent == "player" and turn_type in [TurnType.PLAYER, TurnType.GENERAL]
        ) or (agent == "enemy" and turn_type in [TurnType.ENEMY, TurnType.GENERAL])

    def write_actions(
        self, turn_type: TurnType, actions: Dict[str, int]
    ) -> Dict[str, bool]:
        """Write actions based on turn type"""
        action_written = {agent: False for agent in actions.keys()}
        for agent, action in actions.items():
            if not self.check_agent_match_turntype(agent, turn_type):
                raise ValueError(
                    f'Error : write_actions : invalid agent, expected "player" or "enemy", got {agent}'
                )

            if not self.is_valid_action(agent, action):
                logger.warning(
                    f"Trying to write invalid action : authorized, values {self.get_action_mask(agent)}, got {action}."
                )
                action_written[agent] = False
                continue

            self.battle_core.write_action(agent, actions[agent])
            action_written[agent] = True

        return action_written

    def get_action_mask(self, agent: str) -> npt.NDArray[int]:
        """
        Creates an action mask for the agent. The mask is here to signal the unauthorized actions
        It returns illegal moves : (PP == 0)
        """
        legal_moves = self.battle_core.gba.read_u16_list(
            self.battle_core.addrs[f"legalMoveActions{agent.capitalize()}"], 4
        )
        legal_switches = self.battle_core.gba.read_u16_list(
            self.battle_core.addrs[f"legalSwitchActions{agent.capitalize()}"], 6
        )

        valid_moves = [i for i, move in enumerate(legal_moves) if move]
        valid_switches = [
            i + 4 for i, switch in enumerate(legal_switches) if switch
        ]  # Offset switches by 4

        # Combine moves and switches into a single list of legal actions
        valid_actions = [valid_moves + valid_switches]

        action_mask = np.zeros(shape=ACTION_SPACE_SIZE)
        action_mask[valid_actions] = 1

        return action_mask
