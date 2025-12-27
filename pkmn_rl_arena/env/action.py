from pkmn_rl_arena.logging import log
from .battle_core import CoreContext
from .battle_state import TurnType

import numpy as np
import numpy.typing as npt
from typing import Dict

ACTION_SPACE_SIZE = 10


class ActionManager:
    """
    Manages action validation and execution.
    """

    allowed_turntype = {
        "player": [TurnType.PLAYER, TurnType.GENERAL],
        "enemy": [TurnType.ENEMY, TurnType.GENERAL],
    }

    def __init__(self, core_context: CoreContext):
        self.ctxt = core_context
        self.action_space_size = 10  # Actions 0-9

    def is_valid_action(self, agent, action_id: int) -> bool:
        """Check if action is valid."""
        return 0 <= action_id <= 9 and self.get_action_mask(agent)[action_id] == 1

    def write_actions(self, actions: Dict[str, int]) -> Dict[str, bool]:
        """Write actions based on turn type"""
        action_written = {agent: False for agent in actions.keys()}

        current_turn = self.ctxt.core.get_current_turn()
        required_agents = self.ctxt.core.get_required_agents()
        log.debug(f"write_actions: current_turn={current_turn}, required_agents={required_agents}, incoming={actions}")

        for agent, action in actions.items():
            if agent not in required_agents:
                log.warning(f"write_actions: skipping {agent} (not required this turn)")
                continue

            mask = self.get_action_mask(agent)
            log.debug(f"write_actions: action_mask for {agent} = {mask}")

            if not self.is_valid_action(agent, int(action)):
                log.critical(
                    f"Trying to write invalid action : authorized, values {mask}, got {action}."
                )
                raise ValueError(
                    f"Error : write_actions : invalid action ({action}), for current agent ({agent})."
                )

            self.ctxt.core.write_action(agent, int(action))
            action_written[agent] = True

        return action_written

    def get_valid_action_ids(self, agent: str) -> list[int]:
        """
        Return valid action ids
        It returns a subset of the action space, removing illegal moves
        (such as PP == 0 & ko  pkmn (forbidden switch))
        Note : Action space :
            [0,1,2,3,4,5,6,7,8,9]
        """
        legal_moves = self.ctxt.core.gba.read_u16_list(
            self.ctxt.core.mem_addrs[f"legalMoveActions{agent.capitalize()}"], 4
        )
        legal_switches = self.ctxt.core.gba.read_u16_list(
            self.ctxt.core.mem_addrs[f"legalSwitchActions{agent.capitalize()}"], 6
        )

        valid_moves = [i for i, move in enumerate(legal_moves) if move]
        valid_switches = [
            i + 4 for i, switch in enumerate(legal_switches) if switch
        ]  # Offset switches by 4

        # Combine moves and switches into a single list of legal actions
        return valid_moves + valid_switches

    def get_action_mask(self, agent: str) -> npt.NDArray[np.float32]:
        """
        Creates an action mask for the agent. The mask is here to signal the unauthorized actions
        It returns illegal moves : (PP == 0 & ko  pkmn (forbidden switch))
        Action space :
            [0,1,2,3,4,5,6,7,8,9]
        Example for an action mask :
            [0,1,1,0,1,1,1,0,0,1]
        """

        action_mask = np.zeros(shape=ACTION_SPACE_SIZE, dtype=np.float32) # Explicit dtype
        action_mask[self.get_valid_action_ids(agent)] = 1.0
        return action_mask
