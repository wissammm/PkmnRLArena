from .battle_core import BattleCore
from .battle_state import BattleState, TurnType
from .action import ActionManager

from typing import Dict, List


class TurnManager:
    """
    Manages turn logic and coordination between agents.
    """

    def __init__(self, battle_core: BattleCore, action_manager: ActionManager):
        self.battle_core = battle_core
        self.action_manager = action_manager
        self.state = BattleState()

    def process_turn(self, actions: Dict[str, int]) -> bool:
        """
        Process a turn with given actions.
        Returns True if turn was completed, False if waiting for more actions.
        """
        if self.state.current_turn == TurnType.DONE:
            return True

        # Check if we have all required actions
        required_agents = self._get_required_agents()
        if not all(agent in actions for agent in required_agents):
            return False

        # Write actions
        self.action_manager.write_actions(self.state.current_turn, actions)

        # Clear stop condition to continue
        self.battle_core.clear_stop_condition(self.state.current_turn)

        # Reset pending actions
        self.state.pending_actions = {"player": None, "enemy": None}
        self.state.waiting_for_action = False

        return True

