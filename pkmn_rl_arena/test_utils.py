from .env.action import ActionManager
from .env.battle_core import BattleCore

import random

def advance_turn(core: BattleCore, action_manager: ActionManager):
    """
    Choses a random action from valid actions & run to next turn.
    """
    actions = {
        agent: random.choice(action_manager.get_valid_action_ids(agent))
        for agent in core.get_required_agents()
    }

    for agent, action in actions.items():
        core.write_action(agent, action)

    core.advance_to_next_turn()
    return


