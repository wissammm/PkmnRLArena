from pkmn_rl_arena.env.battle_core import BattleCore, TurnType
from pkmn_rl_arena.env.save_state import SaveStateManager
from pkmn_rl_arena.env.action import ActionManager
from pkmn_rl_arena.env.pkmn_team_factory import PkmnTeamFactory
from pkmn_rl_arena import logger

import unittest
import random

import picologging as logging

from pkmn_rl_arena import (
    ROM_PATH,
    BIOS_PATH,
    MAP_PATH,
    POKEMON_CSV_PATH,
    MOVES_CSV_PATH,
)


class TestSaveState(unittest.TestCase):
    #############################################
    # TEST CASES
    save_load_cases = [
        ("state_1", random.randint(0, 20)),
        ("state_2", random.randint(0, 20)),
        ("state_3", random.randint(0, 20)),
    ]

    def setUp(self):
        logger.setLevel(logging.DEBUG)

        self.core = BattleCore(ROM_PATH, BIOS_PATH, MAP_PATH)
        turn = self.core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.CREATE_TEAM)

        self.save_manager = SaveStateManager(self.core)
        self.action_manager = ActionManager(self.core)

        team_factory = PkmnTeamFactory()
        teams = {
            "player": team_factory.create_random_team(),
            "enemy": team_factory.create_random_team(),
        }
        self.core.write_team_data(teams)
        turn = self.core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)

    def tearDown(self):
        self.save_manager.remove_save_states()
        return

    def advance_turn(self):
        actions = {}
        actions = {
            agent: random.choice(self.action_manager.get_valid_action_ids(agent))
            for agent in self.core.get_required_agents()
        }

        for agent, action in actions.items():
            self.core.write_action(agent, action)

    def test_save_load_state(self):
        for save_name, nb_turn_to_advance in TestSaveState.save_load_cases:
            logger.debug(
                f"test : {save_name}, nb turn to advance : {nb_turn_to_advance}"
            )

            # skip turns with random moves to simulate battle
            for i in range(nb_turn_to_advance):
                self.advance_turn()

            self.save_manager.save_state(save_name)
            battle_state = self.core.state
            self.advance_turn()
            self.save_manager.load_state(save_name)
            self.assertEqual(battle_state, self.core.state)
