from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.battle_state import TurnType
from pkmn_rl_arena.env.save_state import SaveStateManager
from pkmn_rl_arena.env.action import ActionManager
from pkmn_rl_arena.env.pkmn_team_factory import PkmnTeamFactory
from pkmn_rl_arena import log

import unittest
import random

import picologging as logging

from pkmn_rl_arena.paths import PATHS

from pkmn_rl_arena.test_utils import advance_turn


class TestSaveState(unittest.TestCase):
    #############################################
    # TEST CASES
    save_load_cases = [
        ("state_1", random.randint(0, 20)),
        ("state_2", random.randint(0, 20)),
        ("state_3", random.randint(0, 20)),
    ]

    def setUp(self):
        log.setLevel(logging.DEBUG)

        self.core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
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

    def test_save_load_state(self):
        for save_name, nb_turn_to_advance in TestSaveState.save_load_cases:
            log.debug(f"test : {save_name}, nb turn to advance : {nb_turn_to_advance}")

            # skip turns with random moves to simulate battle
            for i in range(nb_turn_to_advance):
                advance_turn(self.core, self.action_manager)

            self.save_manager.save_state(save_name)
            battle_state = self.core.state
            advance_turn(self.core, self.action_manager)
            self.save_manager.load_state(save_name)
            self.assertEqual(battle_state, self.core.state)
