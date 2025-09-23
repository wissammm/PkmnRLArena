from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.battle_state import TurnType, BattleStateFactory, BattleState

from pkmn_rl_arena.env.save_state import SaveStateManager
from pkmn_rl_arena.env.action import ActionManager
from pkmn_rl_arena.env.pkmn_team_factory import PkmnTeamFactory
from pkmn_rl_arena import log

import copy
import random
import unittest

import picologging as logging

from pkmn_rl_arena.paths import PATHS

from pkmn_rl_arena.test_utils import advance_turn


class TestSaveState(unittest.TestCase):
    #############################################
    # TEST CASES
    save_load_cases = [
        ("state_1", random.randint(1, 20)),
        ("state_2", random.randint(1, 20)),
        ("state_3", random.randint(1, 20)),
    ]

    def setUp(self):
        log.setLevel(logging.DEBUG)

        self.core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.assertEqual(self.core.state.turn, TurnType.CREATE_TEAM)

        self.save_manager = SaveStateManager(self.core)
        self.action_manager = ActionManager(self.core)

        team_factory = PkmnTeamFactory()
        teams = {
            "player": team_factory.create_random_team(),
            "enemy": team_factory.create_random_team(),
        }

        self.core.write_team_data(teams)
        self.core.advance_to_next_turn(count_step=False)
        log.debug(f"state : {self.core.state}")
        self.assertEqual(self.core.state.turn, TurnType.GENERAL)

    def tearDown(self):
        self.save_manager.remove_save_states()
        return

    def test_save_load_state(self):
        total_step_run = 0

        for save_name, nb_turn_to_advance in TestSaveState.save_load_cases:
            log.debug(f"test : {save_name}, nb turn to advance : {nb_turn_to_advance}")

            ground_truth = copy.deepcopy(self.core.state)

            save_path = self.save_manager.save_state(save_name)
            save_path_state = BattleStateFactory.from_save_path(save_path)

            self.assertEqual(ground_truth, save_path_state)

            # skip turns with random moves to simulate battle
            for i in range(nb_turn_to_advance):
                advance_turn(self.core, self.action_manager)

            self.assertEqual(self.core.state.step, nb_turn_to_advance + total_step_run)

            returned_state = self.save_manager.load_state(save_name)

            self.assertEqual(ground_truth, returned_state)
            self.assertEqual(ground_truth, self.core.state)

            total_step_run += self.core.state.step
