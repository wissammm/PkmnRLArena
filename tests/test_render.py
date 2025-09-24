import pkmn_rl_arena
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena import log
from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.test_utils import advance_turn
from pkmn_rl_arena.env.battle_arena import BattleArena, RenderMode
from pkmn_rl_arena.env.pkmn_team_factory import DataSize
from pkmn_rl_arena.data import pokemon_data
from pkmn_rl_arena.env.turn_type import TurnType


from pettingzoo.test import parallel_api_test

import unittest
import picologging as logging

import random


class TestArena(unittest.TestCase):
    def setUp(self):
        log.setLevel(logging.DEBUG)
        core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.arena = BattleArena(core, render_mode=RenderMode.TURN_TERMINAL)

    def tearDown(self):
        self.arena.close()

    def test_render(self):
        options = {
            "save_state": None,
            "teams": {
                "player": [
                    129,  # Magikarp lvl 1 with splash wich does nothing
                    1,  # lvl 1
                    150,  # splash
                    150,
                    150,
                    150,
                    100,  # 100% hp
                    0,
                ],
                "enemy": [
                    # Squirtle
                    7,
                    99,  # lvl 99
                    111,  # DEFENSE CURL
                    0,
                    0,
                    0,
                    10,  # 10 % hp
                    0,
                    # WARTORTLE
                    8,
                    99,
                    5,  # MEGAPUNCH
                    5,  # MEGAPUNCH
                    5,  # MEGAPUNCH
                    5,  # MEGAPUNCH
                    11,
                    0,
                ],
            },
        }

        observations, infos = self.arena.reset(seed=None, options=options)

        for i in range(5):
            actions = {
                agent: random.choice(
                    self.arena.action_manager.get_valid_action_ids(agent)
                )
                for agent in self.arena.core.get_required_agents()
            }

            observations, rewards, terminations, truncations, infos = self.arena.step(
                actions
            )
