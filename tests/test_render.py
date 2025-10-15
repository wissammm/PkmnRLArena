from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena import log
from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.turn_type import TurnType
from pkmn_rl_arena.env.battle_arena import BattleArena, RenderMode
from pkmn_rl_arena.test_utils import init_rng

import picologging as logging

import random
from time import sleep
import unittest

class TestRendering(unittest.TestCase):
    def setUp(self):
        log.setLevel(logging.DEBUG)
        seed_used = init_rng()
        log.debug(f"Current random seed : {seed_used}")
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
                    # Bulbasaur
                    1,
                    15,
                    14,
                    34,
                    38,
                    102,
                    50,
                    0,
                    # Snorlax
                    143,
                    60,
                    118,
                    157,
                    164,
                    223,
                    95,
                    0,
                    # Ditto
                    132,
                    100,
                    144,
                    0,
                    0,
                    0,
                    100,
                    0,
                    # METAGROSS
                    400,
                    164,
                    223,
                    205,
                    244,
                    207,
                    60,
                    0,
                    # Rayquaza
                    406,
                    2,
                    157,
                    164,
                    244,
                    173,
                    1,
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
                    # TYPHLOSION
                    157,
                    40,
                    52,
                    98,
                    172,
                    53,
                    0,
                    178,
                ],
            },
        }

        observations, infos = self.arena.reset(seed=None, options=options)
        self.assertEqual(self.arena.core.state.turn, TurnType.GENERAL)

        for i in range(80):
            actions = {
                agent: random.choice(
                    self.arena.action_manager.get_valid_action_ids(agent)
                )
                for agent in self.arena.core.get_required_agents()
            }

            observations, rewards, terminations, truncations, infos = self.arena.step(
                actions
            )
            sleep(0.2)
            if self.arena.core.state.turn == TurnType.DONE:
                break
