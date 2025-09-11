from pkmn_rl_arena import (
    ROM_PATH,
    BIOS_PATH,
    MAP_PATH,
)
from pkmn_rl_arena import logger
from pkmn_rl_arena.env.battle_core import BattleCore, TurnType
from pkmn_rl_arena.env.battle_arena import BattleArena
from pkmn_rl_arena.env.pkmn_team_factory import DataSize
from pkmn_rl_arena.data import pokemon_data

from pettingzoo.test import parallel_api_test

import unittest
import picologging as logging

from typing import Dict


class TestArena(unittest.TestCase):
    def setUp(self):
        logger.setLevel(logging.DEBUG)
        core = BattleCore(ROM_PATH, BIOS_PATH, MAP_PATH)
        self.arena = BattleArena(core)
        self.required_agents = self.arena.battle_state.get_required_agents()

    def tearDown(self):
        BattleArena.close()

    def test_step(self):
        for agent in self.arena.battle_state.get_required_agents():
            observation, reward, termination, truncation, info = self.arena.last()

            if termination or truncation:
                action = None
            else:
                # insert neural network here to choose action from observation space
                #
                # until then here is a dummy fctn
                action[agent] = self.arena.action_space(agent).sample()

        self.arena.step(action)

    def test_reset(self):
        required_agents = self.arena.battle_state.get_required_agents()
        action: Dict[str, int] | None = {} if required_agents == [] else None

        for agent in required_agents:
            observation, reward, termination, truncation, info = self.arena.last()

            if not (termination or truncation):
                action = None
                break
            else:
                # insert neural network here to choose action from observation space
                #
                # until then here is a dummy fctn
                action[agent] = self.arena.action_space(agent).sample()

        self.arena.step(action)
        self.arena.reset()

    # def test_load_save_state():
    #     pass

    # def test_render(self):
    #     self.arena.reset(seed=42)
    #     self.arena.render()




if __name__ == "__main__":
    suite = unittest.TestSuite()
