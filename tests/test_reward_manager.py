from pkmn_rl_arena import log
from pkmn_rl_arena.env.action import ActionManager
from pkmn_rl_arena.env.reward.manager import RewardManager
from pkmn_rl_arena.env.reward.functions import reward_functions
from pkmn_rl_arena.env.observation import ObservationFactory
from pkmn_rl_arena.env.battle_core import BattleCore, CoreContext
from pkmn_rl_arena.paths import PATHS

from pkmn_rl_arena.env.pkmn_team_factory import PkmnTeamFactory

import unittest
import picologging as logging

from pkmn_rl_arena.test_utils import advance_turn

class TestRewardManager(unittest.TestCase):
    def setUp(self):
        log.setLevel(logging.DEBUG)

        # create managers
        self.core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.ctxt = CoreContext(self.core)
        self.action_manager = ActionManager(self.ctxt)
        self.obs_factory = ObservationFactory(self.ctxt)

        self.previous_obs = []

        # create teams and go to turntype general
        self.core.advance_to_next_turn()

        self.pkmn_factory = PkmnTeamFactory(PATHS["POKEMON_CSV"], PATHS["MOVES_CSV"])
        self.core.write_team_data(
            {
                agent: self.pkmn_factory.create_random_team()
                for agent in ["player", "enemy"]
            }
        )
        self.core.advance_to_next_turn()
        self.previous_obs.append(self.obs_factory.from_game())
        self.agents = ["player", "enemy"]

    def test_constructor_default_arg(self):
        rem = RewardManager()
        self.assertEqual(
            len(rem.obs),
            0,
            f"Length of reward manager obs should be 0, as previous_obs constructor arg is empty got {len(rem.obs)}.",
        )
        self.assertEqual(
            len(self.previous_obs),
            1,
            f"Length of previous obs should be 1, got {len(self.previous_obs)}.",
        )
        for agent in self.agents:
            self.assertEqual(
                rem.compute_reward(agent),
                0.0,
                "Default reward function should return 0.0 if there are less than 2 rewards. Maybe reward function was not set properly.",
            )

        advance_turn(self.core, self.action_manager)
        rem.add_observation(self.obs_factory.from_game())
        advance_turn(self.core, self.action_manager)
        rem.add_observation(self.obs_factory.from_game())

        for agent in self.agents:
            self.assertEqual(
                rem.compute_reward(agent),
                reward_functions[0](agent, rem.obs),
                f"Reward manager reward function not set properly as RewardManager.compute_reward() doesn't return the right value. Expected {reward_functions[0](agent, rem.obs)} got {rem.compute_reward(agent)}",
            )
