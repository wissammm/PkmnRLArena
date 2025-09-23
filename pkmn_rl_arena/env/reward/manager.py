from collections.abc import Callable
from pkmn_rl_arena.env.observation import Observation

from .functions import reward_functions


class RewardManager:
    def __init__(
        self,
        reward_func: Callable[[str, list[Observation]], float] = reward_functions[0],
        previous_observations: list[Observation] = [],
    ):
        """
        RewardManager constructor
        Takes as input a function that will serve to compute the reward.
        New function additions must be stored within reward_funcs module.

        Args :
            prev_obs : observations to take in count prior to current obs
            reward_func : reward function, must be able to compute a reward for
                - end of step
                - end of episode
        """
        self.prev_obs = previous_observations
        self.obs = previous_observations
        self.reward_func = reward_func

    def reset(self, to_prev_obs=False):
        self.obs = []
        return

    def add_observation(self, obs: Observation):
        """Add a new observation to observation list"""
        self.obs.append(obs)

    def compute_reward(self, agent):
        return self.reward_func(agent, self.obs)
