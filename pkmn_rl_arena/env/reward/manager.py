from collections.abc import Callable

import numpy as np
from pkmn_rl_arena.env.observation import Observation

from .functions import reward_functions


class RewardManager:
    def __init__(
        self,
        reward_func: Callable[[str, list[Observation]], float] = reward_functions[0],
        previous_observations: list[Observation] = [],
        clip_range=(-10.0, 10.0)
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
        self.clip_range = clip_range
        self.reward_stats = {"sum": 0.0, "count": 0}

    def reset(self, to_prev_obs=False):
        self.obs = []
        return

    def add_observation(self, obs: Observation):
        """Add a new observation to observation list"""
        self.obs.append(obs)

    def compute_reward(self, agent):
        raw_reward = self.reward_func(agent, self.obs)
        
        # Clip pour éviter explosions
        clipped_reward = np.clip(raw_reward, *self.clip_range)
        
        # Track stats pour monitoring
        self.reward_stats["sum"] += clipped_reward
        self.reward_stats["count"] += 1
        
        return clipped_reward
