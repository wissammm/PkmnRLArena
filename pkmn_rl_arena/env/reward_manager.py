from dataclasses import dataclass
from typing import Optional

import numpy as np

from pkmn_rl_arena.env.observation import Observation, ObsIdx, ObservationFactory


@dataclass
class RewardCoeff:
    hp_loss: float = -0.01
    hp_healed: float = -0.01
    hp_damage: float = 0.02
    status_effect: float = 0.5    
    stat_boost: float = 0.2        
    stat_decrease: float = 0.1     
    ko_enemy: float = 1.0         
    ko_self: float = -1.0         
    victory: float = 5.0          
    defeat: float = -5.0          


class RewardManager:
    def __init__(self, coeffs: RewardCoeff = None):
        self.coeffs = coeffs if coeffs is not None else RewardCoeff()
        self.prev_obs: Optional[Observation] = None
        self.current_obs: Optional[Observation] = None
    
    def update_observation(self, obs: Observation):
        """Add a new observation and update state"""
        self.prev_obs = self.current_obs
        self.current_obs = obs

    def compute_reward(self, agent: str) -> float:
        """
        Compute reward for the specified agent based on changes between observations
        
        Args:
            agent: The agent to compute reward for ('player' or 'enemy')
        
        Returns:
            float: The calculated reward
        """
        if self.prev_obs is None or self.current_obs is None:
            return 0.0
        
        opponent = "enemy" if agent == "player" else "player"
        
        total_reward = 0.0
        
        prev_hp = self.prev_obs.hp()
        curr_hp = self.current_obs.hp()
        
        # Calculate HP loss (negative HP change for agent)
        for i in range(len(prev_hp[agent])):
            hp_change = curr_hp[agent][i] - prev_hp[agent][i]
            if hp_change < 0:  # Lost HP
                total_reward += self.coeffs.hp_loss * hp_change  # Negative reward for HP loss
        
        for i in range(len(prev_hp[opponent])):
            hp_change = curr_hp[opponent][i] - prev_hp[opponent][i]
            if hp_change < 0: 
                total_reward += self.coeffs.hp_damage * abs(hp_change) 
        
        curr_ko = self.current_obs.pkmn_ko()
        
      
        if all(curr_ko[opponent]): 
            total_reward += self.coeffs.victory
        
        if all(curr_ko[agent]):  
            total_reward += self.coeffs.defeat
        
        return total_reward
