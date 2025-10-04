from pkmn_rl_arena.env.observation import Observation

from collections.abc import Callable
from typing import Dict


def default_reward(agent: str, observations: list[Observation]) -> float:
    """
    Compute reward for the specified agent based on changes between observations

    Args:
        agent: The agent to compute reward for ('player' or 'enemy')

    Returns:
        float: The calculated reward
    """
    reward_coeffs: Dict[str, float] = {
        "hp_loss": -0.01,
        "hp_healed": 0.01,
        "hp_damage": 0.02,
        "status_effect": 0.5,
        "stat_boost": 0.2,
        "stat_decrease": 0.1,
        "ko_enemy": 1.0,
        "ko_self": -1.0,
        "victory": 5.0,
        "defeat": -5.0,
    }

    if len(observations) <= 1:
        return 0.0

    current_obs = observations[-1]
    prev_obs = observations[-2]

    opponent = "enemy" if agent == "player" else "player"

    total_reward = 0.0

    prev_hp = prev_obs.hp()
    curr_hp = current_obs.hp()
    prev_ko = prev_obs.pkmn_ko()
    curr_ko = current_obs.pkmn_ko()
    prev_stats = prev_obs.stats()
    curr_stats = current_obs.stats()

    NB_DATA_PKMN = 60 
    STATUS_IDX = 17
    STATS_BEGIN = 2
    STATS_END = 7

    for i in range(len(prev_hp[agent])):
        hp_change = curr_hp[agent][i] - prev_hp[agent][i]
        if hp_change < 0: 
            total_reward += reward_coeffs["hp_loss"] * hp_change  # Negative
        elif hp_change > 0:  
            total_reward += reward_coeffs["hp_healed"] * hp_change  # Positive

    for i in range(len(prev_hp[opponent])):
        hp_change = curr_hp[opponent][i] - prev_hp[opponent][i]
        if hp_change < 0:  
            total_reward += reward_coeffs["hp_damage"] * abs(hp_change)

    for i in range(len(curr_ko[agent])):
        if curr_ko[agent][i] and not prev_ko[agent][i]:
            total_reward += reward_coeffs["ko_self"]  
    for i in range(len(curr_ko[opponent])):
        if curr_ko[opponent][i] and not prev_ko[opponent][i]:
            total_reward += reward_coeffs["ko_enemy"] 

    for i in range(len(prev_stats[agent])):
        for stat_idx in range(5): 
            stat_change = curr_stats[agent][i][stat_idx] - prev_stats[agent][i][stat_idx]
            if stat_change > 0:
                total_reward += reward_coeffs["stat_boost"]
            elif stat_change < 0:
                total_reward -= reward_coeffs["stat_decrease"] 
    for i in range(len(prev_stats[opponent])):
        for stat_idx in range(5):
            stat_change = curr_stats[opponent][i][stat_idx] - prev_stats[opponent][i][stat_idx]
            if stat_change > 0:
                total_reward -= reward_coeffs["stat_boost"]  # Penalty if opponent boosts
            elif stat_change < 0:
                total_reward += reward_coeffs["stat_decrease"]  # Reward if opponent decreases

    for i in range(6):  
        agent_status_prev = prev_obs._o[agent][i * NB_DATA_PKMN + STATUS_IDX]
        agent_status_curr = current_obs._o[agent][i * NB_DATA_PKMN + STATUS_IDX]
        if agent_status_prev == 0 and agent_status_curr != 0:
            total_reward -= reward_coeffs["status_effect"]  

        opp_status_prev = prev_obs._o[opponent][i * NB_DATA_PKMN + STATUS_IDX]
        opp_status_curr = current_obs._o[opponent][i * NB_DATA_PKMN + STATUS_IDX]
        if opp_status_prev == 0 and opp_status_curr != 0:
            total_reward += reward_coeffs["status_effect"]

    if all(curr_ko[opponent]):
        total_reward += reward_coeffs["victory"]
    if all(curr_ko[agent]):
        total_reward += reward_coeffs["defeat"]

    return total_reward


reward_functions: list[Callable[[str, list[Observation]], float]] = [default_reward]