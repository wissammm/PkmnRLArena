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
        "hp_healed": -0.01,
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

    # Calculate HP loss (negative HP change for agent)
    for i in range(len(prev_hp[agent])):
        hp_change = curr_hp[agent][i] - prev_hp[agent][i]
        if hp_change < 0:  # Lost HP
            total_reward += (
                reward_coeffs["hp_loss"] * hp_change
            )  # Negative reward for HP loss

    for i in range(len(prev_hp[opponent])):
        hp_change = curr_hp[opponent][i] - prev_hp[opponent][i]
        if hp_change < 0:
            total_reward += reward_coeffs["hp_damage"] * abs(hp_change)

    curr_ko = current_obs.pkmn_ko()

    if all(curr_ko[opponent]):
        total_reward += reward_coeffs["victory"]

    if all(curr_ko[agent]):
        total_reward += reward_coeffs["defeat"]

    return total_reward


reward_functions: list[Callable[[str, list[Observation]], float]] = [default_reward]
