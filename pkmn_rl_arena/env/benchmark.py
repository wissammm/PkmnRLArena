import sys
import os
import random
import time
from pkmn_rl_arena.env.core import PokemonRLCore
from pkmn_rl_arena.env.BattleState import TurnType
import rustboyadvance_py
import pkmn_rl_arena.data.parser
import pkmn_rl_arena.data.pokemon_data

STEPS = 50
from pkmn_rl_arena.paths import PATHS

class Benchmark:
    def __init__(self):
        self.rl_core = PokemonRLCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        obs = self.rl_core.reset()

    def run(self):
        step_times = []
        last_time = time.time()
        for step in range(STEPS):
            current_turn = self.rl_core.get_current_turn_type()
            required_agents = self.rl_core.get_required_agents()
            
            actions = {}
            for agent in required_agents:
                legal_actions = self.rl_core.action_manager.get_legal_actions(agent)
                if legal_actions:
                    actions[agent] = random.choice(legal_actions)  # Choose a random legal action
                else:
                    print(f"No legal actions available for {agent}")
            
            observations, rewards, done, info = self.rl_core.step(actions)
            
            now = time.time()
            step_time = now - last_time
            step_times.append(step_time)
            last_time = now

            if done:
                print("Episode finished!")
                break

        if step_times:
            avg_time = sum(step_times) / len(step_times)
            print(f"Average step time: {avg_time:.4f} seconds")

if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.run()
