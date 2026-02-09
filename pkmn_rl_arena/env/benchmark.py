import sys
import os
import random
import time
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena.env.battle_arena_aec import BattleArenaAEC, BattleCore

STEPS = 50


class Benchmark:
    def __init__(self):
        core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.env = BattleArenaAEC(core)
        self.env.reset()

    def run(self):
        step_times = []
        last_time = time.time()
        
        for step in range(STEPS):
            # Get the current agent that needs to act
            current_agent = self.env.agent_selection
            
            # Get legal actions for current agent
            action_mask = self.env.action_manager.get_action_mask(current_agent)
            legal_actions = [i for i, valid in enumerate(action_mask) if valid > 0]
            
            if not legal_actions:
                print(f"No legal actions available for {current_agent}")
                break
            
            # Choose random legal action
            action = random.choice(legal_actions)
            
            # Step the environment
            self.env.step(action)
            
            now = time.time()
            step_time = now - last_time
            step_times.append(step_time)
            last_time = now

            # Check if episode is done
            if self.env.terminations[current_agent] or self.env.truncations[current_agent]:
                print(f"Episode finished at step {step}!")
                break

        if step_times:
            avg_time = sum(step_times) / len(step_times)
            total_time = sum(step_times)
            print(f"\n=== Benchmark Results ===")
            print(f"Total steps: {len(step_times)}")
            print(f"Total time: {total_time:.4f} seconds")
            print(f"Average step time: {avg_time:.4f} seconds")
            print(f"Steps per second: {1/avg_time:.2f}")


if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.run()
