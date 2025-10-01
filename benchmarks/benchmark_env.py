import pytest
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.battle_arena import BattleArena
import random

@pytest.fixture
def arena():
    core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
    env = BattleArena(core)
    env.reset()
    yield env
    env.close()

@pytest.mark.parametrize("step", range(20))
def test_env_step_benchmark_env_step(benchmark, arena, step):
    def run_step():
        actions = {
            agent: random.choice(arena.action_manager.get_valid_action_ids(agent))
            for agent in arena.core.get_required_agents()
        }
        arena.step(actions)
    
    benchmark(run_step)
    print(f"Time for step {step + 1}: {benchmark.stats['mean']:.8f} seconds")
