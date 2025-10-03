import pytest
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.battle_arena import BattleArena
import random

@pytest.fixture
def core():
    return BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])

@pytest.fixture
def env(core):
    arena = BattleArena(core)
    arena.reset()
    yield arena
    arena.close()

def test_gba_step(benchmark, core):
    def run_gba_step():
        id = core.run_to_next_stop()
        if id == 4:
            raise StopIteration("Reached id == 4, stopping benchmark.")
        core.clear_stop_condition_id(id)
    benchmark(run_gba_step)
    print(f"Time for GBA step: {benchmark.stats['mean']:.8f} seconds")
    
def test_env_step(benchmark, env):
    def run_env_step():
        actions = {
            agent: random.choice(env.action_manager.get_valid_action_ids(agent))
            for agent in env.core.get_required_agents()
        }
        env.step(actions)
    benchmark(run_env_step)
    print(f"Time for env step: {benchmark.stats['mean']:.8f} seconds")