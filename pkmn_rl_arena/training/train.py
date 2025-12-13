import os
import random
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import Policy

from pkmn_rl_arena.env.battle_arena import BattleArena, BattleCore
from pkmn_rl_arena.env.pkmn_team_factory import PkmnTeamFactory
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena.training.wrappers.curriculum_wrappers import CurriculumWrapper, TeamBatchWrapper
from pkmn_rl_arena.training.models.pkmn_model import PokemonTransformerModel
from pkmn_rl_arena.training.config import TrainingConfig

# 1. Register Custom Model
ModelCatalog.register_custom_model("pkmn_transformer", PokemonTransformerModel)

# 2. Define Env Creator
def env_creator(config):
    core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
    env = BattleArena(core)
    team_factory = PkmnTeamFactory(PATHS["POKEMON_CSV"], PATHS["MOVES_CSV"])
    
    env = TeamBatchWrapper(env, team_factory, batch_size=TrainingConfig.ENV.TEAM_BATCH_SIZE)
    env = CurriculumWrapper(
        env, 
        win_rate_threshold=TrainingConfig.ENV.WIN_RATE_THRESHOLD,
        min_size=TrainingConfig.ENV.MIN_TEAM_SIZE,
        max_size=TrainingConfig.ENV.MAX_TEAM_SIZE,
        check_interval=TrainingConfig.ENV.CURRICULUM_CHECK_INTERVAL
    )
    return PettingZooEnv(env)

register_env("pkmn_battle_env", env_creator)

# --- LEAGUE & METRICS MANAGER ---

class LeagueManager:
    def __init__(self):
        self.past_policies = []

    def add_policy(self, policy_id):
        self.past_policies.append(policy_id)

    def get_opponent(self):
        # 80% chance to play against current self (main_policy)
        # 20% chance to play against past version (if any exist)
        if len(self.past_policies) > 0 and random.random() < 0.2:
            return random.choice(self.past_policies)
        return "main_policy"

LEAGUE = LeagueManager()

class LeagueCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # --- TENSORBOARD LOGGING ---
        last_info = episode.last_info_for("player") or {}
        
        if "team_size" in last_info:
            episode.custom_metrics["curriculum_team_size"] = last_info["team_size"]
        
        total_reward = episode.agent_rewards.get(("player", "main_policy"), 0)
        episode.custom_metrics["is_win"] = 1 if total_reward > 0 else 0

    def on_train_result(self, *, algorithm, result, **kwargs):
        # --- LEAGUE SNAPSHOTTING ---
        iteration = result["training_iteration"]
        
        if iteration > 0 and iteration % 50 == 0:
            new_policy_id = f"policy_v{iteration}"
            print(f"--- LEAGUE: Snapshotting main_policy to {new_policy_id} ---")
            
            main_weights = algorithm.get_weights("main_policy")
            
            new_policy = algorithm.add_policy(
                policy_id=new_policy_id,
                policy_cls=type(algorithm.get_policy("main_policy")),
                policy_mapping_fn=None,
            )
            
            algorithm.set_weights({new_policy_id: main_weights})
            LEAGUE.add_policy(new_policy_id)
            
            policies_to_train = result["config"]["policies_to_train"]
            if new_policy_id in policies_to_train:
                policies_to_train.remove(new_policy_id)

# 3. Policy Mapping Function
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "player":
        return "main_policy"
    return LEAGUE.get_opponent()

if __name__ == "__main__":
    ray.init()

    # Calculate project root dynamically to avoid KeyError
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # Initialize Config
    config = (
        PPOConfig()
        # --- CRITICAL FIX: Disable new API stack to support custom ModelV2 ---
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment("pkmn_battle_env")
        .framework("torch")
        .callbacks(LeagueCallbacks)
        .resources(
            num_gpus=TrainingConfig.RESOURCES.NUM_GPUS,
        )
        .env_runners(
            num_env_runners=TrainingConfig.RESOURCES.NUM_ENV_RUNNERS,
            num_envs_per_env_runner=TrainingConfig.RESOURCES.NUM_ENVS_PER_WORKER,
            num_cpus_per_env_runner=TrainingConfig.RESOURCES.NUM_CPUS_PER_WORKER,
            rollout_fragment_length=TrainingConfig.PPO.ROLLOUT_FRAGMENT_LENGTH,
        )
        # Generic Training Args
        .training(
            train_batch_size=TrainingConfig.PPO.TRAIN_BATCH_SIZE,
            lr=TrainingConfig.PPO.LR,
            gamma=TrainingConfig.PPO.GAMMA,
            model={
                "custom_model": "pkmn_transformer",
                "custom_model_config": {},
            },
        )
        .multi_agent(
            policies={"main_policy"}, 
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main_policy"],
        )
    )

    # Set PPO Specific Args directly
    config.sgd_minibatch_size = TrainingConfig.PPO.SGD_MINIBATCH_SIZE
    config.num_sgd_iter = TrainingConfig.PPO.NUM_SGD_ITER

    tune.run(
        "PPO",
        name=TrainingConfig.RUN.EXP_NAME,
        config=config.to_dict(),
        stop={"training_iteration": TrainingConfig.RUN.STOP_ITERATIONS},
        checkpoint_freq=TrainingConfig.RUN.CHECKPOINT_FREQ,
        storage_path=os.path.join(project_root, TrainingConfig.RUN.STORAGE_PATH_SUFFIX)
    )