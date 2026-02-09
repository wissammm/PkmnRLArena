import os
import warnings
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from pkmn_rl_arena.env.battle_arena_aec import BattleArenaAEC, BattleCore
from pkmn_rl_arena.env.pkmn_team_factory import PkmnTeamFactory
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena.training.wrappers.aec_wrappers import CurriculumWrapperAEC, TeamBatchWrapperAEC
from pkmn_rl_arena.training.models.pkmn_model import PokemonTransformerModel
from pkmn_rl_arena.training.league import LeagueManager, RANDOM_POLICY_ID
from pkmn_rl_arena.training.league.random_policy import MaskedRandomPolicy
from pkmn_rl_arena.training.config import TrainingConfig

# 1. Register Custom Model
ModelCatalog.register_custom_model("pkmn_transformer", PokemonTransformerModel)

# 2. League (global instance shared by callbacks and policy mapping)
LEAGUE = LeagueManager(snapshot_interval=50)


# 3. Environment factory ─ AEC pipeline
def env_creator(config):
    core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
    team_factory = PkmnTeamFactory(PATHS["POKEMON_CSV"], PATHS["MOVES_CSV"])

    # Base AEC env
    env = BattleArenaAEC(core)

    # Wrap: pre-generated team batches (AEC-compatible)
    env = TeamBatchWrapperAEC(
        env, team_factory, batch_size=TrainingConfig.ENV.TEAM_BATCH_SIZE
    )

    # Wrap: curriculum progression (AEC-compatible)
    env = CurriculumWrapperAEC(
        env,
        win_rate_threshold=TrainingConfig.ENV.WIN_RATE_THRESHOLD,
        min_size=TrainingConfig.ENV.MIN_TEAM_SIZE,
        max_size=TrainingConfig.ENV.MAX_TEAM_SIZE,
        check_interval=TrainingConfig.ENV.CURRICULUM_CHECK_INTERVAL,
    )

    # RLlib's built-in AEC → MultiAgentEnv adapter
    return PettingZooEnv(env)


register_env("pkmn_battle_env", env_creator)


# --- CALLBACKS ---

class LeagueCallbacks(DefaultCallbacks):
    """Handles TensorBoard metrics, per-opponent win rates, ELO, and league snapshots."""

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        last_info = episode.last_info_for("player") or {}

        if "team_size" in last_info:
            episode.custom_metrics["curriculum_team_size"] = last_info["team_size"]

        total_reward = episode.agent_rewards.get(("player", "main_policy"), 0)
        is_win = 1 if total_reward > 0 else 0

        episode.custom_metrics["is_win"] = is_win
        episode.custom_metrics["episode_length"] = episode.length

        # --- Per-opponent win rate (the real progress metric) ---
        enemy_policy = episode.policy_for("enemy")
        episode.custom_metrics[f"win_vs_{enemy_policy}"] = is_win

        # --- ELO update ---
        if is_win:
            LEAGUE.update_elo("main_policy", enemy_policy)
        else:
            LEAGUE.update_elo(enemy_policy, "main_policy")

        episode.custom_metrics["elo_main_policy"] = LEAGUE.elo_ratings.get("main_policy", 1500.0)
        episode.custom_metrics["elo_vs_random_gap"] = (
            LEAGUE.elo_ratings.get("main_policy", 1500.0)
            - LEAGUE.elo_ratings.get(RANDOM_POLICY_ID, 1000.0)
        )

    def on_train_result(self, *, algorithm, result, **kwargs):
        iteration = result["training_iteration"]

        if LEAGUE.should_snapshot(iteration):
            new_policy_id = LEAGUE.get_snapshot_id(iteration)
            print(f"--- LEAGUE: Snapshotting main_policy → {new_policy_id} ---")

            main_weights = algorithm.get_weights(["main_policy"])

            algorithm.add_policy(
                policy_id=new_policy_id,
                policy_cls=type(algorithm.get_policy("main_policy")),
                policy_mapping_fn=None,
            )

            algorithm.set_weights({new_policy_id: main_weights["main_policy"]})
            LEAGUE.add_policy(new_policy_id)


# 4. Policy Mapping Function
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "player":
        return "main_policy"
    return LEAGUE.get_opponent()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*_get_slice_indices.*has been deprecated.*")
    ray.init()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    config = (
        PPOConfig()
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
        .training(
            train_batch_size=TrainingConfig.PPO.TRAIN_BATCH_SIZE,
            lr=TrainingConfig.PPO.LR,
            gamma=TrainingConfig.PPO.GAMMA,
        )
        .multi_agent(
            policies={
                "main_policy": (
                    None,  # use default PPO policy class
                    None,  # infer obs space
                    None,  # infer act space
                    {
                        "model": {
                            "custom_model": "pkmn_transformer",
                            "custom_model_config": {},
                        },
                    },
                ),
                RANDOM_POLICY_ID: (
                    MaskedRandomPolicy,  # custom policy class that respects masks
                    None,  # infer obs space
                    None,  # infer act space
                    {
                        "model": {
                            "_disable_preprocessor_api": True,
                        },
                    },
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main_policy"],  # never train random
        )
    )

    config.sgd_minibatch_size = TrainingConfig.PPO.SGD_MINIBATCH_SIZE
    config.num_sgd_iter = TrainingConfig.PPO.NUM_SGD_ITER

    tune.run(
        "PPO",
        name=TrainingConfig.RUN.EXP_NAME,
        config=config.to_dict(),
        stop={"training_iteration": TrainingConfig.RUN.STOP_ITERATIONS},
        checkpoint_freq=TrainingConfig.RUN.CHECKPOINT_FREQ,
        storage_path=os.path.join(project_root, TrainingConfig.RUN.STORAGE_PATH_SUFFIX),
    )