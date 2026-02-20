from dataclasses import dataclass

@dataclass
class EnvConfig:
    TEAM_BATCH_SIZE: int = 60
    WIN_RATE_THRESHOLD: float = 0.70
    CURRICULUM_CHECK_INTERVAL: int = 700
    MIN_TEAM_SIZE: int = 1
    MAX_TEAM_SIZE: int = 2

@dataclass
class ResourceConfig:
    NUM_GPUS: int = 1
    NUM_ENV_RUNNERS: int = 7
    NUM_CPUS_PER_WORKER: int = 1
    NUM_ENVS_PER_WORKER: int = 3

@dataclass
class PPOHyperparams:
    ROLLOUT_FRAGMENT_LENGTH: str = 'auto' 
    TRAIN_BATCH_SIZE: int = 2048
    SGD_MINIBATCH_SIZE: int = 512
    NUM_SGD_ITER: int = 6
    LR: float = 3e-4
    GAMMA: float = 0.99
    ENTROPY_COEFF: float = 0.02
    VFCOEFF: float = 1.0
    CLIP_PARAM: float = 0.2

@dataclass
class RunConfig:
    STOP_ITERATIONS: int = 1000
    CHECKPOINT_FREQ: int = 20
    STORAGE_PATH_SUFFIX: str = "ray_results"
    EXP_NAME: str = "PPO_Pokemon_2v2_Overnight"

@dataclass
class LeagueConfig:
    SNAPSHOT_INTERVAL: int = 25
    PROB_RANDOM_BASELINE_NO_SNAPSHOT: float = 0.70
    PROB_SELFPLAY_NO_SNAPSHOT: float = 0.30
    PROB_RANDOM_BASELINE: float = 0.20
    PROB_SELFPLAY: float = 0.35
    PROB_RECENT: float = 0.30
    PROB_HISTORICAL: float = 0.15
    RECENT_WINDOW: int = 5
    ELO_MAIN_INIT: float = 1500.0
    ELO_RANDOM_INIT: float = 1000.0
    ELO_K: float = 32.0
    ELO_MIN: float = 0.0
    ELO_MAX: float = 3000.0

class TrainingConfig:
    ENV = EnvConfig()
    RESOURCES = ResourceConfig()
    PPO = PPOHyperparams()
    RUN = RunConfig()
    LEAGUE = LeagueConfig()