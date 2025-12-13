from dataclasses import dataclass

@dataclass
class EnvConfig:
    """Configuration for the Environment and Wrappers"""
    TEAM_BATCH_SIZE: int = 100      # Number of teams to pre-generate
    WIN_RATE_THRESHOLD: float = 0.70 # Win rate required to increase team size
    CURRICULUM_CHECK_INTERVAL: int = 1000 # How often to check win rate
    MIN_TEAM_SIZE: int = 1
    MAX_TEAM_SIZE: int = 6

@dataclass
class ResourceConfig:
    """Hardware resources"""
    NUM_GPUS: int = 0
    # For 8 cores: Leave 1 for the OS/Driver, use 7 for workers.
    # If your PC becomes laggy, reduce NUM_ENV_RUNNERS to 5 or 6.
    NUM_ENV_RUNNERS: int = 6       
    NUM_CPUS_PER_WORKER: int = 1    # 1 CPU core per emulator instance
    NUM_ENVS_PER_WORKER: int = 1

@dataclass
class PPOHyperparams:
    """PPO Algorithm Hyperparameters"""
    ROLLOUT_FRAGMENT_LENGTH: int = 100
    TRAIN_BATCH_SIZE: int = 4000    # Reduced slightly for local training
    SGD_MINIBATCH_SIZE: int = 256   # Reduced for CPU training
    NUM_SGD_ITER: int = 10
    LR: float = 5e-5
    GAMMA: float = 0.99

@dataclass
class RunConfig:
    """Experiment running configuration"""
    STOP_ITERATIONS: int = 10
    CHECKPOINT_FREQ: int = 1
    STORAGE_PATH_SUFFIX: str = "ray_results"
    EXP_NAME: str = "PPO_Pokemon_League_Local"

class TrainingConfig:
    ENV = EnvConfig()
    RESOURCES = ResourceConfig()
    PPO = PPOHyperparams()
    RUN = RunConfig()