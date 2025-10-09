class Config:
    def __init__(self, 
                 num_workers: int = 7,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.9995,
                 batch_size: int = 256,
                 learning_rate: float = 1e-4,
                 target_update: int = 100,
                 buffer_size: int = 50000,
                 num_episodes: int = 20000,
                 stats_window: int = 100,
                 update_frequency: int = 4,
                 save_interval: int = 500,
                 plot_interval: int = 100,
                 initial_elo: float = 1500.0,
                 elo_k_factor: float = 32.0):
        
        self.num_workers = num_workers
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update = target_update
        self.buffer_size = buffer_size
        self.num_episodes = num_episodes
        self.stats_window = stats_window
        self.update_frequency = update_frequency
        self.save_interval = save_interval
        self.plot_interval = plot_interval
        self.initial_elo = initial_elo
        self.elo_k_factor = elo_k_factor

    @classmethod
    def from_dict(cls, config_dict):
        """Create a Config instance from a dictionary."""
        return cls(**{k.lower(): v for k, v in config_dict.items()})