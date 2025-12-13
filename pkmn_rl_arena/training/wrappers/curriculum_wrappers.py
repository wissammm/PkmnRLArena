import numpy as np
from collections import deque
from pkmn_rl_arena.logging import log

class BaseParallelWrapper:
    """
    A simple wrapper for PettingZoo ParallelEnv that delegates attributes to the wrapped env.
    """
    def __init__(self, env):
        self.env = env
    
    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        return self.env.step(actions)

class CurriculumWrapper(BaseParallelWrapper):
    """
    Adjusts the team_size in reset options based on win rate.
    """
    def __init__(self, env, win_rate_threshold=0.75, min_size=1, max_size=6, check_interval=1000):
        super().__init__(env)
        self.team_size = min_size
        self.max_size = max_size
        self.threshold = win_rate_threshold
        self.check_interval = check_interval
        self.win_history = deque(maxlen=check_interval)
        self.episodes = 0

    def reset(self, seed=None, options=None):
        options = options or {}
        # Inject current curriculum level
        options["team_size"] = self.team_size
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        for agent in infos:
            infos[agent]["team_size"] = self.team_size
        # Check for episode end
        if any(terminations.values()) or any(truncations.values()):
            # Assuming 1v1 agent setup for now (player vs enemy)
            # If player reward > 0, we consider it a win. 
            # Adjust this logic if your reward function is complex.
            is_win = 1 if rewards.get("player", 0) > 0 else 0
            self.win_history.append(is_win)
            self.episodes += 1

            if self.episodes % self.check_interval == 0:
                self._update_curriculum()
                
        return obs, rewards, terminations, truncations, infos

    def _update_curriculum(self):
        if len(self.win_history) < self.check_interval:
            return

        win_rate = np.mean(self.win_history)
        log.info(f"Curriculum Check: Win Rate {win_rate:.2f} (Threshold {self.threshold})")

        if win_rate >= self.threshold and self.team_size < self.max_size:
            self.team_size += 1
            self.win_history.clear() # Reset history for new difficulty
            log.info(f"*** CURRICULUM LEVEL UP! New Team Size: {self.team_size} ***")


class TeamBatchWrapper(BaseParallelWrapper):
    """
    Pre-generates batches of teams to reduce variance during training.
    """
    def __init__(self, env, team_factory, batch_size=100, refresh_interval=100):
        super().__init__(env)
        self.team_factory = team_factory
        self.batch_size = batch_size
        self.refresh_interval = refresh_interval
        
        self.team_buffer = []
        self.current_buffer_size = -1 # Tracks what team size the buffer was generated for
        self.episode_count = 0

    def reset(self, seed=None, options=None):
        options = options or {}
        requested_size = options.get("team_size", 6)

        # Check if we need to regenerate buffer (size changed or interval hit)
        size_changed = requested_size != self.current_buffer_size
        interval_hit = self.episode_count >= self.refresh_interval

        if size_changed or interval_hit:
            self._refresh_buffer(requested_size)

        # Pick a random team from buffer
        # We use the same team structure for both agents to ensure fair "mirror" matches 
        # in terms of complexity, though specific mons might differ if we sampled 2 indices.
        # Here we just grab one set of teams.
        
        if options.get("teams") is None:
            # We need two teams, one for player, one for enemy
            # We can pick two random indices
            idx_p = np.random.randint(0, len(self.team_buffer))
            idx_e = np.random.randint(0, len(self.team_buffer))
            
            options["teams"] = {
                "player": self.team_buffer[idx_p],
                "enemy": self.team_buffer[idx_e]
            }

        self.episode_count += 1
        return self.env.reset(seed=seed, options=options)

    def _refresh_buffer(self, team_size):
        log.info(f"Refreshing Team Batch Buffer (Size: {team_size})")
        # Generate a batch of teams
        # Note: team_factory.create_random_team returns a List[int]
        self.team_buffer = [
            self.team_factory.create_random_team(size_of_team=team_size) 
            for _ in range(self.batch_size)
        ]
        self.current_buffer_size = team_size
        self.episode_count = 0