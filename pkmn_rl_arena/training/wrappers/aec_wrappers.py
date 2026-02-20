"""
AEC-compatible wrappers for PettingZoo AECEnv environments.

These wrap a BattleArenaAEC and intercept reset/step to add:
  - TeamBatchWrapperAEC: pre-generated team batches
  - CurriculumWrapperAEC: progressive team size based on win rate
"""

import numpy as np
from collections import deque
from pettingzoo.utils import BaseWrapper
from pkmn_rl_arena.logging import log


class TeamBatchWrapperAEC(BaseWrapper):
    """
    Pre-generates batches of teams to reduce variance during training.
    AEC-compatible: delegates all AEC protocol methods to the wrapped env.
    """

    def __init__(self, env, team_factory, batch_size=100, refresh_interval=100):
        super().__init__(env)
        self.team_factory = team_factory
        self.batch_size = batch_size
        self.refresh_interval = refresh_interval

        self.team_buffer = []
        self.current_buffer_size = -1
        self.episode_count = 0

    def reset(self, seed=None, options=None):
        options = options or {}
        options.setdefault("save_state", "boot_state")
        requested_size = options.get("team_size", 6)

        size_changed = requested_size != self.current_buffer_size
        interval_hit = self.episode_count >= self.refresh_interval

        if size_changed or interval_hit:
            self._refresh_buffer(requested_size)

        if options.get("teams") is None:
            idx_p = np.random.randint(0, len(self.team_buffer))
            idx_e = np.random.randint(0, len(self.team_buffer))
            options["teams"] = {
                "player": list(self.team_buffer[idx_p]),
                "enemy": list(self.team_buffer[idx_e]),
            }

        self.episode_count += 1
        self.env.reset(seed=seed, options=options)

    def _refresh_buffer(self, team_size):
        log.info(f"Refreshing Team Batch Buffer (Size: {team_size})")
        self.team_buffer = [
            self.team_factory.create_random_team(size_of_team=team_size)
            for _ in range(self.batch_size)
        ]
        self.current_buffer_size = team_size
        self.episode_count = 0


class CurriculumWrapperAEC(BaseWrapper):
    """
    Adjusts team_size in reset options based on win rate.
    AEC-compatible: tracks rewards across sequential agent steps.
    """

    def __init__(
        self, env, win_rate_threshold=0.75, min_size=1, max_size=6, check_interval=1000
    ):
        super().__init__(env)
        self.team_size = min_size
        self.max_size = max_size
        self.threshold = win_rate_threshold
        self.check_interval = check_interval
        self.win_history = deque(maxlen=check_interval)
        self.episodes = 0
        self._episode_player_reward = 0.0

    def reset(self, seed=None, options=None):
        options = options or {}
        options.setdefault("save_state", "boot_state")
        options["team_size"] = self.team_size
        self._episode_player_reward = 0.0
        self.env.reset(seed=seed, options=options)

    def step(self, action):
        self.env.step(action)

        # Accumulate player reward across steps
        self._episode_player_reward += self.env.rewards.get("player", 0.0)

        # Inject curriculum info into infos for all agents
        for agent in self.env.possible_agents:
            if agent in self.env.infos:
                self.env.infos[agent]["team_size"] = self.team_size

        # Check for episode end (all agents terminated or truncated)
        all_terminated = all(
            self.env.terminations.get(a, False) for a in self.env.possible_agents
        )
        all_truncated = all(
            self.env.truncations.get(a, False) for a in self.env.possible_agents
        )

        if all_terminated or all_truncated:
            is_win = 1 if self._episode_player_reward > 0 else 0
            self.win_history.append(is_win)
            self.episodes += 1

            if self.episodes % self.check_interval == 0:
                self._update_curriculum()

    def _update_curriculum(self):
        if len(self.win_history) < self.check_interval:
            return

        win_rate = np.mean(self.win_history)
        log.info(
            f"Curriculum Check: Win Rate {win_rate:.2f} (Threshold {self.threshold})"
        )

        if win_rate >= self.threshold and self.team_size < self.max_size:
            self.team_size += 1
            self.win_history.clear()
            log.info(f"*** CURRICULUM LEVEL UP! New Team Size: {self.team_size} ***")
