"""
League Manager for self-play training.

Handles:
  - Policy snapshot management
  - Opponent selection with Prioritized Fictitious Self-Play (PFSP)
  - ELO rating tracking
  - Random baseline benchmark for measuring true progress
"""

import random
from pkmn_rl_arena.logging import log
from pkmn_rl_arena.training.config import TrainingConfig

RANDOM_POLICY_ID = "random"


class LeagueManager:
    """
    Manages a league of policy snapshots for self-play.

    Opponent selection strategy uses probabilities from TrainingConfig.LEAGUE.
    """

    def __init__(self, snapshot_interval: int | None = None):
        self.config = TrainingConfig.LEAGUE
        self.past_policies: list[str] = []
        self.snapshot_interval = (
            self.config.SNAPSHOT_INTERVAL
            if snapshot_interval is None
            else snapshot_interval
        )
        self.elo_ratings: dict[str, float] = {
            "main_policy": self.config.ELO_MAIN_INIT,
            RANDOM_POLICY_ID: self.config.ELO_RANDOM_INIT,
        }

    def add_policy(self, policy_id: str):
        """Register a new frozen policy snapshot."""
        self.past_policies.append(policy_id)
        self.elo_ratings[policy_id] = self.elo_ratings["main_policy"]
        log.info(
            f"League: Added {policy_id} | "
            f"Total snapshots: {len(self.past_policies)} | "
            f"ELO: {self.elo_ratings[policy_id]:.0f}"
        )

    def get_opponent(self) -> str:
        """
        Select an opponent policy using PFSP-inspired strategy.
        Returns a policy_id string.
        """
        roll = random.random()
        if len(self.past_policies) == 0:
            if roll < self.config.PROB_RANDOM_BASELINE_NO_SNAPSHOT:
                return RANDOM_POLICY_ID
            return "main_policy"

        if roll < self.config.PROB_RANDOM_BASELINE:
            return RANDOM_POLICY_ID
        elif roll < self.config.PROB_RANDOM_BASELINE + self.config.PROB_SELFPLAY:
            return "main_policy"
        elif roll < (
            self.config.PROB_RANDOM_BASELINE
            + self.config.PROB_SELFPLAY
            + self.config.PROB_RECENT
        ):
            recent = self.past_policies[
                -min(self.config.RECENT_WINDOW, len(self.past_policies)) :
            ]
            return random.choice(recent)
        return random.choice(self.past_policies)

    def update_elo(self, winner_id: str, loser_id: str, k: float | None = None):
        """Update ELO ratings after a match result."""
        k_factor = self.config.ELO_K if k is None else k
        r_w = self.elo_ratings.get(winner_id, self.config.ELO_MAIN_INIT)
        r_l = self.elo_ratings.get(loser_id, self.config.ELO_MAIN_INIT)

        expected_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
        expected_l = 1.0 - expected_w

        self.elo_ratings[winner_id] = r_w + k_factor * (1.0 - expected_w)
        self.elo_ratings[loser_id] = r_l + k_factor * (0.0 - expected_l)

        self.elo_ratings[winner_id] = max(
            self.config.ELO_MIN, min(self.config.ELO_MAX, self.elo_ratings[winner_id])
        )
        self.elo_ratings[loser_id] = max(
            self.config.ELO_MIN, min(self.config.ELO_MAX, self.elo_ratings[loser_id])
        )

    def should_snapshot(self, iteration: int) -> bool:
        """Check if it's time to snapshot the current policy."""
        return iteration > 0 and iteration % self.snapshot_interval == 0

    def get_snapshot_id(self, iteration: int) -> str:
        """Generate a policy id for the snapshot at this iteration."""
        return f"policy_v{iteration}"
