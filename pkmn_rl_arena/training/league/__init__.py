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

# Sentinel name for the frozen random baseline policy
RANDOM_POLICY_ID = "random"


class LeagueManager:
    """
    Manages a league of policy snapshots for self-play.

    Opponent selection strategy (with snapshots available):
      - 20% random baseline  (fixed benchmark — the real progress indicator)
      - 35% self-play         (main_policy vs itself)
      - 30% recent snapshots  (last 5 frozen copies)
      - 15% random historical (any past snapshot)

    When no snapshots exist yet:
      - 50% random baseline
      - 50% self-play
    """

    def __init__(self, snapshot_interval: int = 50):
        self.past_policies: list[str] = []
        self.snapshot_interval = snapshot_interval
        self.elo_ratings: dict[str, float] = {
            "main_policy": 1500.0,
            RANDOM_POLICY_ID: 1000.0,
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

        # Always include random baseline matches
        if roll < 0.20:
            return RANDOM_POLICY_ID

        # Before any snapshots exist: 50/50 self-play vs random
        if len(self.past_policies) == 0:
            return "main_policy"

        if roll < 0.55:
            # 35%: self-play
            return "main_policy"
        elif roll < 0.85:
            # 30%: recent opponents
            recent = self.past_policies[-min(5, len(self.past_policies)) :]
            return random.choice(recent)
        else:
            # 15%: random historical
            return random.choice(self.past_policies)

    def update_elo(self, winner_id: str, loser_id: str, k: float = 32.0):
        """Update ELO ratings after a match result."""
        r_w = self.elo_ratings.get(winner_id, 1500.0)
        r_l = self.elo_ratings.get(loser_id, 1500.0)

        expected_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))

        self.elo_ratings[winner_id] = r_w + k * (1.0 - expected_w)
        self.elo_ratings[loser_id] = r_l + k * (0.0 - (1.0 - expected_w))

    def should_snapshot(self, iteration: int) -> bool:
        """Check if it's time to snapshot the current policy."""
        return iteration > 0 and iteration % self.snapshot_interval == 0

    def get_snapshot_id(self, iteration: int) -> str:
        """Generate a policy id for the snapshot at this iteration."""
        return f"policy_v{iteration}"
