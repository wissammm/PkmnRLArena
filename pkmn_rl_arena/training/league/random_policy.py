"""
A random policy that respects action masks.
Used as a frozen baseline benchmark to measure true training progress.
"""
import numpy as np
from ray.rllib.policy.policy import Policy
from gymnasium.spaces import Discrete

from pkmn_rl_arena.env.observation import ObsIdx
from pkmn_rl_arena.env.action import ACTION_SPACE_SIZE


class MaskedRandomPolicy(Policy):
    """
    Random policy that respects action masks.
    Always picks uniformly among LEGAL actions only.
    Never trained — serves as a fixed benchmark.

    Observation layout (Dict space, flattened alphabetically by RLlib):
      action_mask (10) | categorical (324) | continuous (360) = 694 total
    The action_mask occupies the FIRST 10 elements of the flattened vector.
    """

    # Pre-compute the expected flat observation size so the mask slice is
    # only applied when the shape matches exactly.
    EXPECTED_FLAT_SIZE = ACTION_SPACE_SIZE + ObsIdx.CATEGORICAL_SIZE + ObsIdx.CONTINUOUS_SIZE

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        if hasattr(action_space, "n"):
            self.action_space_size = action_space.n
        else:
            self.action_space_size = ACTION_SPACE_SIZE

    @staticmethod
    def extract_mask(obs, action_space_size: int, expected_flat_size: int) -> np.ndarray:
        """
        Extract a binary action mask from an observation.

        Handles three layouts:
          1. dict  – obs["action_mask"]
          2. flat numpy (694,) – first 10 elements, binarised with a
             threshold to survive RLlib normalisation.
          3. anything else – fallback to all-ones (every action legal).
        """
        if isinstance(obs, dict):
            return np.asarray(obs.get("action_mask", np.ones(action_space_size)))

        if isinstance(obs, np.ndarray) and obs.shape == (expected_flat_size,):
            raw = obs[:action_space_size]
            # RLlib's preprocessor may normalise [0,1] → [-1,1].
            # Binarise around the midpoint of the observed range.
            lo, hi = raw.min(), raw.max()
            if hi - lo < 1e-8:
                # All values identical → cannot distinguish; fallback.
                return np.ones(action_space_size, dtype=np.float32)
            threshold = (lo + hi) / 2.0
            return (raw > threshold).astype(np.float32)

        # Unknown layout – everything legal.
        return np.ones(action_space_size, dtype=np.float32)

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):
        """Pick a random LEGAL action for each observation in the batch."""
        actions = []
        for obs in obs_batch:
            mask = self.extract_mask(obs, self.action_space_size, self.EXPECTED_FLAT_SIZE)
            legal_actions = np.where(mask > 0)[0]

            if len(legal_actions) == 0:
                actions.append(0)
            else:
                actions.append(int(np.random.choice(legal_actions)))

        return actions, [], {}

    def learn_on_batch(self, samples):
        """Never learn — this is a frozen baseline."""
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass