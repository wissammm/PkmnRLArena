"""
Tests for training features:
  - AEC wrappers (TeamBatchWrapperAEC, CurriculumWrapperAEC)
  - LeagueManager (opponent selection, ELO, snapshots)
  - TrainingConfig
"""

import unittest
from unittest.mock import MagicMock, patch
from collections import deque

import numpy as np

from pkmn_rl_arena.training.wrappers.aec_wrappers import (
    TeamBatchWrapperAEC,
    CurriculumWrapperAEC,
)
from pkmn_rl_arena.training.league import LeagueManager, RANDOM_POLICY_ID
from pkmn_rl_arena.training.config import (
    TrainingConfig,
    EnvConfig,
    ResourceConfig,
    PPOHyperparams,
    RunConfig,
)
from pkmn_rl_arena.training.league.random_policy import MaskedRandomPolicy
from pkmn_rl_arena.env.observation import ObsIdx
from pkmn_rl_arena.env.action import ACTION_SPACE_SIZE


# ---------------------------------------------------------------------------
# Mock AEC environment for wrapper tests
# ---------------------------------------------------------------------------
class MockAECEnv:
    """
    Minimal mock that satisfies the AEC protocol expected by BaseWrapper.
    """

    def __init__(self):
        self.possible_agents = ["player", "enemy"]
        self.agents = ["player", "enemy"]
        self.rewards = {"player": 0.0, "enemy": 0.0}
        self.terminations = {"player": False, "enemy": False}
        self.truncations = {"player": False, "enemy": False}
        self.infos = {"player": {}, "enemy": {}}
        self._reset_options = None

    # AEC required attributes / methods --------------------------------
    def reset(self, seed=None, options=None):
        self._reset_options = options or {}
        self.rewards = {"player": 0.0, "enemy": 0.0}
        self.terminations = {"player": False, "enemy": False}
        self.truncations = {"player": False, "enemy": False}
        self.infos = {"player": {}, "enemy": {}}

    def step(self, action):
        pass

    def observe(self, agent):
        return {}

    def render(self):
        pass

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self

    def __getattr__(self, name):
        raise AttributeError(name)


# ======================================================================
# AEC Wrapper Tests
# ======================================================================
class TestTeamBatchWrapperAEC(unittest.TestCase):
    def setUp(self):
        self.env = MockAECEnv()
        self.mock_factory = MagicMock()
        self.mock_factory.create_random_team.side_effect = (
            lambda size_of_team: list(range(size_of_team))
        )
        self.wrapper = TeamBatchWrapperAEC(
            self.env,
            team_factory=self.mock_factory,
            batch_size=10,
            refresh_interval=5,
        )

    # --- basic ---
    def test_initial_buffer_generation(self):
        """First reset triggers buffer generation with correct size."""
        self.wrapper.reset(options={"team_size": 3})

        self.assertEqual(len(self.wrapper.team_buffer), 10)
        self.assertEqual(self.wrapper.current_buffer_size, 3)
        # Each team should be create_random_team(size_of_team=3) -> [0,1,2]
        for team in self.wrapper.team_buffer:
            self.assertEqual(team, [0, 1, 2])

    def test_reset_injects_teams(self):
        """When no teams passed, wrapper picks from buffer."""
        self.wrapper.reset(options={"team_size": 2})

        opts = self.env._reset_options
        self.assertIn("teams", opts)
        self.assertIn("player", opts["teams"])
        self.assertIn("enemy", opts["teams"])

    def test_manual_teams_not_overwritten(self):
        """If caller provides teams, wrapper should not replace them."""
        manual = {"player": [999], "enemy": [888]}
        self.wrapper.reset(options={"team_size": 1, "teams": manual})

        opts = self.env._reset_options
        self.assertEqual(opts["teams"]["player"], [999])
        self.assertEqual(opts["teams"]["enemy"], [888])

    def test_buffer_refresh_on_size_change(self):
        """Changing requested team_size triggers buffer regeneration."""
        self.wrapper.reset(options={"team_size": 1})
        self.assertEqual(self.wrapper.current_buffer_size, 1)

        self.wrapper.reset(options={"team_size": 4})
        self.assertEqual(self.wrapper.current_buffer_size, 4)
        for team in self.wrapper.team_buffer:
            self.assertEqual(team, [0, 1, 2, 3])

    def test_buffer_refresh_on_interval(self):
        """After refresh_interval resets, buffer is regenerated."""
        # refresh_interval=5, first reset triggers refresh due to size_changed (-1 != 1)
        # After that, episode_count increments each reset: 1,2,3,4,5
        # On the 6th reset, episode_count=5 >= refresh_interval=5 → refresh
        for _ in range(6):
            self.wrapper.reset(options={"team_size": 1})

        # After the 6th reset triggers refresh: count resets to 0, then +1 = 1
        self.assertEqual(self.wrapper.episode_count, 1)

    def test_episode_count_increments(self):
        """Each reset increments episode_count (until refresh)."""
        self.wrapper.reset(options={"team_size": 1})
        self.assertEqual(self.wrapper.episode_count, 1)
        self.wrapper.reset(options={"team_size": 1})
        self.assertEqual(self.wrapper.episode_count, 2)


class TestCurriculumWrapperAEC(unittest.TestCase):
    def setUp(self):
        self.env = MockAECEnv()
        self.wrapper = CurriculumWrapperAEC(
            self.env,
            win_rate_threshold=0.75,
            min_size=1,
            max_size=3,
            check_interval=4,
        )

    # --- basic ---
    def test_initial_state(self):
        self.assertEqual(self.wrapper.team_size, 1)
        self.assertEqual(len(self.wrapper.win_history), 0)
        self.assertEqual(self.wrapper.episodes, 0)

    def test_reset_injects_team_size(self):
        self.wrapper.reset(options={"other": "value"})
        opts = self.env._reset_options
        self.assertEqual(opts["team_size"], 1)
        self.assertEqual(opts["other"], "value")

    def test_reset_clears_episode_reward(self):
        self.wrapper._episode_player_reward = 42.0
        self.wrapper.reset()
        self.assertEqual(self.wrapper._episode_player_reward, 0.0)

    # --- curriculum progression ---
    def _simulate_episode_end(self, player_reward: float, terminated: bool = True):
        """Helper: simulate one complete episode ending."""
        self.env.rewards = {"player": player_reward, "enemy": -player_reward}
        if terminated:
            self.env.terminations = {"player": True, "enemy": True}
        else:
            self.env.truncations = {"player": True, "enemy": True}
        self.wrapper.step(action=0)

    def test_level_up_on_high_win_rate(self):
        """75% win rate at check_interval=4 → should level up."""
        self.wrapper.reset()
        for reward in [1.0, 1.0, 1.0, -1.0]:  # 3/4 = 75%
            self._simulate_episode_end(reward)

        self.assertEqual(self.wrapper.team_size, 2)
        self.assertEqual(len(self.wrapper.win_history), 0)  # cleared after level-up

    def test_no_level_up_on_low_win_rate(self):
        """50% win rate → should NOT level up."""
        self.wrapper.reset()
        for reward in [1.0, -1.0, 1.0, -1.0]:  # 2/4 = 50%
            self._simulate_episode_end(reward)

        self.assertEqual(self.wrapper.team_size, 1)
        self.assertEqual(len(self.wrapper.win_history), 4)

    def test_max_level_cap(self):
        """Already at max_size → should not exceed it."""
        self.wrapper.team_size = 3
        self.wrapper.reset()
        for _ in range(4):
            self._simulate_episode_end(1.0)

        self.assertEqual(self.wrapper.team_size, 3)

    def test_truncation_counts_as_episode_end(self):
        """Truncated episodes should also be tracked."""
        self.wrapper.reset()
        for reward in [1.0, 1.0, 1.0, -1.0]:
            self._simulate_episode_end(reward, terminated=False)

        self.assertEqual(self.wrapper.episodes, 4)
        self.assertEqual(self.wrapper.team_size, 2)

    def test_infos_contain_team_size(self):
        """After step, infos for all agents should contain team_size."""
        self.wrapper.reset()
        self.env.infos = {"player": {}, "enemy": {}}
        self.wrapper.step(action=0)

        for agent in self.env.possible_agents:
            self.assertIn("team_size", self.env.infos[agent])
            self.assertEqual(self.env.infos[agent]["team_size"], 1)

    def test_accumulated_reward_across_steps(self):
        """Player reward should accumulate across multiple steps in one episode."""
        self.wrapper.reset()
        # Step 1: partial reward, no termination
        self.env.rewards = {"player": 0.5, "enemy": -0.5}
        self.wrapper.step(action=0)
        self.assertAlmostEqual(self.wrapper._episode_player_reward, 0.5)

        # Step 2: more reward, still no termination
        self.env.rewards = {"player": 0.3, "enemy": -0.3}
        self.wrapper.step(action=0)
        self.assertAlmostEqual(self.wrapper._episode_player_reward, 0.8)


# ======================================================================
# LeagueManager Tests
# ======================================================================
class TestLeagueManager(unittest.TestCase):
    def setUp(self):
        self.league = LeagueManager(snapshot_interval=10)

    # --- initialization ---
    def test_initial_state(self):
        self.assertEqual(len(self.league.past_policies), 0)
        self.assertIn("main_policy", self.league.elo_ratings)
        self.assertIn(RANDOM_POLICY_ID, self.league.elo_ratings)
        self.assertEqual(self.league.elo_ratings["main_policy"], 1500.0)
        self.assertEqual(self.league.elo_ratings[RANDOM_POLICY_ID], 1000.0)

    # --- snapshot management ---
    def test_add_policy(self):
        self.league.add_policy("policy_v10")
        self.assertIn("policy_v10", self.league.past_policies)
        self.assertIn("policy_v10", self.league.elo_ratings)
        # New policy inherits main_policy ELO
        self.assertEqual(self.league.elo_ratings["policy_v10"], 1500.0)

    def test_should_snapshot(self):
        self.assertFalse(self.league.should_snapshot(0))
        self.assertFalse(self.league.should_snapshot(5))
        self.assertTrue(self.league.should_snapshot(10))
        self.assertTrue(self.league.should_snapshot(20))
        self.assertFalse(self.league.should_snapshot(15))

    def test_get_snapshot_id(self):
        self.assertEqual(self.league.get_snapshot_id(10), "policy_v10")
        self.assertEqual(self.league.get_snapshot_id(100), "policy_v100")

    # --- opponent selection ---
    def test_opponent_no_snapshots(self):
        """Without snapshots, should only return random or main_policy."""
        opponents = set()
        for _ in range(200):
            opponents.add(self.league.get_opponent())

        self.assertIn(RANDOM_POLICY_ID, opponents)
        self.assertIn("main_policy", opponents)
        # No snapshot policies should appear
        self.assertEqual(opponents, {RANDOM_POLICY_ID, "main_policy"})

    def test_opponent_with_snapshots(self):
        """With snapshots, all categories should eventually be selected."""
        for i in range(1, 11):
            self.league.add_policy(f"policy_v{i * 10}")

        opponents = set()
        for _ in range(1000):
            opponents.add(self.league.get_opponent())

        self.assertIn(RANDOM_POLICY_ID, opponents)
        self.assertIn("main_policy", opponents)
        # At least some historical policies should appear
        snapshot_opponents = opponents - {RANDOM_POLICY_ID, "main_policy"}
        self.assertGreater(len(snapshot_opponents), 0, "Should select snapshot policies")

    def test_opponent_distribution_approximate(self):
        """Rough check that opponent selection follows the expected distribution."""
        for i in range(1, 21):
            self.league.add_policy(f"policy_v{i * 10}")

        counts = {RANDOM_POLICY_ID: 0, "main_policy": 0, "recent": 0, "historical": 0}
        recent_ids = {f"policy_v{i * 10}" for i in range(16, 21)}  # last 5

        n = 10000
        for _ in range(n):
            opp = self.league.get_opponent()
            if opp == RANDOM_POLICY_ID:
                counts[RANDOM_POLICY_ID] += 1
            elif opp == "main_policy":
                counts["main_policy"] += 1
            elif opp in recent_ids:
                counts["recent"] += 1
            else:
                counts["historical"] += 1

        # Approximate checks (allow ±5%)
        self.assertAlmostEqual(counts[RANDOM_POLICY_ID] / n, 0.20, delta=0.05)
        self.assertAlmostEqual(counts["main_policy"] / n, 0.35, delta=0.05)
        self.assertAlmostEqual(counts["recent"] / n, 0.30, delta=0.05)
        self.assertAlmostEqual(counts["historical"] / n, 0.15, delta=0.05)

    # --- ELO ---
    def test_elo_update_winner_gains(self):
        """Winner should gain ELO, loser should lose ELO."""
        initial_main = self.league.elo_ratings["main_policy"]
        initial_random = self.league.elo_ratings[RANDOM_POLICY_ID]

        self.league.update_elo("main_policy", RANDOM_POLICY_ID)

        self.assertGreater(self.league.elo_ratings["main_policy"], initial_main)
        self.assertLess(self.league.elo_ratings[RANDOM_POLICY_ID], initial_random)

    def test_elo_conservation(self):
        """Total ELO change should sum to zero (zero-sum)."""
        initial_sum = sum(self.league.elo_ratings.values())

        self.league.update_elo("main_policy", RANDOM_POLICY_ID)

        new_sum = sum(self.league.elo_ratings.values())
        self.assertAlmostEqual(initial_sum, new_sum, places=6)

    def test_elo_update_equal_rating(self):
        """Two equally rated players: winner gets +16, loser gets -16 (k=32)."""
        self.league.elo_ratings["a"] = 1500.0
        self.league.elo_ratings["b"] = 1500.0

        self.league.update_elo("a", "b")

        # Expected win probability = 0.5, so change = k * (1 - 0.5) = 16
        self.assertAlmostEqual(self.league.elo_ratings["a"], 1516.0, places=4)
        self.assertAlmostEqual(self.league.elo_ratings["b"], 1484.0, places=4)

    def test_elo_underdog_wins_more(self):
        """An underdog winning should gain more ELO than a favourite winning."""
        self.league.elo_ratings["strong"] = 2000.0
        self.league.elo_ratings["weak"] = 1000.0

        # Underdog wins
        self.league.update_elo("weak", "strong")
        underdog_gain = self.league.elo_ratings["weak"] - 1000.0

        # Reset
        self.league.elo_ratings["strong"] = 2000.0
        self.league.elo_ratings["weak"] = 1000.0

        # Favourite wins
        self.league.update_elo("strong", "weak")
        favourite_gain = self.league.elo_ratings["strong"] - 2000.0

        self.assertGreater(underdog_gain, favourite_gain)


# ======================================================================
# TrainingConfig Tests
# ======================================================================
class TestTrainingConfig(unittest.TestCase):
    def test_default_values(self):
        config = TrainingConfig()
        self.assertEqual(config.ENV.MIN_TEAM_SIZE, 1)
        self.assertEqual(config.ENV.MAX_TEAM_SIZE, 6)
        self.assertEqual(config.ENV.WIN_RATE_THRESHOLD, 0.70)
        self.assertEqual(config.ENV.TEAM_BATCH_SIZE, 100)
        self.assertEqual(config.ENV.CURRICULUM_CHECK_INTERVAL, 1000)

    def test_resource_defaults(self):
        config = TrainingConfig()
        self.assertEqual(config.RESOURCES.NUM_GPUS, 0)
        self.assertEqual(config.RESOURCES.NUM_ENV_RUNNERS, 1)

    def test_ppo_defaults(self):
        ppo = PPOHyperparams()
        self.assertEqual(ppo.ROLLOUT_FRAGMENT_LENGTH, 100)
        self.assertEqual(ppo.TRAIN_BATCH_SIZE, 4000)
        self.assertEqual(ppo.SGD_MINIBATCH_SIZE, 256)
        self.assertEqual(ppo.NUM_SGD_ITER, 10)
        self.assertAlmostEqual(ppo.LR, 5e-5)
        self.assertAlmostEqual(ppo.GAMMA, 0.99)

    def test_run_config_defaults(self):
        rc = RunConfig()
        self.assertEqual(rc.STOP_ITERATIONS, 10)
        self.assertEqual(rc.CHECKPOINT_FREQ, 1)
        self.assertEqual(rc.EXP_NAME, "PPO_Pokemon_League_Local")

    def test_custom_env_config(self):
        env_cfg = EnvConfig(MIN_TEAM_SIZE=2, MAX_TEAM_SIZE=4, WIN_RATE_THRESHOLD=0.80)
        self.assertEqual(env_cfg.MIN_TEAM_SIZE, 2)
        self.assertEqual(env_cfg.MAX_TEAM_SIZE, 4)
        self.assertAlmostEqual(env_cfg.WIN_RATE_THRESHOLD, 0.80)


# ======================================================================
# MaskedRandomPolicy Tests
# ======================================================================
class TestMaskedRandomPolicy(unittest.TestCase):
    """
    Tests for MaskedRandomPolicy.extract_mask and compute_actions.

    The policy must handle three observation formats:
      1. Dict obs      → obs["action_mask"]
      2. Flat np array  → first 10 elements (alphabetical: action_mask, categorical, continuous)
      3. Unknown format → fallback to all-ones mask
    """

    FLAT_SIZE = ACTION_SPACE_SIZE + ObsIdx.CATEGORICAL_SIZE + ObsIdx.CONTINUOUS_SIZE  # 694

    # ------------------------------------------------------------------
    # extract_mask – dict observations
    # ------------------------------------------------------------------
    def test_extract_mask_dict_obs(self):
        """Dict obs with action_mask key → use it directly."""
        mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        obs = {"action_mask": mask, "categorical": np.zeros(324), "continuous": np.zeros(360)}

        result = MaskedRandomPolicy.extract_mask(obs, ACTION_SPACE_SIZE, self.FLAT_SIZE)
        np.testing.assert_array_equal(result, mask)

    def test_extract_mask_dict_obs_missing_key(self):
        """Dict obs without action_mask key → fallback to all-ones."""
        obs = {"categorical": np.zeros(324), "continuous": np.zeros(360)}
        result = MaskedRandomPolicy.extract_mask(obs, ACTION_SPACE_SIZE, self.FLAT_SIZE)
        np.testing.assert_array_equal(result, np.ones(ACTION_SPACE_SIZE))

    # ------------------------------------------------------------------
    # extract_mask – flattened observations (as RLlib sends them)
    # ------------------------------------------------------------------
    def test_extract_mask_flat_raw_binary(self):
        """Flat obs with raw binary [0,1] mask in the first 10 elements."""
        flat_obs = np.zeros(self.FLAT_SIZE, dtype=np.float32)
        # Set action_mask portion: actions 0,1,2,3 are legal
        flat_obs[0:4] = 1.0

        result = MaskedRandomPolicy.extract_mask(flat_obs, ACTION_SPACE_SIZE, self.FLAT_SIZE)
        expected = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_extract_mask_flat_normalized(self):
        """Flat obs with RLlib-normalized mask [-1, 1] in the first 10 elements.

        RLlib normalises Box(0,1) to [-1,1]: 0→-1, 1→+1.
        The threshold approach should recover the original binary mask.
        """
        flat_obs = np.zeros(self.FLAT_SIZE, dtype=np.float32)
        # Simulate normalisation: legal=+1, illegal=-1
        flat_obs[0:10] = -1.0   # all illegal first
        flat_obs[0] = 1.0       # action 0 legal
        flat_obs[3] = 1.0       # action 3 legal

        result = MaskedRandomPolicy.extract_mask(flat_obs, ACTION_SPACE_SIZE, self.FLAT_SIZE)
        expected = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_extract_mask_flat_all_legal(self):
        """All mask values are identical (all 1 or all -1) → fallback to all-ones."""
        flat_obs = np.zeros(self.FLAT_SIZE, dtype=np.float32)
        flat_obs[0:10] = 1.0  # all same value → can't distinguish

        result = MaskedRandomPolicy.extract_mask(flat_obs, ACTION_SPACE_SIZE, self.FLAT_SIZE)
        # When all values identical, hi-lo < 1e-8 → fallback
        np.testing.assert_array_equal(result, np.ones(ACTION_SPACE_SIZE, dtype=np.float32))

    def test_extract_mask_flat_single_legal(self):
        """Only one action legal — should correctly isolate it."""
        flat_obs = np.zeros(self.FLAT_SIZE, dtype=np.float32)
        flat_obs[0:10] = -1.0
        flat_obs[5] = 1.0  # only action 5 legal

        result = MaskedRandomPolicy.extract_mask(flat_obs, ACTION_SPACE_SIZE, self.FLAT_SIZE)
        expected = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        expected[5] = 1.0
        np.testing.assert_array_equal(result, expected)

    # ------------------------------------------------------------------
    # extract_mask – wrong shape / unknown format
    # ------------------------------------------------------------------
    def test_extract_mask_wrong_flat_size(self):
        """Flat obs with unexpected size → fallback to all-ones."""
        wrong_obs = np.zeros(100, dtype=np.float32)
        result = MaskedRandomPolicy.extract_mask(wrong_obs, ACTION_SPACE_SIZE, self.FLAT_SIZE)
        np.testing.assert_array_equal(result, np.ones(ACTION_SPACE_SIZE))

    def test_extract_mask_unknown_type(self):
        """Non-dict, non-ndarray obs → fallback to all-ones."""
        result = MaskedRandomPolicy.extract_mask([1, 0, 0], ACTION_SPACE_SIZE, self.FLAT_SIZE)
        np.testing.assert_array_equal(result, np.ones(ACTION_SPACE_SIZE))

    # ------------------------------------------------------------------
    # compute_actions – end-to-end action selection
    # ------------------------------------------------------------------
    def test_actions_respect_dict_mask(self):
        """Actions chosen from dict obs should only be from legal actions."""
        mask = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        obs = {"action_mask": mask, "categorical": np.zeros(324), "continuous": np.zeros(360)}

        policy = MaskedRandomPolicy.__new__(MaskedRandomPolicy)
        policy.action_space_size = ACTION_SPACE_SIZE

        for _ in range(50):
            actions, _, _ = policy.compute_actions([obs])
            self.assertIn(actions[0], [2, 3], f"Action {actions[0]} is not legal (mask={mask})")

    def test_actions_respect_flat_mask(self):
        """Actions chosen from flattened obs should only be from legal actions."""
        flat_obs = np.zeros(self.FLAT_SIZE, dtype=np.float32)
        flat_obs[0:10] = -1.0
        flat_obs[0] = 1.0
        flat_obs[7] = 1.0  # legal: 0, 7

        policy = MaskedRandomPolicy.__new__(MaskedRandomPolicy)
        policy.action_space_size = ACTION_SPACE_SIZE

        for _ in range(50):
            actions, _, _ = policy.compute_actions([flat_obs])
            self.assertIn(actions[0], [0, 7], f"Action {actions[0]} is not legal")

    def test_actions_batch(self):
        """Batch of observations should each be handled independently."""
        mask_a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        mask_b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        obs_a = {"action_mask": mask_a, "categorical": np.zeros(324), "continuous": np.zeros(360)}
        obs_b = {"action_mask": mask_b, "categorical": np.zeros(324), "continuous": np.zeros(360)}

        policy = MaskedRandomPolicy.__new__(MaskedRandomPolicy)
        policy.action_space_size = ACTION_SPACE_SIZE

        for _ in range(30):
            actions, _, _ = policy.compute_actions([obs_a, obs_b])
            self.assertEqual(actions[0], 0, "obs_a only allows action 0")
            self.assertEqual(actions[1], 9, "obs_b only allows action 9")

    def test_actions_empty_mask_fallback(self):
        """If mask is all zeros, fallback action should be 0."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        obs = {"action_mask": mask, "categorical": np.zeros(324), "continuous": np.zeros(360)}

        policy = MaskedRandomPolicy.__new__(MaskedRandomPolicy)
        policy.action_space_size = ACTION_SPACE_SIZE

        actions, _, _ = policy.compute_actions([obs])
        self.assertEqual(actions[0], 0, "Fallback when mask is all-zeros should be action 0")

    def test_frozen_policy_no_learning(self):
        """learn_on_batch should be a no-op returning empty dict."""
        policy = MaskedRandomPolicy.__new__(MaskedRandomPolicy)
        result = policy.learn_on_batch(None)
        self.assertEqual(result, {})

    def test_frozen_policy_weights(self):
        """get_weights / set_weights should be no-ops."""
        policy = MaskedRandomPolicy.__new__(MaskedRandomPolicy)
        self.assertEqual(policy.get_weights(), {})
        policy.set_weights({"anything": 42})  # should not raise


if __name__ == "__main__":
    unittest.main()
