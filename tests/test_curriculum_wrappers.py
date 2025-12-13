import unittest
from unittest.mock import MagicMock
from pettingzoo.utils.env import ParallelEnv

from pkmn_rl_arena.training.wrappers.curriculum_wrappers import CurriculumWrapper, TeamBatchWrapper

class MockEnv(ParallelEnv):
    def __init__(self):
        self.agents = ["player", "enemy"]
        self.reset_options = {}
    
    def reset(self, seed=None, options=None):
        self.reset_options = options
        return {}, {}

    def step(self, actions):
        # Default dummy return
        return {}, {}, {}, {}, {}

class TestCurriculumWrapper(unittest.TestCase):
    def setUp(self):
        self.env = MockEnv()
        # Threshold 0.75, check every 4 episodes for easy testing
        self.wrapper = CurriculumWrapper(
            self.env, 
            win_rate_threshold=0.75, 
            min_size=1, 
            max_size=3, 
            check_interval=4
        )

    def test_initialization(self):
        self.assertEqual(self.wrapper.team_size, 1)
        self.assertEqual(len(self.wrapper.win_history), 0)

    def test_reset_injects_team_size(self):
        self.wrapper.reset(options={"other_opt": "value"})
        
        # Check if underlying env received the team_size
        self.assertIn("team_size", self.env.reset_options)
        self.assertEqual(self.env.reset_options["team_size"], 1)
        self.assertEqual(self.env.reset_options["other_opt"], "value")

    def test_win_tracking(self):
        # Simulate a win
        # step returns: obs, rewards, terminations, truncations, infos
        self.wrapper.step({}) # Normal step
        
        # Terminal step with win
        rewards = {"player": 1.0}
        terminations = {"player": True}
        
        # Mock the super().step call
        self.env.step = MagicMock(return_value=({}, rewards, terminations, {}, {}))
        
        self.wrapper.step({})
        
        self.assertEqual(len(self.wrapper.win_history), 1)
        self.assertEqual(self.wrapper.win_history[0], 1) # 1 for win

    def test_curriculum_level_up(self):
        # We need 3 wins out of 4 to hit 0.75 threshold
        wins = [1, 1, 1, 0] # 75% win rate
        
        self.env.step = MagicMock()
        
        for w in wins:
            rewards = {"player": 1.0 if w else -1.0}
            terminations = {"player": True}
            self.env.step.return_value = ({}, rewards, terminations, {}, {})
            self.wrapper.step({})

        # Should have leveled up
        self.assertEqual(self.wrapper.team_size, 2)
        # History should be cleared after level up
        self.assertEqual(len(self.wrapper.win_history), 0)

    def test_curriculum_stay_same(self):
        # 2 wins out of 4 = 50% < 75%
        wins = [1, 1, 0, 0] 
        
        self.env.step = MagicMock()
        
        for w in wins:
            rewards = {"player": 1.0 if w else -1.0}
            terminations = {"player": True}
            self.env.step.return_value = ({}, rewards, terminations, {}, {})
            self.wrapper.step({})

        # Should NOT have leveled up
        self.assertEqual(self.wrapper.team_size, 1)
        # History should NOT be cleared
        self.assertEqual(len(self.wrapper.win_history), 4)

    def test_max_level_cap(self):
        self.wrapper.team_size = 3 # Max size
        
        wins = [1, 1, 1, 1] # 100% win rate
        
        self.env.step = MagicMock()
        for w in wins:
            rewards = {"player": 1.0}
            terminations = {"player": True}
            self.env.step.return_value = ({}, rewards, terminations, {}, {})
            self.wrapper.step({})

        # Should stay at max
        self.assertEqual(self.wrapper.team_size, 3)


class TestTeamBatchWrapper(unittest.TestCase):
    def setUp(self):
        self.env = MockEnv()
        self.mock_factory = MagicMock()
        # Mock create_random_team to return a dummy list based on size
        self.mock_factory.create_random_team.side_effect = lambda size_of_team: [size_of_team] * 6
        
        self.batch_size = 10
        self.refresh_interval = 5
        
        self.wrapper = TeamBatchWrapper(
            self.env, 
            self.mock_factory, 
            batch_size=self.batch_size, 
            refresh_interval=self.refresh_interval
        )

    def test_initial_buffer_generation(self):
        # First reset should trigger generation
        self.wrapper.reset(options={"team_size": 2})
        
        self.assertEqual(len(self.wrapper.team_buffer), self.batch_size)
        self.assertEqual(self.wrapper.current_buffer_size, 2)
        # Check content of generated teams (based on our mock lambda)
        self.assertEqual(self.wrapper.team_buffer[0], [2]*6)
        
        # Verify env received teams
        opts = self.env.reset_options
        self.assertIsNotNone(opts["teams"]["player"])
        self.assertIsNotNone(opts["teams"]["enemy"])

    def test_buffer_refresh_on_size_change(self):
        # Initial setup size 1
        self.wrapper.reset(options={"team_size": 1})
        self.assertEqual(self.wrapper.current_buffer_size, 1)
        
        # Reset with size 2 (Curriculum level up)
        self.wrapper.reset(options={"team_size": 2})
        
        self.assertEqual(self.wrapper.current_buffer_size, 2)
        self.assertEqual(self.wrapper.team_buffer[0], [2]*6)
        # Episode count should reset
        self.assertEqual(self.wrapper.episode_count, 1)

    def test_buffer_refresh_interval(self):
        self.wrapper.reset(options={"team_size": 1})
        initial_buffer = list(self.wrapper.team_buffer) # Copy
        
        # Run episodes up to interval
        # We already ran 1 in the first reset above
        for _ in range(self.refresh_interval - 1):
            self.wrapper.reset(options={"team_size": 1})
            
        # Next reset should trigger refresh
        self.mock_factory.create_random_team.reset_mock() # Reset call count
        self.wrapper.reset(options={"team_size": 1})
        
        self.assertTrue(self.mock_factory.create_random_team.called)
        self.assertEqual(self.wrapper.episode_count, 1) # Resets to 1 after refresh

    def test_manual_team_override(self):
        # If we manually pass teams, wrapper should NOT overwrite them
        manual_team = [999]
        options = {
            "teams": {
                "player": manual_team,
                "enemy": manual_team
            },
            "team_size": 1
        }
        
        self.wrapper.reset(options=options)
        
        self.assertEqual(self.env.reset_options["teams"]["player"], manual_team)
        self.assertEqual(self.env.reset_options["teams"]["enemy"], manual_team)

if __name__ == "__main__":
    unittest.main()