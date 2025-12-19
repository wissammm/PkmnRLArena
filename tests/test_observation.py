from pkmn_rl_arena.env.rendering import TYPES_ID
import unittest
import numpy as np
import picologging as logging
import random

import pkmn_rl_arena
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena import log
from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.battle_arena import BattleArena
from pkmn_rl_arena.data import pokemon_data
from pkmn_rl_arena.env.turn_type import TurnType

from pkmn_rl_arena.env.observation import Observation, ObsIdx
from pkmn_rl_arena.env.action import ACTION_SPACE_SIZE


class TestObservation(unittest.TestCase):
    def setUp(self):
        log.setLevel(logging.DEBUG)
        core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.arena = BattleArena(core)

    def tearDown(self):
        self.arena.close()

    def test_observation_space(self):
        teams = {
            "player": [
                25, 10, 84, 84, 84, 84, 100, 0,  # Pikachu (ID 25) as active
                1, 10, 14, 14, 14, 14, 50, 0,        # Bulbasaur
            ],
            "enemy": [
                21, 10, 45, 45, 45, 45, 80, 0,    # SPEAROW (ID 21) as active
                4, 10, 5, 14, 25, 34, 60, 0,
            ],
        }

        options = {"teams": teams}
        observations, infos = self.arena.reset(options=options)
        self.assertEqual(self.arena.ctxt.core.state.turn, TurnType.GENERAL)

        player_team_dump = self.arena.ctxt.core.read_team_data("player")
        enemy_team_dump = self.arena.ctxt.core.read_team_data("enemy")

        player_df = pokemon_data.to_pandas_team_dump_data(player_team_dump)
        enemy_df = pokemon_data.to_pandas_team_dump_data(enemy_team_dump)

        self.assertEqual(player_df[player_df["isActive"] == 1].iloc[0]["id"], 25)
        self.assertEqual(enemy_df[enemy_df["isActive"] == 1].iloc[0]["id"], 21)

        observation_factory = self.arena.observation_factory
        observations = observation_factory.from_game()
        self.assertIsInstance(observations, Observation)
        self.assertIn("player", observations._o)
        self.assertIn("enemy", observations._o)

        for agent in ["player", "enemy"]:
            obs = observations._o[agent]
            self.assertIsInstance(obs, np.ndarray)
            self.assertTrue(np.issubdtype(obs.dtype, np.integer))
            expected_shape = (6 * ObsIdx.NB_DATA_PKMN,)
            self.assertEqual(obs.shape, expected_shape)

        player_obs = observations._o["player"]
        enemy_obs = observations._o["enemy"]

        player_active_idx = 0
        enemy_active_idx = 0

        # Check species ID
        self.assertEqual(
            player_obs[player_active_idx + ObsIdx.RAW_DATA["species"]], 25
        )
        self.assertEqual(
            enemy_obs[enemy_active_idx + ObsIdx.RAW_DATA["species"]], 21
        )

        self.assertEqual(
            player_obs[player_active_idx + ObsIdx.RAW_DATA["is_active"]], 1
        )
        self.assertEqual(
            enemy_obs[enemy_active_idx + ObsIdx.RAW_DATA["is_active"]], 1
        )

        player_second_idx = ObsIdx.NB_DATA_PKMN
        enemy_second_idx = ObsIdx.NB_DATA_PKMN

        self.assertEqual(
            player_obs[player_second_idx + ObsIdx.RAW_DATA["species"]], 1
        )
        self.assertEqual(
            enemy_obs[enemy_second_idx + ObsIdx.RAW_DATA["species"]], 4
        )

        self.assertEqual(
            player_obs[player_second_idx + ObsIdx.RAW_DATA["is_active"]], 0
        )
        self.assertEqual(
            enemy_obs[enemy_second_idx + ObsIdx.RAW_DATA["is_active"]], 0
        )

        move_start = ObsIdx.RAW_DATA["moves_begin"]
        first_move_offset = move_start + ObsIdx.RAW_DATA["move_id_offset"]
        power_offset = move_start + ObsIdx.RAW_DATA["power_offset"]

        self.assertEqual(player_obs[player_active_idx + first_move_offset], 84)  # Move ID
        # power for Thunder Shock expected 40 in observation representation
        self.assertEqual(player_obs[player_active_idx + power_offset], 40)

        self.assertEqual(
            int(player_obs[ObsIdx.RAW_DATA["type_1"]]),
            13,
            "Pikachu type 1 should be ELECTRIK (13)",
        )
        self.assertEqual(
            int(player_obs[ObsIdx.RAW_DATA["type_2"]]),
            13,
            "Pikachu type 2 should be None (0)",
        )

        move_start = ObsIdx.RAW_DATA["moves_begin"]
        first_move_offset = move_start + ObsIdx.RAW_DATA["move_id_offset"]
        power_offset = move_start + ObsIdx.RAW_DATA["power_offset"]
        pp_offset = move_start + ObsIdx.RAW_DATA["pp_offset"] 

        self.assertEqual(player_obs[player_active_idx + first_move_offset], 84)  # Move ID
        self.assertEqual(player_obs[player_active_idx + power_offset], 40)
        self.assertGreaterEqual(player_obs[player_active_idx + pp_offset], 29, "First move should have at least 29 PP, thundershock=30pp")

        for i in range(2, 6):  # Pokémon 3 to 6
            pkmn_start = i * ObsIdx.NB_DATA_PKMN
            self.assertEqual(
                player_obs[pkmn_start + ObsIdx.RAW_DATA["species"]], 0
            )
            self.assertEqual(
                enemy_obs[pkmn_start + ObsIdx.RAW_DATA["species"]], 0
            )

    def test_reduced_observation(self):
        """Test that reduced observation has correct size and content"""
        teams = {
            "player": [25, 50, 84, 0, 0, 0, 100, 0],
            "enemy": [129, 10, 150, 0, 0, 0, 10, 0],
        }
        options = {"teams": teams}
        self.arena.reset(options=options)
        
        obs = self.arena.observation_factory.from_game()
        reduced_obs = obs.get_reduced_agent_data("player")
        
        # 6 pokemon * 52 features = 312 (16 base + 36 move features per Pokémon)
        expected_size = 312
        self.assertEqual(reduced_obs.shape[0], expected_size)

    def test_embedding_data(self):
        """Test the splitting of data into categorical and continuous for embeddings"""
        teams = {
            "player": [25, 50, 84, 0, 0, 0, 100, 0],
            "enemy": [129, 10, 150, 0, 0, 0, 10, 0],
        }
        options = {"teams": teams}
        self.arena.reset(options=options)
        
        obs = self.arena.observation_factory.from_game()
        emb_data = obs.get_embedding_data("player")
        
        self.assertIn("categorical", emb_data)
        self.assertIn("continuous", emb_data)
        
        cat_data = emb_data["categorical"]
        cont_data = emb_data["continuous"]
        
        self.assertTrue(np.issubdtype(cat_data.dtype, np.integer))
        self.assertTrue(np.issubdtype(cont_data.dtype, np.floating))
        
        self.assertEqual(cat_data[0], 25)
        
        self.assertEqual(cont_data[0], 1.0)
        
        self.assertTrue(np.all(cont_data >= 0.0))
        self.assertTrue(np.all(cont_data <= 1.0))

    def test_helper_methods(self):
        """Test helper methods like active_pkmn, hp, etc."""
        teams = {
            "player": [25, 50, 84, 0, 0, 0, 100, 0],
            "enemy": [129, 10, 150, 0, 0, 0, 10, 0],
        }
        options = {"teams": teams}
        self.arena.reset(options=options)
        obs = self.arena.observation_factory.from_game()
        
        # active_pkmn
        active = obs.active_pkmn()
        self.assertEqual(active["player"], 0) 
        
        # hp
        hps = obs.hp()
        # Pikachu lvl 50 has some HP, check it's > 0
        self.assertGreater(hps["player"][0], 0)
        
        # pkmn_ko
        kos = obs.pkmn_ko()
        self.assertFalse(kos["player"][0]) 
        self.assertTrue(kos["player"][1])

        # who_won
        self.assertIsNone(obs.who_won()) # Battle just started

    def test_embedding_data_shapes_and_dtypes(self):
        """Test that get_embedding_data returns exact shapes and dtypes to match observation_space."""
        teams = {
            "player": [25, 50, 84, 0, 0, 0, 100, 0],  # Pikachu with one move
            "enemy": [129, 10, 150, 0, 0, 0, 10, 0],  # Magikarp
        }
        options = {"teams": teams}
        self.arena.reset(options=options)
        
        obs = self.arena.observation_factory.from_game()
        
        for agent in ["player", "enemy"]:
            emb_data = obs.get_embedding_data(agent)
            
            # Check keys exist
            self.assertIn("categorical", emb_data)
            self.assertIn("continuous", emb_data)
            
            cat_data = emb_data["categorical"]
            cont_data = emb_data["continuous"]
            
            # Exact shape checks (must match ObsIdx constants and observation_space)
            self.assertEqual(cat_data.shape, (ObsIdx.CATEGORICAL_SIZE,), 
                             f"Categorical shape mismatch for {agent}: expected {(ObsIdx.CATEGORICAL_SIZE,)}, got {cat_data.shape}")
            self.assertEqual(cont_data.shape, (ObsIdx.CONTINUOUS_SIZE,), 
                             f"Continuous shape mismatch for {agent}: expected {(ObsIdx.CONTINUOUS_SIZE,)}, got {cont_data.shape}")
            
            # Dtype checks (must match Box dtypes in observation_space)
            self.assertTrue(np.issubdtype(cat_data.dtype, np.int64), 
                            f"Categorical dtype mismatch for {agent}: expected int64-compatible, got {cat_data.dtype}")
            self.assertTrue(np.issubdtype(cont_data.dtype, np.float32), 
                            f"Continuous dtype mismatch for {agent}: expected float32-compatible, got {cont_data.dtype}")
            
            # Value range checks (categorical should be non-negative ints, continuous 0-1)
            self.assertTrue(np.all(cat_data >= 0), f"Categorical values for {agent} should be >= 0")
            self.assertTrue(np.all(cont_data >= 0.0) and np.all(cont_data <= 1.0), 
                            f"Continuous values for {agent} should be in [0, 1]")

    def test_full_observations_space_compatibility(self):
        """Test that _get_observations in BattleArena produces dict compatible with observation_space."""
        teams = {
            "player": [25, 50, 84, 0, 0, 0, 100, 0],
            "enemy": [129, 10, 150, 0, 0, 0, 10, 0],
        }
        options = {"teams": teams}
        self.arena.reset(options=options)
        
        full_obs = self.arena._get_observations()  # From battle_arena.py
        
        for agent in ["player", "enemy"]:
            agent_obs = full_obs[agent]
            
            # Check required keys
            self.assertIn("categorical", agent_obs)
            self.assertIn("continuous", agent_obs)
            self.assertIn("action_mask", agent_obs)
            
            # Validate against observation_space (simulate space.contains check)
            space = self.arena.observation_space(agent)
            
            # Categorical: shape and dtype
            cat_space = space["categorical"]
            self.assertEqual(agent_obs["categorical"].shape, cat_space.shape, 
                             f"Categorical shape mismatch for {agent}")
            self.assertTrue(np.issubdtype(agent_obs["categorical"].dtype, cat_space.dtype), 
                            f"Categorical dtype mismatch for {agent}")
            self.assertTrue(np.all(agent_obs["categorical"] >= cat_space.low) and 
                            np.all(agent_obs["categorical"] <= cat_space.high), 
                            f"Categorical values out of bounds for {agent}")
            
            # Continuous: shape and dtype
            cont_space = space["continuous"]
            self.assertEqual(agent_obs["continuous"].shape, cont_space.shape, 
                             f"Continuous shape mismatch for {agent}")
            self.assertTrue(np.issubdtype(agent_obs["continuous"].dtype, cont_space.dtype), 
                            f"Continuous dtype mismatch for {agent}")
            self.assertTrue(np.all(agent_obs["continuous"] >= cont_space.low) and 
                            np.all(agent_obs["continuous"] <= cont_space.high), 
                            f"Continuous values out of bounds for {agent}")
            
            # Action mask: shape and dtype
            mask_space = space["action_mask"]
            self.assertEqual(agent_obs["action_mask"].shape, mask_space.shape, 
                             f"Action mask shape mismatch for {agent}")
            self.assertTrue(np.issubdtype(agent_obs["action_mask"].dtype, mask_space.dtype), 
                            f"Action mask dtype mismatch for {agent}")

    def test_embedding_data_edge_cases(self):
        """Test get_embedding_data with edge cases to prevent silent failures."""
        # Test with minimal team (1 Pokémon)
        teams_min = {
            "player": [25, 50, 84, 0, 0, 0, 100, 0],  # 1 Pokémon
            "enemy": [129, 10, 150, 0, 0, 0, 10, 0],
        }
        options = {"teams": teams_min, "team_size": 1}
        self.arena.reset(options=options)
        obs = self.arena.observation_factory.from_game()
        
        for agent in ["player", "enemy"]:
            emb_data = obs.get_embedding_data(agent)
            # Should still produce full shapes (padding with zeros for empty slots)
            self.assertEqual(emb_data["categorical"].shape, (ObsIdx.CATEGORICAL_SIZE,))
            self.assertEqual(emb_data["continuous"].shape, (ObsIdx.CONTINUOUS_SIZE,))
            # Verify no NaN/Inf
            self.assertFalse(np.any(np.isnan(emb_data["continuous"])))
            self.assertFalse(np.any(np.isinf(emb_data["continuous"])))
        
        # Test with completely invalid observation (all zeros = no Pokémon at all)
        broken_obs = Observation(_o={
            "player": np.zeros(ObsIdx.OBS_SIZE, dtype=int),
            "enemy": np.zeros(ObsIdx.OBS_SIZE, dtype=int),
        })
        
        # This should raise ValueError because no valid Pokémon exist
        with self.assertRaises(ValueError) as context:
            broken_obs.get_embedding_data("player")
        self.assertIn("all Pokémon species IDs are 0", str(context.exception))


    def test_action_mask_in_observations(self):
        """Test that action masks in observations have correct shape and dtype."""
        teams = {
            "player": [25, 50, 84, 0, 0, 0, 100, 0],
            "enemy": [129, 10, 150, 0, 0, 0, 10, 0],
        }
        options = {"teams": teams}
        self.arena.reset(options=options)
        
        full_obs = self.arena._get_observations()
        
        for agent in ["player", "enemy"]:
            action_mask = full_obs[agent]["action_mask"]
            
            # Shape check (from ACTION_SPACE_SIZE = 10)
            self.assertEqual(action_mask.shape, (10,), f"Action mask shape mismatch for {agent}")
            
            # Dtype check (float32 as per space)
            self.assertTrue(np.issubdtype(action_mask.dtype, np.float32), 
                            f"Action mask dtype mismatch for {agent}: expected float32, got {action_mask.dtype}")
            
            # Value check (should be 0 or 1, as masks)
            self.assertTrue(np.all((action_mask == 0) | (action_mask == 1)), 
                            f"Action mask values for {agent} should be 0 or 1")


if __name__ == "__main__":
    unittest.main()
