import unittest
import torch
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete

from pkmn_rl_arena.training.models.pkmn_model import PokemonTransformerModel, LayoutConfig, EmbedConfig
from pkmn_rl_arena.env.observation import ObsIdx


class TestPokemonModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_outputs = 10  # Action space size
        
        # Use LayoutConfig sizes to ensure consistency
        self.obs_space = Dict({
            "categorical": Box(low=0, high=1000, shape=(LayoutConfig.CATEGORICAL_SIZE,), dtype=np.int64),  # 324
            "continuous": Box(low=0, high=1, shape=(LayoutConfig.CONTINUOUS_SIZE,), dtype=np.float32),    # 360
            "action_mask": Box(low=0, high=1, shape=(self.num_outputs,), dtype=np.float32),
        })
        
        self.action_space = Discrete(self.num_outputs)
        
        self.model = PokemonTransformerModel(
            obs_space=self.obs_space,
            action_space=self.action_space,
            num_outputs=self.num_outputs,
            model_config={},
            name="test_model"
        )

    def test_forward_pass_dimensions(self):
        """Test if the model accepts input and produces correct output shapes."""
        
        input_dict = {
            "obs": {
                "categorical": torch.randint(0, 20, (self.batch_size, LayoutConfig.CATEGORICAL_SIZE)),
                "continuous": torch.rand(self.batch_size, LayoutConfig.CONTINUOUS_SIZE),  # Use rand for [0,1]
                "action_mask": torch.ones(self.batch_size, self.num_outputs)
            }
        }
        
        logits, state = self.model(input_dict, state=[], seq_lens=None)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_outputs))
        
        value = self.model.value_function()
        self.assertEqual(value.shape, (self.batch_size,))

    def test_action_masking(self):
        """Test if invalid actions are effectively masked out (set to -inf)."""
        
        mask = torch.ones(self.batch_size, self.num_outputs)
        mask[:, 0] = 0  # Action 0 is invalid
        
        input_dict = {
            "obs": {
                "categorical": torch.randint(0, 20, (self.batch_size, LayoutConfig.CATEGORICAL_SIZE)),
                "continuous": torch.rand(self.batch_size, LayoutConfig.CONTINUOUS_SIZE),
                "action_mask": mask
            }
        }
        
        logits, _ = self.model(input_dict, state=[], seq_lens=None)
        
        self.assertTrue(torch.all(logits[:, 0] < -1e5), "Invalid action 0 was not masked correctly!")
        self.assertTrue(torch.all(logits[:, 1] > -1e5), "Valid action 1 was incorrectly masked!")

    def test_embedding_indices_safety(self):
        """Test that max vocabulary indices don't crash the model."""
        
        # Fill with 0s first - 12 pokemon × 27 features = 324
        cat_data = torch.zeros((1, LayoutConfig.CATEGORICAL_SIZE), dtype=torch.long)
        
        # Reshape to access specific slots (12 pokemon, 27 features each)
        cat_reshaped = cat_data.reshape(1, LayoutConfig.TOTAL_POKEMON, LayoutConfig.FEATURES_PER_PKMN)
        
        # Set max values for one pokemon in own team (index 0)
        cat_reshaped[0, 0, LayoutConfig.IDX_SPECIES] = EmbedConfig.VOCAB_SPECIES - 1
        cat_reshaped[0, 0, LayoutConfig.IDX_ITEM] = EmbedConfig.VOCAB_ITEM - 1
        cat_reshaped[0, 0, LayoutConfig.IDX_MOVES_START] = EmbedConfig.VOCAB_MOVE - 1  # Move ID
        
        # Set max values for one pokemon in opponent team (index 6)
        cat_reshaped[0, 6, LayoutConfig.IDX_SPECIES] = EmbedConfig.VOCAB_SPECIES - 1
        cat_reshaped[0, 6, LayoutConfig.IDX_ABILITY] = EmbedConfig.VOCAB_ABILITY - 1
        
        # Flatten back
        input_dict = {
            "obs": {
                "categorical": cat_reshaped.flatten(start_dim=1),
                "continuous": torch.rand(1, LayoutConfig.CONTINUOUS_SIZE),
                "action_mask": torch.ones(1, self.num_outputs)
            }
        }
        
        # Should not raise IndexError
        try:
            self.model(input_dict, state=[], seq_lens=None)
        except IndexError as e:
            self.fail(f"Model crashed with max vocab indices: {e}")

    def test_both_teams_processing(self):
        """Test that model processes both own and opponent team data."""
        
        cat_data = torch.zeros((1, LayoutConfig.CATEGORICAL_SIZE), dtype=torch.long)
        cont_data = torch.zeros((1, LayoutConfig.CONTINUOUS_SIZE), dtype=torch.float32)
        
        # Reshape
        cat_reshaped = cat_data.reshape(1, LayoutConfig.TOTAL_POKEMON, LayoutConfig.FEATURES_PER_PKMN)
        cont_reshaped = cont_data.reshape(1, LayoutConfig.TOTAL_POKEMON, LayoutConfig.CONTINUOUS_PER_PKMN)
        
        # Set species for own team (indices 0-5)
        cat_reshaped[0, 0, LayoutConfig.IDX_SPECIES] = 25   # Pikachu
        cat_reshaped[0, 1, LayoutConfig.IDX_SPECIES] = 1    # Bulbasaur
        
        # Set species for opponent team (indices 6-11)
        cat_reshaped[0, 6, LayoutConfig.IDX_SPECIES] = 129  # Magikarp
        cat_reshaped[0, 7, LayoutConfig.IDX_SPECIES] = 4    # Charmander
        
        # Set is_active (index 0 in continuous features)
        cont_reshaped[0, 0, 0] = 1.0  # Own active
        cont_reshaped[0, 6, 0] = 1.0  # Opponent active
        
        input_dict = {
            "obs": {
                "categorical": cat_reshaped.flatten(start_dim=1),
                "continuous": cont_reshaped.flatten(start_dim=1),
                "action_mask": torch.ones(1, self.num_outputs)
            }
        }
        
        # Should process without errors
        logits, _ = self.model(input_dict, state=[], seq_lens=None)
        self.assertEqual(logits.shape, (1, self.num_outputs))

    def test_padding_mask_sparse_teams(self):
        """Test that empty pokemon slots (species=0) are properly handled."""
        
        cat_data = torch.zeros((1, LayoutConfig.CATEGORICAL_SIZE), dtype=torch.long)
        cont_data = torch.zeros((1, LayoutConfig.CONTINUOUS_SIZE), dtype=torch.float32)
        
        cat_reshaped = cat_data.reshape(1, LayoutConfig.TOTAL_POKEMON, LayoutConfig.FEATURES_PER_PKMN)
        cont_reshaped = cont_data.reshape(1, LayoutConfig.TOTAL_POKEMON, LayoutConfig.CONTINUOUS_PER_PKMN)
        
        # Only 1 pokemon per team (rest are empty slots with species=0)
        cat_reshaped[0, 0, LayoutConfig.IDX_SPECIES] = 25   # Pikachu only
        cat_reshaped[0, 6, LayoutConfig.IDX_SPECIES] = 129  # Magikarp only
        
        cont_reshaped[0, 0, 0] = 1.0  # Own active
        cont_reshaped[0, 6, 0] = 1.0  # Opponent active
        
        input_dict = {
            "obs": {
                "categorical": cat_reshaped.flatten(start_dim=1),
                "continuous": cont_reshaped.flatten(start_dim=1),
                "action_mask": torch.ones(1, self.num_outputs)
            }
        }
        
        # Should handle sparse teams without NaN
        logits, _ = self.model(input_dict, state=[], seq_lens=None)
        self.assertFalse(torch.any(torch.isnan(logits)), "Model produced NaN with sparse teams")
        # Valid actions should not be -inf
        self.assertTrue(torch.all(logits[:, 1:] > -1e9), "Model produced unexpected -inf for valid actions")

    def test_value_function(self):
        """Test that value function returns correct shape."""
        
        input_dict = {
            "obs": {
                "categorical": torch.randint(0, 20, (self.batch_size, LayoutConfig.CATEGORICAL_SIZE)),
                "continuous": torch.rand(self.batch_size, LayoutConfig.CONTINUOUS_SIZE),
                "action_mask": torch.ones(self.batch_size, self.num_outputs)
            }
        }
        
        self.model(input_dict, state=[], seq_lens=None)
        value = self.model.value_function()
        
        self.assertEqual(value.shape, (self.batch_size,))
        self.assertFalse(torch.any(torch.isnan(value)), "Value function produced NaN")


if __name__ == "__main__":
    unittest.main()