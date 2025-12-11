import unittest
import torch
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete

from pkmn_rl_arena.training.models.pkmn_model import PokemonTransformerModel, LayoutConfig, EmbedConfig
from pkmn_rl_arena.env.observation import ObsIdx

class TestPokemonModel(unittest.TestCase):
    def setUp(self):
        # 1. Define the Observation Space exactly as Ray RLLib sees it
        # Note: Ray flattens Dict spaces, but we pass the raw dict structure in our test
        # to simulate the 'input_dict' wrapper.
        
        self.batch_size = 4
        self.num_outputs = 10 # Action space size
        
        self.obs_space = Dict({
            "categorical": Box(low=0, high=1000, shape=(ObsIdx.CATEGORICAL_SIZE,), dtype=np.int32),
            "continuous": Box(low=0, high=1, shape=(ObsIdx.CONTINUOUS_SIZE,), dtype=np.float32),
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
                "categorical": torch.randint(0, 20, (self.batch_size, ObsIdx.CATEGORICAL_SIZE)),
                "continuous": torch.randn(self.batch_size, ObsIdx.CONTINUOUS_SIZE),
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
        mask[:, 0] = 0 # Action 0 is invalid
        
        input_dict = {
            "obs": {
                "categorical": torch.randint(0, 20, (self.batch_size, ObsIdx.CATEGORICAL_SIZE)),
                "continuous": torch.randn(self.batch_size, ObsIdx.CONTINUOUS_SIZE),
                "action_mask": mask
            }
        }
        
        logits, _ = self.model(input_dict, state=[], seq_lens=None)
        
        self.assertTrue(torch.all(logits[:, 0] < -1e5), "Invalid action 0 was not masked correctly!")
        
        self.assertTrue(torch.all(logits[:, 1] > -1e5), "Valid action 1 was incorrectly masked!")

    def test_embedding_indices_safety(self):
        """Test that max vocabulary indices don't crash the model."""
        
        # Fill with 0s first
        cat_data = torch.zeros((1, ObsIdx.CATEGORICAL_SIZE), dtype=torch.long)
        
        # Reshape to access specific slots
        cat_reshaped = cat_data.reshape(1, LayoutConfig.TEAM_SIZE, LayoutConfig.FEATURES_PER_PKMN)
        
        # Set max values for one pokemon
        cat_reshaped[0, 0, LayoutConfig.IDX_SPECIES] = EmbedConfig.VOCAB_SPECIES - 1
        cat_reshaped[0, 0, LayoutConfig.IDX_ITEM] = EmbedConfig.VOCAB_ITEM - 1
        cat_reshaped[0, 0, LayoutConfig.IDX_MOVES_START] = EmbedConfig.VOCAB_MOVE - 1 # Move ID
        
        # Flatten back
        input_dict = {
            "obs": {
                "categorical": cat_reshaped.flatten(start_dim=1),
                "continuous": torch.randn(1, ObsIdx.CONTINUOUS_SIZE),
                "action_mask": torch.ones(1, self.num_outputs)
            }
        }
        
        # Should not raise IndexError
        try:
            self.model(input_dict, state=[], seq_lens=None)
        except IndexError as e:
            self.fail(f"Model crashed with max vocab indices: {e}")

if __name__ == "__main__":
    unittest.main()