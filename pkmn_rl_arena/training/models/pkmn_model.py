import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

class LayoutConfig:
    """Indices for the reshaped (6, Features) tensor"""
    TEAM_SIZE = 6
    FEATURES_PER_PKMN = 27 # From ObsIdx.CATEGORICAL_PER_PKMN
    
    # Indices within the 27 features
    IDX_SPECIES = 0
    IDX_ABILITY = 1
    IDX_TYPE1 = 2
    IDX_TYPE2 = 3
    IDX_ITEM = 4
    IDX_STATUS = 5
    IDX_STATUS2 = 6
    # Moves start at index 7. Each move has 5 categorical fields.
    IDX_MOVES_START = 7 

class EmbedConfig:
    """Vocabulary sizes for embeddings"""
    VOCAB_SPECIES = 412
    VOCAB_ITEM = 501
    VOCAB_MOVE = 355
    VOCAB_TYPE = 19
    VOCAB_ABILITY = 101 # Approx
    VOCAB_STATUS = 9
    
    # Embedding Dimensions
    DIM_SPECIES = 32
    DIM_ITEM = 16
    DIM_MOVE = 16
    DIM_TYPE = 8
    DIM_ABILITY = 8
    DIM_STATUS = 4

class PokemonTransformerModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.num_outputs = num_outputs
        
        # --- Embeddings ---
        self.embed_species = nn.Embedding(EmbedConfig.VOCAB_SPECIES, EmbedConfig.DIM_SPECIES)
        self.embed_item = nn.Embedding(EmbedConfig.VOCAB_ITEM, EmbedConfig.DIM_ITEM)
        self.embed_move = nn.Embedding(EmbedConfig.VOCAB_MOVE, EmbedConfig.DIM_MOVE)
        self.embed_type = nn.Embedding(EmbedConfig.VOCAB_TYPE, EmbedConfig.DIM_TYPE)
        self.embed_ability = nn.Embedding(EmbedConfig.VOCAB_ABILITY, EmbedConfig.DIM_ABILITY)
        self.embed_status = nn.Embedding(EmbedConfig.VOCAB_STATUS, EmbedConfig.DIM_STATUS)

        # Calculate size of concatenated embeddings per Pokemon
        # Species(1) + Ability(1) + Type(2) + Item(1) + Status(2) + Moves(4 * 1) 
        # Note: We only embed the Move ID. Other move attributes (target, etc) are treated as raw ints or ignored for now for simplicity, 
        # or you can add embeddings for them too.
        
        self.emb_dim_total = (
            EmbedConfig.DIM_SPECIES + 
            EmbedConfig.DIM_ABILITY + 
            (2 * EmbedConfig.DIM_TYPE) + 
            EmbedConfig.DIM_ITEM + 
            (2 * EmbedConfig.DIM_STATUS) + 
            (4 * EmbedConfig.DIM_MOVE) # 4 moves
        )

        # --- Continuous Features ---
        # 180 floats total / 6 pokemon = 30 floats per pokemon
        self.continuous_dim_per_pkmn = 30
        self.continuous_proj = nn.Linear(self.continuous_dim_per_pkmn, 64)

        # --- Transformer ---
        self.model_dim = self.emb_dim_total + 64 # Embeddings + Projected Continuous
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- Heads ---
        # We pool the 6 pokemon into 1 vector
        self.policy_head = nn.Linear(self.model_dim, num_outputs)
        self.value_head = nn.Linear(self.model_dim, 1)
        
        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # 1. Unpack Observation
        obs = input_dict["obs"]
        # Ray flattens dicts sometimes, but if we use the Dict space correctly it keeps structure
        # If using 'custom_model', input_dict['obs'] is usually the raw dict if 'flatten_observations' is False in config
        
        cat_data = obs["categorical"].long() # (Batch, 162)
        cont_data = obs["continuous"].float() # (Batch, 180)
        action_mask = obs["action_mask"].float() # (Batch, 10)

        batch_size = cat_data.shape[0]

        # 2. Reshape
        # Categorical: (Batch, 6, 27)
        cat_reshaped = cat_data.reshape(batch_size, LayoutConfig.TEAM_SIZE, LayoutConfig.FEATURES_PER_PKMN)
        # Continuous: (Batch, 6, 30)
        cont_reshaped = cont_data.reshape(batch_size, LayoutConfig.TEAM_SIZE, self.continuous_dim_per_pkmn)

        # 3. Embeddings
        # Helper to gather and flatten
        def get_emb(idx, layer):
            return layer(torch.clamp(cat_reshaped[:, :, idx], 0, layer.num_embeddings - 1))

        e_species = get_emb(LayoutConfig.IDX_SPECIES, self.embed_species)
        e_ability = get_emb(LayoutConfig.IDX_ABILITY, self.embed_ability)
        e_type1 = get_emb(LayoutConfig.IDX_TYPE1, self.embed_type)
        e_type2 = get_emb(LayoutConfig.IDX_TYPE2, self.embed_type)
        e_item = get_emb(LayoutConfig.IDX_ITEM, self.embed_item)
        e_stat1 = get_emb(LayoutConfig.IDX_STATUS, self.embed_status)
        e_stat2 = get_emb(LayoutConfig.IDX_STATUS2, self.embed_status)

        # Moves: We have 4 moves. Their IDs are at specific offsets.
        # Move 1 ID is at IDX_MOVES_START + 0
        # Move 2 ID is at IDX_MOVES_START + 5 (stride is 5 categorical fields per move)
        move_stride = 5
        m1_id = get_emb(LayoutConfig.IDX_MOVES_START, self.embed_move)
        m2_id = get_emb(LayoutConfig.IDX_MOVES_START + move_stride, self.embed_move)
        m3_id = get_emb(LayoutConfig.IDX_MOVES_START + (2*move_stride), self.embed_move)
        m4_id = get_emb(LayoutConfig.IDX_MOVES_START + (3*move_stride), self.embed_move)

        # Concatenate all embeddings
        # Shape: (Batch, 6, Total_Emb_Dim)
        embeddings = torch.cat([
            e_species, e_ability, e_type1, e_type2, e_item, e_stat1, e_stat2,
            m1_id, m2_id, m3_id, m4_id
        ], dim=2)

        # 4. Process Continuous
        cont_proj = self.continuous_proj(cont_reshaped) # (Batch, 6, 64)

        # 5. Combine
        x = torch.cat([embeddings, cont_proj], dim=2) # (Batch, 6, Model_Dim)

        # 6. Transformer
        # Masking: We might want to mask empty pokemon slots (species 0), but for now let's process all
        x = self.transformer(x) # (Batch, 6, Model_Dim)

        # 7. Pooling
        # Max pooling over the team dimension (dim 1)
        # This creates a permutation-invariant representation of the team
        pooled, _ = torch.max(x, dim=1) # (Batch, Model_Dim)

        # 8. Heads
        logits = self.policy_head(pooled)
        self._value_out = self.value_head(pooled).squeeze(1)

        # 9. Action Masking
        # Set logits of invalid actions to a very large negative number
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e10)
        masked_logits = logits + inf_mask

        return masked_logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out