import numpy as np
import gymnasium as gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_torch
from dataclasses import dataclass

torch, nn = try_import_torch()

@dataclass(frozen=True)
class EmbedConfig:
    """Configuration for Embedding Vocabulary Sizes and Dimensions"""
    # Vocab sizes (Input IDs)
    VOCAB_SPECIES = 500
    VOCAB_ITEM = 500
    VOCAB_MOVE = 500
    VOCAB_TYPE = 20
    VOCAB_STATUS = 32
    VOCAB_ABILITY = 300
    VOCAB_MOVE_EFFECT = 300
    VOCAB_MISC = 100

    # Embedding dimensions (Output Vectors)
    DIM_SPECIES = 32
    DIM_ITEM = 16
    DIM_MOVE = 16
    DIM_TYPE = 8
    DIM_STATUS = 8
    DIM_STATUS2 = 4
    DIM_ABILITY = 16
    DIM_MOVE_EFFECT = 16
    DIM_MISC = 4

@dataclass(frozen=True)
class LayoutConfig:
    """Configuration for Input Data Layout"""
    TEAM_SIZE = 6
    FEATURES_PER_PKMN = 27
    MOVES_PER_PKMN = 4
    MOVE_ATTRS_STRIDE = 5 # ID, Effect, Type, Target, Flags
    
    # Indices in categorical vector
    IDX_SPECIES = 0
    IDX_ABILITY = 1
    IDX_TYPE1 = 2
    IDX_TYPE2 = 3
    IDX_ITEM = 4
    IDX_STATUS = 5
    IDX_STATUS2 = 6
    IDX_MOVES_START = 7
    
    # Continuous data size
    CONT_SIZE_PER_PKMN = 30

class PokemonTransformerModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Network Architecture Hyperparameters
        hidden_dim_1 = 1024
        hidden_dim_2 = 512
        hidden_dim_3 = 256

        # --- 1. Define Embedding Layers ---
        self.species_embed = nn.Embedding(EmbedConfig.VOCAB_SPECIES, EmbedConfig.DIM_SPECIES)
        self.item_embed = nn.Embedding(EmbedConfig.VOCAB_ITEM, EmbedConfig.DIM_ITEM)
        self.move_embed = nn.Embedding(EmbedConfig.VOCAB_MOVE, EmbedConfig.DIM_MOVE)
        self.type_embed = nn.Embedding(EmbedConfig.VOCAB_TYPE, EmbedConfig.DIM_TYPE)
        self.status_embed = nn.Embedding(EmbedConfig.VOCAB_STATUS, EmbedConfig.DIM_STATUS)
        self.ability_embed = nn.Embedding(EmbedConfig.VOCAB_ABILITY, EmbedConfig.DIM_ABILITY)
        self.move_effect_embed = nn.Embedding(EmbedConfig.VOCAB_MOVE_EFFECT, EmbedConfig.DIM_MOVE_EFFECT)
        self.misc_embed = nn.Embedding(EmbedConfig.VOCAB_MISC, EmbedConfig.DIM_MISC)

        # --- 2. Calculate Dimensions ---
        # Base stats embedding size sum
        base_stats_emb_size = (
            EmbedConfig.DIM_SPECIES + 
            EmbedConfig.DIM_ABILITY + 
            EmbedConfig.DIM_TYPE + 
            EmbedConfig.DIM_TYPE + 
            EmbedConfig.DIM_ITEM + 
            EmbedConfig.DIM_STATUS +
            EmbedConfig.DIM_STATUS2
        )

        # Move embedding size sum (per single move)
        single_move_emb_size = (
            EmbedConfig.DIM_MOVE + 
            EmbedConfig.DIM_MOVE_EFFECT + 
            EmbedConfig.DIM_TYPE + 
            EmbedConfig.DIM_MISC + 
            EmbedConfig.DIM_MISC
        )

        self.emb_size_per_pkmn = base_stats_emb_size + (LayoutConfig.MOVES_PER_PKMN * single_move_emb_size)
        self.total_per_pkmn = self.emb_size_per_pkmn + LayoutConfig.CONT_SIZE_PER_PKMN
        self.total_input_size = self.total_per_pkmn * LayoutConfig.TEAM_SIZE

        # --- 3. Network Architecture ---
        self.pre_net = nn.Sequential(
            nn.Linear(self.total_input_size, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.ReLU()
        )

        self.action_head = nn.Linear(hidden_dim_3, num_outputs)
        self.value_head = nn.Linear(hidden_dim_3, 1)
        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        
        cat_data = obs["categorical"].long()
        cont_data = obs["continuous"].float()
        action_mask = obs["action_mask"]
        
        batch_size = cat_data.shape[0]

        # Reshape to [Batch, 6 Pokemon, 26 Features]
        cat_reshaped = cat_data.reshape(batch_size, LayoutConfig.TEAM_SIZE, LayoutConfig.FEATURES_PER_PKMN)

        # --- Apply Embeddings Slice by Slice ---
        # Base Stats
        e_species = self.species_embed(cat_reshaped[:, :, LayoutConfig.IDX_SPECIES])
        e_ability = self.ability_embed(cat_reshaped[:, :, LayoutConfig.IDX_ABILITY])
        e_type1 = self.type_embed(cat_reshaped[:, :, LayoutConfig.IDX_TYPE1])
        e_type2 = self.type_embed(cat_reshaped[:, :, LayoutConfig.IDX_TYPE2])
        e_item = self.item_embed(cat_reshaped[:, :, LayoutConfig.IDX_ITEM])
        e_status = self.status_embed(cat_reshaped[:, :, LayoutConfig.IDX_STATUS])
        e_status2 = self.misc_embed(cat_reshaped[:, :, LayoutConfig.IDX_STATUS2])
        # Moves
        move_embeddings = []
        for i in range(LayoutConfig.MOVES_PER_PKMN):
            base = LayoutConfig.IDX_MOVES_START + (i * LayoutConfig.MOVE_ATTRS_STRIDE)
            # [ID, Effect, Type, Target, Flags]
            m_id = self.move_embed(cat_reshaped[:, :, base])
            m_eff = self.move_effect_embed(cat_reshaped[:, :, base+1])
            m_type = self.type_embed(cat_reshaped[:, :, base+2])
            m_targ = self.misc_embed(cat_reshaped[:, :, base+3])
            m_flag = self.misc_embed(cat_reshaped[:, :, base+4])
            move_embeddings.extend([m_id, m_eff, m_type, m_targ, m_flag])

        # Concatenate all embeddings
        all_embeds = [e_species, e_ability, e_type1, e_type2, e_item, e_status, e_status2] + move_embeddings
        
        # Concatenate along feature dim (2)
        embed_out = torch.cat(all_embeds, dim=2)
        
        # Flatten Pokemon dim -> [Batch, Total_Embed_Size]
        embed_flat = embed_out.view(batch_size, -1)

        # Concatenate with continuous data
        x = torch.cat([embed_flat, cont_data], dim=1)

        # --- Pass through Network ---
        features = self.pre_net(x)
        
        logits = self.action_head(features)
        self._last_value = self.value_head(features).squeeze(1)

        # --- Action Masking ---
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e10)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        return self._last_value