import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class LayoutConfig:
    """Layout for (12, Features) tensor - 6 own + 6 opponent pokemon"""
    TEAM_SIZE = 6
    TOTAL_POKEMON = 12  # Both teams
    
    # Categorical: 7 base + 20 move = 27 per pokemon (includes status_2)
    FEATURES_PER_PKMN = 27
    CATEGORICAL_SIZE = TOTAL_POKEMON * FEATURES_PER_PKMN  # 324
    
    # Continuous: 10 base + 20 move = 30 per pokemon
    CONTINUOUS_PER_PKMN = 30
    CONTINUOUS_SIZE = TOTAL_POKEMON * CONTINUOUS_PER_PKMN  # 360
    
    # Indices within categorical (27 features per pokemon)
    IDX_SPECIES = 0
    IDX_ABILITY = 1
    IDX_TYPE1 = 2
    IDX_TYPE2 = 3
    IDX_ITEM = 4
    IDX_STATUS = 5
    IDX_STATUS2 = 6
    IDX_MOVES_START = 7  # Move features start here
    
    # Move categorical stride (5 features per move)
    MOVE_CAT_STRIDE = 5


class EmbedConfig:
    """Vocabulary sizes and embedding dimensions"""
    VOCAB_SPECIES = 412
    VOCAB_ITEM = 501
    VOCAB_MOVE = 400
    VOCAB_TYPE = 20
    VOCAB_ABILITY = 151
    VOCAB_STATUS = 257
    VOCAB_EFFECT = 301
    VOCAB_TARGET = 21
    
    DIM_SPECIES = 32
    DIM_ITEM = 16
    DIM_MOVE = 24
    DIM_TYPE = 8
    DIM_ABILITY = 12
    DIM_STATUS = 8
    DIM_EFFECT = 8
    DIM_TARGET = 4


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
        self.embed_effect = nn.Embedding(EmbedConfig.VOCAB_EFFECT, EmbedConfig.DIM_EFFECT)
        self.embed_target = nn.Embedding(EmbedConfig.VOCAB_TARGET, EmbedConfig.DIM_TARGET)

        # Per-pokemon base embedding dim (includes status + status2)
        self.emb_dim_pokemon_base = (
            EmbedConfig.DIM_SPECIES +
            EmbedConfig.DIM_ABILITY +
            2 * EmbedConfig.DIM_TYPE +  # type1 + type2
            EmbedConfig.DIM_ITEM +
            2 * EmbedConfig.DIM_STATUS  # status + status2
        )  # 32 + 12 + 16 + 16 + 16 = 92
        
        # Per-move embedding dim
        self.emb_dim_move = (
            EmbedConfig.DIM_MOVE +
            EmbedConfig.DIM_EFFECT +
            EmbedConfig.DIM_TYPE +
            EmbedConfig.DIM_TARGET
        )  # 24 + 8 + 8 + 4 = 44

        # --- Continuous Processing ---
        self.continuous_base_dim = 10  # is_active, stats(5), hp, max_hp, lvl, friendship
        self.continuous_move_dim = 5   # pp, power, acc, priority, secondary
        self.continuous_dim_per_pkmn = LayoutConfig.CONTINUOUS_PER_PKMN  # 30
        
        self.base_continuous_proj = nn.Linear(self.continuous_base_dim, 32)
        self.move_continuous_proj = nn.Linear(self.continuous_move_dim, 16)

        # --- Move Aggregation ---
        self.move_total_dim = self.emb_dim_move + 16  # 44 + 16 = 60
        self.move_attention = nn.MultiheadAttention(self.move_total_dim, num_heads=4, batch_first=True)
        self.move_aggregator = nn.Linear(self.move_total_dim, 64)

        # --- Pokemon Representation ---
        self.pokemon_dim = self.emb_dim_pokemon_base + 32 + 64  # 92 + 32 + 64 = 188
        
        # Position embedding: 0=own_active, 1-5=own_bench, 6=opp_active, 7-11=opp_bench
        self.position_embed = nn.Embedding(12, 16)
        
        # Team embedding: 0=own, 1=opponent
        self.team_embed = nn.Embedding(2, 8)
        
        # --- Transformer ---
        self.model_dim = self.pokemon_dim + 16 + 8  # + position + team = 212
        
        # Use standard transformer without nested tensors to avoid padding mask issues
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)

        # --- Output Heads ---
        self.policy_head = nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._value_out = None

    def _safe_embed(self, data, layer):
        """Safely embed with clamping to vocabulary size."""
        return layer(torch.clamp(data, 0, layer.num_embeddings - 1))

    def _embed_pokemon_base(self, cat_reshaped):
        """Embed base pokemon features for all 12 pokemon."""
        # cat_reshaped: (batch, 12, 27)
        
        e_species = self._safe_embed(cat_reshaped[:, :, LayoutConfig.IDX_SPECIES], self.embed_species)
        e_ability = self._safe_embed(cat_reshaped[:, :, LayoutConfig.IDX_ABILITY], self.embed_ability)
        e_type1 = self._safe_embed(cat_reshaped[:, :, LayoutConfig.IDX_TYPE1], self.embed_type)
        e_type2 = self._safe_embed(cat_reshaped[:, :, LayoutConfig.IDX_TYPE2], self.embed_type)
        e_item = self._safe_embed(cat_reshaped[:, :, LayoutConfig.IDX_ITEM], self.embed_item)
        e_status = self._safe_embed(cat_reshaped[:, :, LayoutConfig.IDX_STATUS], self.embed_status)
        e_status2 = self._safe_embed(cat_reshaped[:, :, LayoutConfig.IDX_STATUS2], self.embed_status)
        
        return torch.cat([e_species, e_ability, e_type1, e_type2, e_item, e_status, e_status2], dim=2)

    def _embed_moves(self, cat_reshaped, cont_reshaped):
        """Embed and aggregate moves for all 12 pokemon."""
        batch_size = cat_reshaped.shape[0]
        num_pokemon = LayoutConfig.TOTAL_POKEMON
        
        all_move_repr = []
        
        for pkmn_idx in range(num_pokemon):
            move_reprs = []
            for move_idx in range(4):
                # Categorical indices: starts at 7, stride of 5
                cat_base = LayoutConfig.IDX_MOVES_START + move_idx * LayoutConfig.MOVE_CAT_STRIDE
                
                move_id = cat_reshaped[:, pkmn_idx, cat_base]       # move_id
                effect = cat_reshaped[:, pkmn_idx, cat_base + 1]    # effect
                move_type = cat_reshaped[:, pkmn_idx, cat_base + 2] # type
                target = cat_reshaped[:, pkmn_idx, cat_base + 3]    # target
                # flags at cat_base + 4 - not embedded (bitfield)
                
                e_move = self._safe_embed(move_id, self.embed_move)
                e_effect = self._safe_embed(effect, self.embed_effect)
                e_type = self._safe_embed(move_type, self.embed_type)
                e_target = self._safe_embed(target, self.embed_target)
                
                # Continuous: base_cont is 10, then 5 per move
                cont_base = self.continuous_base_dim + move_idx * self.continuous_move_dim
                move_cont = cont_reshaped[:, pkmn_idx, cont_base:cont_base + self.continuous_move_dim]
                move_cont_proj = self.move_continuous_proj(move_cont)
                
                move_repr = torch.cat([e_move, e_effect, e_type, e_target, move_cont_proj], dim=1)
                move_reprs.append(move_repr)
            
            moves_stacked = torch.stack(move_reprs, dim=1)  # (batch, 4, move_dim)
            moves_attn, _ = self.move_attention(moves_stacked, moves_stacked, moves_stacked)
            moves_pooled = moves_attn.mean(dim=1)
            moves_agg = self.move_aggregator(moves_pooled)
            all_move_repr.append(moves_agg)
        
        return torch.stack(all_move_repr, dim=1)  # (batch, 12, 64)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        
        cat_data = obs["categorical"].long()
        cont_data = obs["continuous"].float()
        action_mask = obs["action_mask"].float()

        batch_size = cat_data.shape[0]

        # Reshape for 12 pokemon (6 own + 6 opponent)
        cat_reshaped = cat_data.reshape(batch_size, LayoutConfig.TOTAL_POKEMON, LayoutConfig.FEATURES_PER_PKMN)
        cont_reshaped = cont_data.reshape(batch_size, LayoutConfig.TOTAL_POKEMON, self.continuous_dim_per_pkmn)

        # 1. Embed base pokemon features
        pokemon_base_emb = self._embed_pokemon_base(cat_reshaped)
        
        # 2. Project base continuous features
        base_cont = cont_reshaped[:, :, :self.continuous_base_dim]
        base_cont_proj = self.base_continuous_proj(base_cont)
        
        # 3. Embed and aggregate moves
        moves_agg = self._embed_moves(cat_reshaped, cont_reshaped)
        
        # 4. Combine pokemon representation
        pokemon_repr = torch.cat([pokemon_base_emb, base_cont_proj, moves_agg], dim=2)
        
        # 5. Position embedding (vectorized - no Python loops)
        is_active = cont_reshaped[:, :, 0]  # (batch, 12)
        
        # Default positions: 0-5 for own team, 6-11 for opponent
        default_positions = torch.arange(LayoutConfig.TOTAL_POKEMON, device=cat_data.device)
        positions = default_positions.unsqueeze(0).expand(batch_size, -1)
        
        pos_emb = self.position_embed(positions)
        
        # 6. Team embedding (own=0, opponent=1)
        team_ids = torch.zeros(batch_size, LayoutConfig.TOTAL_POKEMON, dtype=torch.long, device=cat_data.device)
        team_ids[:, 6:] = 1  # Opponent team
        team_emb = self.team_embed(team_ids)
        
        # 7. Final representation
        x = torch.cat([pokemon_repr, pos_emb, team_emb], dim=2)
        
        # 8. Transformer - handle padding mask safely
        species_ids = cat_reshaped[:, :, LayoutConfig.IDX_SPECIES]
        padding_mask = (species_ids == 0)
        
        # CRITICAL FIX: Ensure at least one token is not masked per batch
        # If all tokens are masked (dummy data), don't use the mask at all
        all_masked = padding_mask.all(dim=1, keepdim=True)  # (batch, 1)
        
        if all_masked.any():
            # For batches where all are masked, set first token to unmasked
            padding_mask = padding_mask.clone()
            padding_mask[:, 0] = padding_mask[:, 0] & ~all_masked.squeeze(1)
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # 9. Pool - use mean over all tokens, weighted by validity
        # Create weight mask: 1 for valid tokens, 0 for padded
        valid_mask = (~padding_mask).float().unsqueeze(-1)  # (batch, 12, 1)
        
        # Weighted mean pooling over own team (first 6 pokemon)
        own_repr = x[:, :6, :]  # (batch, 6, model_dim)
        own_valid = valid_mask[:, :6, :]  # (batch, 6, 1)
        
        # Weight by is_active if available, otherwise by validity
        own_active_weight = is_active[:, :6].unsqueeze(-1)  # (batch, 6, 1)
        combined_weight = own_active_weight * own_valid
        
        # Sum of weights
        weight_sum = combined_weight.sum(dim=1, keepdim=True) + 1e-8  # (batch, 1, 1)
        
        # Weighted pooling
        pooled = (own_repr * combined_weight).sum(dim=1) / weight_sum.squeeze(1)  # (batch, model_dim)
        
        # Fallback: if no valid active pokemon, use mean of own team
        has_valid = (combined_weight.sum(dim=1) > 0).float()  # (batch, 1)
        fallback = own_repr.mean(dim=1)  # (batch, model_dim)
        
        pooled = pooled * has_valid + fallback * (1 - has_valid)

        # 10. Output heads
        logits = self.policy_head(pooled)
        self._value_out = self.value_head(pooled).squeeze(1)

        # 11. Action masking
        inf_mask = torch.where(
            action_mask > 0.5,
            torch.zeros_like(action_mask),
            torch.full_like(action_mask, -1e9)
        )
        masked_logits = logits + inf_mask

        return masked_logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out