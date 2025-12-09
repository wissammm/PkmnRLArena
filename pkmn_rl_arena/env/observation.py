from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from numpy import typing as npt
import pandas as pd
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena.env.pkmn_team_factory import DataSize 

from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.pkmn_team_factory import PkmnTeamFactory

AgentObs = npt.NDArray[int]

@dataclass(frozen=True)
class ObsIdx:
    TEAM_SIZE = 6
    MAX_PKMN_MOVES = 4
    NB_STATS_PER_PKMN = 5  # ATK, DEF, SPEED, SPATK, SPDEF
    MOVE_ATTRIBUTES_PER_MOVE = 10  # id, pp, effect, power, type, accuracy, priority, secondaryEffectChance, target, flags
    NB_DATA_PKMN = 20 + (MAX_PKMN_MOVES * MOVE_ATTRIBUTES_PER_MOVE)  # 20 base fields + move data
    OBS_SIZE = 6 * NB_DATA_PKMN  
    CATEGORICAL_PER_PKMN = 26
    CATEGORICAL_SIZE = TEAM_SIZE * CATEGORICAL_PER_PKMN # 156
    CONTINUOUS_PER_PKMN = 30
    CONTINUOUS_SIZE = TEAM_SIZE * CONTINUOUS_PER_PKMN # 180

    RAW_DATA = {
       "species": 0,
        "is_active": 1,

        "stats_begin": 2,
        "stats_end": 7,  

        "ability": 7,
        "type_1": 8,
        "type_2": 9,

        "HP": 10,
        "level": 11,
        "friendship": 12,
        "max_HP": 13,
        "held_item": 14,
        "pp_bonuses": 15,
        "personality": 16,
        "status": 17,
        "status_2": 18,
        "status_3": 19,
        "moves_begin": 20,
        "moves_end": 20 + (MAX_PKMN_MOVES * MOVE_ATTRIBUTES_PER_MOVE),
        "move_id_offset": 0,
        "pp_offset": 1,
        "effect_offset": 2,
        "power_offset": 3,
        "type_offset": 4,
        "accuracy_offset": 5,
        "priority_offset": 6,
        "secondaryEffectChance_offset": 7,
        "target_offset": 8,
        "flags_offset": 9,
        
        "move_slot_stride": MOVE_ATTRIBUTES_PER_MOVE,
    }

@dataclass
class Observation:
    """
    This class is a wrapper around the observation type Dict[str, Agent]
    It mainly serves as an index helper.
    """

    _o: Dict[str, AgentObs]

    def agent(self, a: str) -> npt.NDArray[int]:
        return self._o[a]

    def get_agent_data(self, agent: str) -> AgentObs:
        return self._o[agent]
    
    def get_normalized_agent_data(self, agent: str) -> AgentObs:
        if agent not in self._o:
            raise ValueError(f"Invalid agent name, must be in {self._o.keys()}, got {agent}.")
            
        raw_data = self._o[agent].astype(float)
        pokemon_data = raw_data.reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
        
        # Constants for normalization
        MAX_SPECIES = 411
        MAX_STAT = 550
        MAX_HP = 550
        MAX_LEVEL = 100
        MAX_FRIENDSHIP = 255
        MAX_ITEM = 500
        MAX_ABILITY = 100
        MAX_TYPE = 18
        MAX_MOVE_ID = 354
        MAX_PP = 40
        MAX_POWER = 255
        MAX_ACCURACY = 100
        MAX_STATUS_1 = 8
        MAX_STATUS_2 = 8
        
        # Normalize Base Fields
        pokemon_data[:, ObsIdx.RAW_DATA["species"]] /= MAX_SPECIES
        # is_active is already 0/1
        pokemon_data[:, ObsIdx.RAW_DATA["stats_begin"]:ObsIdx.RAW_DATA["stats_end"]] /= MAX_STAT
        pokemon_data[:, ObsIdx.RAW_DATA["ability"]] /= MAX_ABILITY
        pokemon_data[:, ObsIdx.RAW_DATA["type_1"]] /= MAX_TYPE
        pokemon_data[:, ObsIdx.RAW_DATA["type_2"]] /= MAX_TYPE
        pokemon_data[:, ObsIdx.RAW_DATA["HP"]] /= MAX_HP
        pokemon_data[:, ObsIdx.RAW_DATA["max_HP"]] /= MAX_HP
        pokemon_data[:, ObsIdx.RAW_DATA["level"]] /= MAX_LEVEL
        pokemon_data[:, ObsIdx.RAW_DATA["friendship"]] /= MAX_FRIENDSHIP
        pokemon_data[:, ObsIdx.RAW_DATA["held_item"]] /= MAX_ITEM
        pokemon_data[:, ObsIdx.RAW_DATA["pp_bonuses"]] /= 255.0
        
        # Personality: Random noise, set to 0 for normalized obs
        pokemon_data[:, ObsIdx.RAW_DATA["personality"]] = 0.0
        
        # Status: Bitfield. Normalize by max uint32 to keep in [0,1]
        pokemon_data[:, ObsIdx.RAW_DATA["status"]] /= MAX_STATUS_1
        pokemon_data[:, ObsIdx.RAW_DATA["status_2"]] /= MAX_STATUS_2
        pokemon_data[:, ObsIdx.RAW_DATA["status_3"]] /= MAX_STATUS_2

        # Normalize Moves
        moves_data = pokemon_data[:, ObsIdx.RAW_DATA["moves_begin"]:].reshape((DataSize.PARTY_SIZE, ObsIdx.MAX_PKMN_MOVES, ObsIdx.MOVE_ATTRIBUTES_PER_MOVE))
        
        moves_data[:, :, ObsIdx.RAW_DATA["move_id_offset"]] /= MAX_MOVE_ID
        moves_data[:, :, ObsIdx.RAW_DATA["pp_offset"]] /= MAX_PP
        moves_data[:, :, ObsIdx.RAW_DATA["power_offset"]] /= MAX_POWER
        moves_data[:, :, ObsIdx.RAW_DATA["type_offset"]] /= MAX_TYPE
        moves_data[:, :, ObsIdx.RAW_DATA["accuracy_offset"]] /= MAX_ACCURACY
        
        # Priority: -7 to 7 -> 0 to 1
        moves_data[:, :, ObsIdx.RAW_DATA["priority_offset"]] = (moves_data[:, :, ObsIdx.RAW_DATA["priority_offset"]] + 7) / 14
        
        moves_data[:, :, ObsIdx.RAW_DATA["secondaryEffectChance_offset"]] /= 100
        moves_data[:, :, ObsIdx.RAW_DATA["target_offset"]] /= 10
        
        return pokemon_data.flatten()
    
    def get_reduced_agent_data(self, agent: str) -> AgentObs:
        if agent not in self._o:
            raise ValueError(f"Invalid agent name, must be in {self._o.keys()}, got {agent}.")
        
        raw_data = self._o[agent].astype(float)
        pokemon_data = raw_data.reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
        
        base_indices = [
            ObsIdx.RAW_DATA["is_active"],
            *range(ObsIdx.RAW_DATA["stats_begin"], ObsIdx.RAW_DATA["stats_end"]),
            ObsIdx.RAW_DATA["ability"],
            ObsIdx.RAW_DATA["type_1"],
            ObsIdx.RAW_DATA["type_2"],
            ObsIdx.RAW_DATA["HP"],
            ObsIdx.RAW_DATA["max_HP"],
            ObsIdx.RAW_DATA["level"],
            ObsIdx.RAW_DATA["friendship"],
            ObsIdx.RAW_DATA["held_item"],
            ObsIdx.RAW_DATA["status"]
        ]
        
        move_offsets = [
            ObsIdx.RAW_DATA["pp_offset"],
            ObsIdx.RAW_DATA["power_offset"],
            ObsIdx.RAW_DATA["accuracy_offset"],
            ObsIdx.RAW_DATA["priority_offset"],
            ObsIdx.RAW_DATA["secondaryEffectChance_offset"],
            ObsIdx.RAW_DATA["effect_offset"],
            ObsIdx.RAW_DATA["type_offset"],
            ObsIdx.RAW_DATA["target_offset"],
            ObsIdx.RAW_DATA["flags_offset"]
        ]
        
        reduced_team = []
        for i in range(DataSize.PARTY_SIZE):
            pkmn = pokemon_data[i]
            reduced_pkmn = pkmn[base_indices]
            moves_part = pkmn[ObsIdx.RAW_DATA["moves_begin"]:].reshape((ObsIdx.MAX_PKMN_MOVES, ObsIdx.MOVE_ATTRIBUTES_PER_MOVE))
            reduced_moves = moves_part[:, move_offsets].flatten()
            reduced_team.append(np.concatenate([reduced_pkmn, reduced_moves]))
            
        return np.concatenate(reduced_team)

    def get_embedding_data(self, agent: str) -> Dict[str, AgentObs]:
        if agent not in self._o:
            raise ValueError(f"Invalid agent name, must be in {self._o.keys()}, got {agent}.")
        
        raw_data = self._o[agent].astype(float)
        pokemon_data = raw_data.reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
        
        MAX_STAT = 550
        MAX_HP = 550
        MAX_POWER = 256
        MAX_ACCURACY = 100
        MAX_PP = 40
        MAX_LEVEL = 100
        MAX_FRIENDSHIP = 255

        cat_base_indices = [
            ObsIdx.RAW_DATA["species"],
            ObsIdx.RAW_DATA["ability"],
            ObsIdx.RAW_DATA["type_1"],
            ObsIdx.RAW_DATA["type_2"],
            ObsIdx.RAW_DATA["held_item"],
            ObsIdx.RAW_DATA["status"]
        ]
        
        cat_move_offsets = [
            ObsIdx.RAW_DATA["move_id_offset"],
            ObsIdx.RAW_DATA["effect_offset"],
            ObsIdx.RAW_DATA["type_offset"],
            ObsIdx.RAW_DATA["target_offset"],
            ObsIdx.RAW_DATA["flags_offset"]
        ]

        continuous_data_list = []
        categorical_data_list = []

        for i in range(DataSize.PARTY_SIZE):
            pkmn = pokemon_data[i]
            categorical_data_list.append(pkmn[cat_base_indices].astype(int))
            
            c_active = pkmn[ObsIdx.RAW_DATA["is_active"]]
            c_stats = pkmn[ObsIdx.RAW_DATA["stats_begin"]:ObsIdx.RAW_DATA["stats_end"]] / MAX_STAT
            c_hp = pkmn[ObsIdx.RAW_DATA["HP"]] / MAX_HP
            c_max_hp = pkmn[ObsIdx.RAW_DATA["max_HP"]] / MAX_HP
            c_lvl = pkmn[ObsIdx.RAW_DATA["level"]] / MAX_LEVEL
            c_friend = pkmn[ObsIdx.RAW_DATA["friendship"]] / MAX_FRIENDSHIP
            
            continuous_data_list.extend([c_active])
            continuous_data_list.append(c_stats)
            continuous_data_list.extend([c_hp, c_max_hp, c_lvl, c_friend])

            moves_part = pkmn[ObsIdx.RAW_DATA["moves_begin"]:].reshape((ObsIdx.MAX_PKMN_MOVES, ObsIdx.MOVE_ATTRIBUTES_PER_MOVE))
            categorical_data_list.append(moves_part[:, cat_move_offsets].flatten().astype(int))
            
            m_pp = moves_part[:, ObsIdx.RAW_DATA["pp_offset"]] / MAX_PP
            m_power = moves_part[:, ObsIdx.RAW_DATA["power_offset"]] / MAX_POWER
            m_acc = moves_part[:, ObsIdx.RAW_DATA["accuracy_offset"]] / MAX_ACCURACY
            m_prio = (moves_part[:, ObsIdx.RAW_DATA["priority_offset"]] + 7) / 14
            m_sec = moves_part[:, ObsIdx.RAW_DATA["secondaryEffectChance_offset"]] / 100
            
            m_cont_stacked = np.stack([m_pp, m_power, m_acc, m_prio, m_sec], axis=1).flatten()
            continuous_data_list.append(m_cont_stacked)
            
        return {
            "categorical": np.concatenate(categorical_data_list),
            "continuous": np.concatenate([np.atleast_1d(x) for x in continuous_data_list])
        }

    def active_pkmn(self) -> Dict[str, int]:
        res = {}
        for agent in ["player", "enemy"]:
            raw = self._o[agent].reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
            actives = np.where(raw[:, ObsIdx.RAW_DATA["is_active"]] == 1)[0]
            res[agent] = actives[0] if len(actives) > 0 else 0
        return res

    def get_pp(self) -> Dict[str, List[int]]:
        res = {}
        for agent in ["player", "enemy"]:
            raw = self._o[agent].reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
            # Extract PPs for all moves of active pokemon
            active_idx = self.active_pkmn()[agent]
            moves = raw[active_idx, ObsIdx.RAW_DATA["moves_begin"]:].reshape((ObsIdx.MAX_PKMN_MOVES, ObsIdx.MOVE_ATTRIBUTES_PER_MOVE))
            res[agent] = moves[:, ObsIdx.RAW_DATA["pp_offset"]].tolist()
        return res
    
    def hp(self) -> Dict[str, List[int]]:
        res = {}
        for agent in ["player", "enemy"]:
            raw = self._o[agent].reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
            res[agent] = raw[:, ObsIdx.RAW_DATA["HP"]].astype(int).tolist()
        return res

    def lvl(self) -> Dict[str, List[int]]:
        res = {}
        for agent in ["player", "enemy"]:
            raw = self._o[agent].reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
            res[agent] = raw[:, ObsIdx.RAW_DATA["level"]].astype(int).tolist()
        return res

    def stats(self) -> Dict[str, List[List[int]]]:
        res = {}
        for agent in ["player", "enemy"]:
            raw = self._o[agent].reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
            res[agent] = raw[:, ObsIdx.RAW_DATA["stats_begin"]:ObsIdx.RAW_DATA["stats_end"]].astype(int).tolist()
        return res

    def pkmn_ko(self) -> Dict[str, List[bool]]:
        res = {}
        hps = self.hp()
        for agent in ["player", "enemy"]:
            # A pokemon is KO if HP is 0. Note: Empty slots (species 0) also have HP 0.
            res[agent] = [h == 0 for h in hps[agent]]
        return res

    def who_won(self) -> str | None:
        player_hp = self.hp()["player"]
        enemy_hp = self.hp()["enemy"]
        
        # Check if ANY pokemon has HP > 0
        player_alive = np.any(np.array(player_hp) > 0)
        enemy_alive = np.any(np.array(enemy_hp) > 0)
        
        if not player_alive:
            return "enemy"
        if not enemy_alive:
            return "player"
        
        return None

class ObservationFactory:
    def __init__(self, battle_core: BattleCore):
        self.battle_core = battle_core
        self.moves_df = pd.read_csv(PATHS["MOVES_CSV"])
        self._init_move_lookup()

    def _init_move_lookup(self):
        max_move_id = self.moves_df['id'].max()
        self.move_lookup = np.zeros((max_move_id + 1, 8), dtype=int)
        
        for _, row in self.moves_df.iterrows():
            mid = int(row['id'])
            self.move_lookup[mid] = [
                row['effect'], row['power'], row['type'], row['accuracy'],
                row['priority'], row['secondaryEffectChance'], row['target'], row['flags']
            ]

    def from_game(self) -> Observation:
        observations = {}
        for agent in ["player", "enemy"]:
            raw = self.battle_core.read_team_data(agent)
            raw_np = np.array(raw, dtype=int).reshape((DataSize.PARTY_SIZE, 28))
            
            obs_full = np.zeros((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN), dtype=int)
            
            # Copy base data (indices 0 to 19)
            obs_full[:, :20] = raw_np[:, :20]
            
            # Process moves (indices 20-23 are IDs, 24-27 are PPs)
            move_ids = raw_np[:, 20:24]
            pps = raw_np[:, 24:28]
            
            for i in range(ObsIdx.MAX_PKMN_MOVES):
                base_offset = ObsIdx.RAW_DATA["moves_begin"] + i * ObsIdx.MOVE_ATTRIBUTES_PER_MOVE
                
                obs_full[:, base_offset + ObsIdx.RAW_DATA["move_id_offset"]] = move_ids[:, i]
                obs_full[:, base_offset + ObsIdx.RAW_DATA["pp_offset"]] = pps[:, i]
                
                safe_ids = np.clip(move_ids[:, i], 0, self.move_lookup.shape[0] - 1)
                attrs = self.move_lookup[safe_ids]
                obs_full[:, base_offset + 2 : base_offset + 10] = attrs
            
            observations[agent] = obs_full.flatten()
            
        return Observation(_o=observations)

    def from_diff(o1: Observation, o2: Observation) -> Observation:
        diff_observations = {}
        return Observation(_o=diff_observations)