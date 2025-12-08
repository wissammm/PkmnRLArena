from typing import Dict, List
from dataclasses import dataclass
from numpy import typing as npt
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena.env.pkmn_team_factory import DataSize 

from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.pkmn_team_factory import PkmnTeamFactory

import numpy as np
import pandas as pd

AgentObs = npt.NDArray[int]



@dataclass
class Observation:
    """
    This class is a wrapper around the observation type Dict[str, Agent]
    It mainly serves as an index helper.

    The functions here are placeholders to fill.

    str : either "player" or "ennemy"
    Agent :
    """

    _o: Dict[str, AgentObs]

    def agent(self, a: str) -> npt.NDArray[int]:
        if a not in self._o.keys():
            raise ValueError(
                f"Invalid agent name, must be in {self.o.keys()}, got {a}."
            )
        return self._o[a]

    def get_agent_data(self, agent: str) -> AgentObs:
        """Get the full observation array for an agent"""
        if agent not in self._o:
            raise ValueError(f"Invalid agent name, must be in {self._o.keys()}, got {agent}.")
        return self._o[agent]
    
    def get_normalized_agent_data(self, agent: str) -> AgentObs:
        """Get the full observation array for an agent, normalized to [0,1]"""
        if agent not in self._o:
            raise ValueError(f"Invalid agent name, must be in {self._o.keys()}, got {agent}.")
        
        raw_data = self._o[agent].copy().astype(float)
        pokemon_data = raw_data.reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
        
        # Constants for normalization
        MAX_SPECIES = 411
        MAX_TYPE = 17
        MAX_STAT = 550
        MAX_HP = 550
        MAX_MOVE_ID = 354
        MAX_POWER = 256
        MAX_ACCURACY = 100
        MAX_PP = 40
        MAX_LEVEL = 100
        MAX_FRIENDSHIP = 255
        
        # Normalize base stats
        pokemon_data[:, ObsIdx.RAW_DATA["species"]] /= MAX_SPECIES
        pokemon_data[:, ObsIdx.RAW_DATA["stats_begin"]:ObsIdx.RAW_DATA["stats_end"]] /= MAX_STAT
        pokemon_data[:, ObsIdx.RAW_DATA["type_1"]] /= MAX_TYPE
        pokemon_data[:, ObsIdx.RAW_DATA["type_2"]] /= MAX_TYPE
        # ability is categorical, skipping or normalizing? Original code didn't normalize ability (just accessed it)
        pokemon_data[:, ObsIdx.RAW_DATA["HP"]] /= MAX_HP
        pokemon_data[:, ObsIdx.RAW_DATA["max_HP"]] /= MAX_HP
        pokemon_data[:, ObsIdx.RAW_DATA["level"]] /= MAX_LEVEL
        pokemon_data[:, ObsIdx.RAW_DATA["friendship"]] /= MAX_FRIENDSHIP
        
        # Normalize moves
        # Moves are at indices 20-59. 4 moves * 10 attributes
        moves_data = pokemon_data[:, ObsIdx.RAW_DATA["moves_begin"]:].reshape((DataSize.PARTY_SIZE, ObsIdx.MAX_PKMN_MOVES, ObsIdx.MOVE_ATTRIBUTES_PER_MOVE))
        
        # 0: id, 1: pp, 2: effect, 3: power, 4: type, 5: accuracy, 6: priority, 7: secondary, 8: target, 9: flags
        moves_data[:, :, ObsIdx.RAW_DATA["move_id_offset"]] /= MAX_MOVE_ID
        moves_data[:, :, ObsIdx.RAW_DATA["pp_offset"]] /= MAX_PP
        moves_data[:, :, ObsIdx.RAW_DATA["power_offset"]] /= MAX_POWER
        moves_data[:, :, ObsIdx.RAW_DATA["type_offset"]] /= MAX_TYPE
        moves_data[:, :, ObsIdx.RAW_DATA["accuracy_offset"]] /= MAX_ACCURACY
        
        # Priority: -7 to 7 -> 0 to 1
        moves_data[:, :, ObsIdx.RAW_DATA["priority_offset"]] = (moves_data[:, :, ObsIdx.RAW_DATA["priority_offset"]] + 7) / 14
        
        moves_data[:, :, ObsIdx.RAW_DATA["secondaryEffectChance_offset"]] /= 100
        moves_data[:, :, ObsIdx.RAW_DATA["target_offset"]] /= 10
        
        # Flatten back
        return pokemon_data.flatten()
    
    def get_reduced_agent_data(self, agent: str) -> AgentObs:
        """
        Return a reduced version of the observation array for an agent.
        - Removes useless identifiers (species, move_id, personality, etc.)
        - Keeps only meaningful numerical + categorical features.

        Returns:
            np.ndarray: Flattened vector of selected features for the entire team.
        """
        if agent not in self._o:
            raise ValueError(f"Invalid agent name, must be in {self._o.keys()}, got {agent}.")
        
        raw_data = self._o[agent].astype(float)
        pokemon_data = raw_data.reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
        
        # Define indices to keep
        # Base: is_active(1), stats(2-6), ability(7), type1(8), type2(9), HP(10), level(11), friendship(12), max_HP(13), held_item(14), status(17)
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
        
        # Moves: pp(1), power(3), accuracy(5), priority(6), secondary(7), effect(2), type(4), target(8), flags(9)
        # Relative offsets
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
            
            # Moves
            moves_part = pkmn[ObsIdx.RAW_DATA["moves_begin"]:].reshape((ObsIdx.MAX_PKMN_MOVES, ObsIdx.MOVE_ATTRIBUTES_PER_MOVE))
            reduced_moves = moves_part[:, move_offsets].flatten()
            
            reduced_team.append(np.concatenate([reduced_pkmn, reduced_moves]))
            
        return np.concatenate(reduced_team)

    def get_embedding_data(self, agent: str) -> Dict[str, AgentObs]:
        """
        Returns a dictionary with 'categorical' and 'continuous' data for embeddings.
        - Categorical data is kept as integers (indices) for Embedding layers.
        - Continuous data is normalized to [0, 1] for direct input.
        """
        if agent not in self._o:
            raise ValueError(f"Invalid agent name, must be in {self._o.keys()}, got {agent}.")
        
        raw_data = self._o[agent].astype(float) # Use float for normalization calculations
        pokemon_data = raw_data.reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
        
        # Constants for normalization
        MAX_STAT = 550
        MAX_HP = 550
        MAX_POWER = 256
        MAX_ACCURACY = 100
        MAX_PP = 40
        MAX_LEVEL = 100
        MAX_FRIENDSHIP = 255

        # --- Categorical Data (Indices) ---
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

        # --- Continuous Data (Values to Normalize) ---
        # We extract them, normalize them, then flatten
        
        continuous_data_list = []
        categorical_data_list = []

        for i in range(DataSize.PARTY_SIZE):
            pkmn = pokemon_data[i]
            
            # 1. Categorical Base
            categorical_data_list.append(pkmn[cat_base_indices].astype(int))
            
            # 2. Continuous Base
            # is_active (0 or 1)
            c_active = pkmn[ObsIdx.RAW_DATA["is_active"]]
            # stats
            c_stats = pkmn[ObsIdx.RAW_DATA["stats_begin"]:ObsIdx.RAW_DATA["stats_end"]] / MAX_STAT
            # HP, max_HP
            c_hp = pkmn[ObsIdx.RAW_DATA["HP"]] / MAX_HP
            c_max_hp = pkmn[ObsIdx.RAW_DATA["max_HP"]] / MAX_HP
            # level, friendship
            c_lvl = pkmn[ObsIdx.RAW_DATA["level"]] / MAX_LEVEL
            c_friend = pkmn[ObsIdx.RAW_DATA["friendship"]] / MAX_FRIENDSHIP
            
            continuous_data_list.extend([c_active])
            continuous_data_list.append(c_stats)
            continuous_data_list.extend([c_hp, c_max_hp, c_lvl, c_friend])

            # Moves
            moves_part = pkmn[ObsIdx.RAW_DATA["moves_begin"]:].reshape((ObsIdx.MAX_PKMN_MOVES, ObsIdx.MOVE_ATTRIBUTES_PER_MOVE))
            
            # 3. Categorical Moves
            categorical_data_list.append(moves_part[:, cat_move_offsets].flatten().astype(int))
            
            # 4. Continuous Moves
            # pp
            m_pp = moves_part[:, ObsIdx.RAW_DATA["pp_offset"]] / MAX_PP
            # power
            m_power = moves_part[:, ObsIdx.RAW_DATA["power_offset"]] / MAX_POWER
            # accuracy
            m_acc = moves_part[:, ObsIdx.RAW_DATA["accuracy_offset"]] / MAX_ACCURACY
            # priority (-7 to 7 -> 0 to 1)
            m_prio = (moves_part[:, ObsIdx.RAW_DATA["priority_offset"]] + 7) / 14
            # secondary chance
            m_sec = moves_part[:, ObsIdx.RAW_DATA["secondaryEffectChance_offset"]] / 100
            
            # Interleave move continuous data or just append blocks? 
            # Appending blocks is easier to manage: [PPs, Powers, Accs, Prios, Secs]
            # But usually we want [Move1_PP, Move1_Power..., Move2_PP...]
            # Let's stack them to shape (4, 5) then flatten
            m_cont_stacked = np.stack([m_pp, m_power, m_acc, m_prio, m_sec], axis=1).flatten()
            continuous_data_list.append(m_cont_stacked)
            
        return {
            "categorical": np.concatenate(categorical_data_list),
            "continuous": np.concatenate([np.atleast_1d(x) for x in continuous_data_list])
        }


    def active_pkmn(self) -> Dict[str, int]:
        """
        Return active pkmn idx for each agent
        """
        result = {}
        for agent in self._o:
            agent_data = self._o[agent].reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
            is_active = agent_data[:, ObsIdx.RAW_DATA["is_active"]]
            indices = np.where(is_active == 1)[0]
            if len(indices) > 0:
                result[agent] = int(indices[0])
            else:
                result[agent] = None
        return result

    def get_pp(self) -> Dict[str, List[int]]:
        """
        Return current PP values for all moves of the active Pokémon in each team
        """
        result = {"player": [], "enemy": []}
        
        for agent in self._o:
            agent_data = self._o[agent].reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
            is_active = agent_data[:, ObsIdx.RAW_DATA["is_active"]]
            indices = np.where(is_active == 1)[0]
            
            if len(indices) > 0:
                active_pkmn = agent_data[indices[0]]
                # Extract PP
                pp_indices = [ObsIdx.RAW_DATA["moves_begin"] + i * ObsIdx.MOVE_ATTRIBUTES_PER_MOVE + ObsIdx.RAW_DATA["pp_offset"] for i in range(ObsIdx.MAX_PKMN_MOVES)]
                result[agent] = active_pkmn[pp_indices].tolist()
            else:
                result[agent] = [0] * ObsIdx.MAX_PKMN_MOVES
                
        return result

    
    def hp(self) -> Dict[str, List[int]]:
        """
        Return current HP values for all Pokémon in each team
        """
        result = {"player": [], "enemy": []}
        
        for agent in self._o:
            agent_data = self._o[agent].reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
            result[agent] = agent_data[:, ObsIdx.RAW_DATA["HP"]].tolist()

        return result

    def lvl(self) -> Dict[str, List[int]]:
        result = {"player": [], "enemy": []}

        for agent in self._o:
            agent_data = self._o[agent].reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
            result[agent] = agent_data[:, ObsIdx.RAW_DATA["level"]].tolist()

        return result

    def stats(self) -> Dict[str, List[List[int]]]:
        """
        Return current stat values for all Pokémon in each team
        Returns a dictionary with agents as keys, and values a stat array for each pkmn
        Each stat array contains [ATK, DEF, SPEED, SPATK, SPDEF]
        """
        result = {"player": [], "enemy": []}
        
        for agent in self._o:
            agent_data = self._o[agent].reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
            stats = agent_data[:, ObsIdx.RAW_DATA["stats_begin"]:ObsIdx.RAW_DATA["stats_end"]]
            result[agent] = stats.tolist()
                
        return result

    def pkmn_ko(self) -> Dict[str, List[bool]]:
        """
        Returns for each agent a list of booleans indicating if each Pokémon is KO'd
        A Pokémon is KO'd if its HP is 0
        """
        result = {"player": [], "enemy": []}
        
        for agent in self._o:
            agent_data = self._o[agent].reshape((DataSize.PARTY_SIZE, ObsIdx.NB_DATA_PKMN))
            hp = agent_data[:, ObsIdx.RAW_DATA["HP"]]
            result[agent] = (hp == 0).tolist()
        return result

    def who_won(self) -> str | None:
        for agent, has_won in {
            agent: not all( ko == 1 for ko in pkmn_ko) for agent, pkmn_ko in self.pkmn_ko().items()
        }.items():
            if has_won:
                return agent
        return None

        
@dataclass(frozen=True)
class ObsIdx:

    MAX_PKMN_MOVES = 4
    NB_STATS_PER_PKMN = 5  # ATK, DEF, SPEED, SPATK, SPDEF
    MOVE_ATTRIBUTES_PER_MOVE = 10  # id, pp, effect, power, type, accuracy, pp, priority, secondaryEffectChance, target, flags
    NB_DATA_PKMN = 20 + (MAX_PKMN_MOVES * MOVE_ATTRIBUTES_PER_MOVE)  # 20 base fields + move data
    OBS_SIZE = 6 * NB_DATA_PKMN  
    RAW_DATA = {
       "species": 0,
        "is_active": 1,

        "stats_begin": 2,   # ATK, DEF, SPEED, SPATK, SPDEF
        "stats_end": 7,  

        "ability": 7,       # ability num
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
        "status_2": 18,     # reserved in dump
        "status_3": 19,     # reserved in dump

        "moves_begin": 20,
        "moves_end": 20 + (MAX_PKMN_MOVES * MOVE_ATTRIBUTES_PER_MOVE),  # exclusive
        
        # Attribute offsets within each move's data block
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
    
class ObservationFactory:
    """
    Manages extraction and formatting of observations from the battle state.

    Observations are Dict[str,npt.NDarray[int]].
    ```python
    obs = {
        "player" : np.array(size of agent state),
         "agent" : np.array(size of agent state)
    }
    ```
    """

    def __init__(self, battle_core: BattleCore):
        self.battle_core = battle_core
        self.moves_df = pd.read_csv(PATHS["MOVES_CSV"])
        self._init_move_lookup()

    def _init_move_lookup(self):
        max_move_id = self.moves_df['id'].max()
        # Attributes: effect, power, type, accuracy, priority, secondaryEffectChance, target, flags
        self.move_lookup = np.zeros((max_move_id + 1, 8), dtype=int)
        
        for _, row in self.moves_df.iterrows():
            mid = int(row['id'])
            self.move_lookup[mid] = [
                int(row['effect']),
                int(row['power']),
                int(row['type']),
                int(row['accuracy']),
                int(row['priority']),
                int(row['secondaryEffectChance']),
                int(row['target']),
                int(row['flags'])
            ]

    def from_game(self) -> Observation:
        """
        For both agents, build an observation vector from raw game data.
        Steps:
        1. Extract relevant Pokémon attributes (active flag, stats, ability → status).
        2. For each move: include id, pp info, and extra move stats.
        3. Concatenate into a flat observation array for the agent.
        """
        observations = {}
        for agent in ["player", "enemy"]:
            raw_team_data = np.array(self.battle_core.read_team_data(agent), dtype=int)
            # Reshape to (6, 28)
            # 28 ints per pokemon: 20 base + 4 * (move_id, pp)
            pkmn_data = raw_team_data.reshape((DataSize.PARTY_SIZE, 28))
            
            # Base stats: first 20
            base_stats = pkmn_data[:, :20]
            
            # Moves data: last 8 (4 pairs of id, pp)
            moves_raw = pkmn_data[:, 20:]
            moves_raw = moves_raw.reshape((DataSize.PARTY_SIZE, ObsIdx.MAX_PKMN_MOVES, 2))
            
            move_ids = moves_raw[:, :, 0]
            current_pps = moves_raw[:, :, 1]
            
            # Ensure move_ids are within bounds
            move_ids_clipped = np.clip(move_ids, 0, self.move_lookup.shape[0] - 1)
            attrs = self.move_lookup[move_ids_clipped] # Shape (6, 4, 8)
            
            # Combine: id, pp, attrs
            moves_full = np.concatenate([
                move_ids[:, :, np.newaxis],
                current_pps[:, :, np.newaxis],
                attrs
            ], axis=2) # Shape (6, 4, 10)
            
            # Flatten moves
            moves_flat = moves_full.reshape((DataSize.PARTY_SIZE, -1)) # (6, 40)
            
            # Combine base and moves
            full_obs = np.concatenate([base_stats, moves_flat], axis=1) # (6, 60)
            
            observations[agent] = full_obs.flatten()
        
        return Observation(_o=observations)

    def from_diff(o1: Observation, o2: Observation) -> Observation:
        diff_observations = {}

        if set(o1._o.keys()) != set(o2._o.keys()):
            raise ValueError("Observations must have the same agents")
        
        for agent in o1._o:
            arr1 = o1._o[agent]
            arr2 = o2._o[agent]
            
            if arr1.shape != arr2.shape:
                raise ValueError(f"Observation arrays for agent {agent} have different shapes")
            
            diff = arr2 - arr1
            
            diff_observations[agent] = diff
        
        return Observation(_o=diff_observations)
    