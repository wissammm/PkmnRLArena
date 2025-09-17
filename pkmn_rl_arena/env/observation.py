from typing import Dict, List
from dataclasses import dataclass
from numpy import typing as npt
from pkmn_rl_arena.paths import PATHS

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

    @property
    def agent(self, a: str):
        if a not in self.o.keys():
            raise ValueError(
                f"Invalid agent name, must be in {self.o.keys()}, got {a}."
            )
        self.o[a]

    def get_agent_data(self, agent: str) -> AgentObs:
        """Get the full observation array for an agent"""
        if agent not in self._o:
            raise ValueError(f"Invalid agent name, must be in {self._o.keys()}, got {agent}.")
        return self._o[agent]

    def active_pkmn(self) -> Dict[str, int]:
        """
        Return active pkmn idx for each agent
        """
        result = {}
        for agent in self._o:
            agent_data = self._o[agent]
            pokemon_data = np.split(agent_data, 6)
            
            for i, pkmn in enumerate(pokemon_data):
                if pkmn[ObsIdx.RAW_DATA["is_active"]] == 1:
                    result[agent] = i
                    break
            
            if agent not in result:
                result[agent] = None
                
        return result

    def hp(self) -> Dict[str, List[int]]:
        """
        Return current HP values for all Pokémon in each team
        """
        result = {"player": [], "enemy": []}
        
        for agent in self._o:
            agent_data = self._o[agent]
            pokemon_data = np.split(agent_data, 6)
            
            for pkmn in pokemon_data:
                hp_value = int(pkmn[ObsIdx.RAW_DATA["HP"]])
                result[agent].append(hp_value)
                
        return result

    def stat_changes(self) -> Dict[str, List[List[int]]]:
        """
        Return current stat values for all Pokémon in each team
        Returns a dictionary with agents as keys, and values as lists of stat arrays
        Each stat array contains [ATK, DEF, SPEED, SPATK, SPDEF]
        """
        result = {"player": [], "enemy": []}
        
        for agent in self._o:
            agent_data = self._o[agent]
            pokemon_data = np.split(agent_data, 6)
            
            for pkmn in pokemon_data:
                stats = pkmn[ObsIdx.RAW_DATA["stats_begin"]:ObsIdx.RAW_DATA["stats_end"]].tolist()
                result[agent].append(stats)
                
        return result

    def pkmn_ko(self) -> Dict[str, List[bool]]:
        """
        Returns for each agent a list of booleans indicating if each Pokémon is KO'd
        A Pokémon is KO'd if its HP is 0
        """
        result = {"player": [], "enemy": []}
        
        for agent in self._o:
            agent_data = self._o[agent]
            pokemon_data = np.split(agent_data, 6)
            
            for pkmn in pokemon_data:
                hp_value = int(pkmn[ObsIdx.RAW_DATA["HP"]])
                result[agent].append(hp_value == 0)
                
        return result


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

        # Stats [2..6] => Python slice end is exclusive, so use 7
        "stats_begin": 2,   # ATK, DEF, SPEED, SPATK, SPDEF
        "stats_end": 7,     # exclusive

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

        # Move data with all attributes from moves_data.csv
        # For each move, we store:
        # [move_id, pp, effect, power, type, accuracy, priority, secondaryEffectChance, target, flags]
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
        
        # Each move occupies MOVE_ATTRIBUTES_PER_MOVE slots
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

    def from_game(self) -> Observation:
        """
        For both agents, build an observation vector from raw game data.
        Steps:
          1. Extract relevant Pokémon attributes (active flag, stats, ability → status).
          2. For each move: include id, pp info, and extra move stats.
          3. Concatenate into a flat observation array for the agent.
        """
        moves_df = pd.read_csv(PATHS["MOVES_CSV"])
    
        move_attrs = {}
        for _, row in moves_df.iterrows():
            move_id = int(row['id'])
            move_attrs[move_id] = {
                'effect': int(row['effect']),
                'power': int(row['power']),
                'type': int(row['type']),
                'accuracy': int(row['accuracy']),
                'pp': int(row['pp']),  # This is max PP, not current PP
                'priority': int(row['priority']),
                'secondaryEffectChance': int(row['secondaryEffectChance']),
                'target': int(row['target']),
                'flags': int(row['flags'])
            }
        
        observations = {}
        for agent in ["player", "enemy"]:
            raw_team_data = self.battle_core.read_team_data(agent)
            raw_data_list = np.split(
                np.array(raw_team_data, dtype=int),  # force int dtype
                indices_or_sections=6,
            )
            
            team_obs = []
            for pkmn_data in raw_data_list:
                pkmn_obs = list(pkmn_data[:20])
                
                for i in range(0, ObsIdx.MAX_PKMN_MOVES):
                    move_idx = 20 + (i * 2)  # 20, 22, 24, 26 (move indices in raw data)
                    pp_idx = move_idx + 1     # 21, 23, 25, 27 (PP indices in raw data)
                    
                    move_id = pkmn_data[move_idx]
                    pp = pkmn_data[pp_idx]
                    
                    move_obs = [move_id, pp]
                    
                    if move_id > 0 and move_id in move_attrs:
                        attrs = move_attrs[move_id]
                        move_obs.extend([
                            attrs['effect'],
                            attrs['power'],
                            attrs['type'],
                            attrs['accuracy'],
                            attrs['priority'],
                            attrs['secondaryEffectChance'],
                            attrs['target'],
                            attrs['flags']
                        ])
                    else:
                        move_obs.extend([0] * 8)  # 8 additional attributes
                    
                    pkmn_obs.extend(move_obs)
                
                team_obs.extend(pkmn_obs)
            
            observations[agent] = np.array(team_obs, dtype=int)
        
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
