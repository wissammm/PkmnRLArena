from typing import Dict, List
from dataclasses import dataclass
from numpy import typing as npt
from pkmn_rl_arena import POKEMON_CSV_PATH, MOVES_CSV_PATH

from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.pkmn_team_factory import PkmnTeamFactory

import numpy as np
import pandas as pd

AgentObs = npt.NDArray[int]

NB_PARAM_OBS = 69

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
        if a not in self._o.keys():
            raise ValueError(
                f"Invalid agent name, must be in {self._o.keys()}, got {a}."
            )
        self._o[a]

    def active_pkmn(self) -> Dict[str, int]:
        """
        return active pkmn idx for each agent
        """
        return {"player": 0, "ennemy": 5}

    def hp(self) -> Dict[str,List[int]]:
        """Would return hp diff between 2 observations? or would return an Observation?"""


        result = {"player" : [], "enemy" : []}

        return result

    def stat_changes(self) -> None:
        """Return per pkmn per agent stat changes since last turn
        Idk if it should return an array, a dict or smthg else? Maybe it would be interesting to create a wrapper / helper class to decode status specifically ?"""
        return None

    def pkmn_ko(self, agent: int, pkmn: int) -> Dict[str, List[int]]:
        """
        returns an array of idx for each agent.
        """
        return 


@dataclass(frozen=True)
class ObsIdx:

    MAX_PKMN_MOVES = 4
    NB_STATS_PER_PKMN = 5  # ATK, DEF, SPEED, SPATK, SPDEF
    MOVE_ATTRIBUTES_PER_MOVE = 10  # id, pp, effect, power, type, accuracy, pp, priority, secondaryEffectChance, target, flags
    NB_DATA_PKMN = 20 + (MAX_PKMN_MOVES * MOVE_ATTRIBUTES_PER_MOVE)  # 20 base fields + move data

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
        moves_df = pd.read_csv(MOVES_CSV_PATH)
    
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
        """Computes difference between 2 observations"""
        pass
