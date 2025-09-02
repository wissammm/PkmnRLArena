from typing import Dict, List
from dataclasses import dataclass
from numpy import typing as npt

from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.pkmn_team_factory import PkmnTeamFactory

import numpy as np

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
        for agent , array in result.items():
            array[self._o[hp_idx_pkmn_1],self._o[hp_idx_pkmn_2],self._o[hp_idx_pkmn_3],self._o[hp_idx_pkmn_4],self._o[hp_idx_pkmn_5],self._o[hp_idx_pkmn_6]]

        return array

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
    #
    #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!
    #  NOT UP TO DATE
    #
    #
    # 
    MAX_PKMN_MOVES = 4
    NB_STATS_PER_PKMN = 0  # ??????
    NB_DATA_PKMN = 72
    # TO BE MODIFIED
    RAW_DATA = {
        "is_active": 0,
        "species": 1,
        "stats_begin": 2,
        "stats_end": 6,
        "moves_begin": 7,
        "moves_end": 10,
        "iv_begin": 11,
        "iv_end": 16,
        "ability": 17,
        "abitlity_2": 18,
        "specie_1": 19,
        "specie_2": 20,
        "HP": 21,
        "level": 22,
        "friendship": 23,
        "max_HP": 24,
        "held_item": 25,
        "pp_bonuses": 26,
        "personality": 27,
        "status": 28,
        "status_1": 29,
        "status_2": 30,
        "PP_begin": 31,
        "PP_end": 34,
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
        observations: Dict[str, np.ndarray | None] = {"player": None, "enemy": None}

        # Pre-index moves_df by id for O(1) lookups
        moves_df = PkmnTeamFactory.moves.set_index("id").drop(columns=["moveName"])
        moves_dict = {idx: row.to_numpy() for idx, row in moves_df.iterrows()}

        for agent in observations.keys():
            # Split raw team data into 6 Pokémon entries
            raw_data_list = np.split(
                np.array(
                    self.battle_core.read_team_data(agent), dtype=int
                ),  # force int dtype
                indices_or_sections=6,
            )

            agent_data = np.array([])

            for raw_pkmn in raw_data_list:
                # Core Pokémon attributes
                core_data = np.concatenate(
                    [
                        [raw_pkmn[ObsIdx.RAW_DATA_IDX["is_active"]]],
                        raw_pkmn[
                            ObsIdx.RAW_DATA_IDX["stats_begin"] : ObsIdx.RAW_DATA_IDX[
                                "stats_end"
                            ]
                        ],
                        raw_pkmn[
                            ObsIdx.RAW_DATA_IDX["ability"] : ObsIdx.RAW_DATA_IDX[
                                "status_2"
                            ]
                        ],
                    ]
                )

                # Moves data (id, pp, stats)
                moves_id = raw_pkmn[7:11]
                moves_data = np.concatenate(
                    [
                        np.concatenate(
                            [
                                [move_id],
                                [
                                    moves_dict[move_id][0]
                                ],  # max_pp from first col of stored row
                                [raw_pkmn[ObsIdx.RAW_DATA_IDX["PP_begin"] + i]],
                                moves_dict[move_id][1:],  # the rest of stats
                            ]
                        )
                        for i, move_id in enumerate(moves_id)
                        if move_id in moves_dict
                    ]
                )

                # Merge into agent array
                agent_data = np.concatenate([agent_data, core_data, moves_data])
                observations[agent] = agent_data

        return observations

    def from_diff(o1: Observation, o2: Observation) -> Observation:
        """Computes difference between 2 observations"""
        pass
