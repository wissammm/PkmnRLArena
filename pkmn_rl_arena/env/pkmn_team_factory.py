from pkmn_rl_arena.paths import PATHS
from .battle_core import BattleCore
from pkmn_rl_arena import log

import ast
from dataclasses import dataclass
from typing import List
import random

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class DataSize:
    ID = 1
    LEVEL = 1
    MOVES = 4
    HP_PERCENT = 1
    ITEM = 1
    PARTY_SIZE = 6
    PKMN = ID + LEVEL + MOVES + HP_PERCENT + ITEM
    TEAM = PARTY_SIZE * PKMN


@dataclass(frozen=True)
class Ranges:
    ITEM_1 = [178, 225]
    ITEM_2 = [132, 175]
    ALL_ITEMS = list(range(ITEM_1[1], ITEM_1[0], -1)) + list(
        range(ITEM_2[1], ITEM_2[0], -1)
    )
    PKMN_ID_BOUNDS = [0, 411]


class PkmnTeamFactory:
    def __init__(
        self,
        pkmn_path=PATHS["POKEMON_CSV"],
        moves_path=PATHS["POKEMON_CSV"],
        seed: int | None = None,
    ):
        # "id" 0 is test
        self.pkmns = pd.read_csv(PATHS["POKEMON_CSV"])
        self.pkmns = self.pkmns[self.pkmns["id"] != 0]

        self.moves = pd.read_csv(PATHS["MOVES_CSV"])

        # indexing DF by pkm ids
        self.pkmns = self.pkmns.set_index("id", drop=False)
        self.moves = self.moves.set_index("id", drop=False)

        self.seed = seed

    def create_random_team(self) -> List[int]:
        """
        Create a random team from the provided CSV files.

        Note : All created pkmn will be lvl 10

        Returns:
            List[Pkmn]: A flat list of pkmns representing the team in the format:

                    [id, level, move0, move1, move2, move3, ...]
        """
        chosen_species = self.pkmns.sample(n=6)
        LVL = 10

        team: List[int] = []

        for _, random_species in chosen_species.iterrows():
            moves_list = eval(random_species["moves"])
            random_moves_idx = random.sample(moves_list, min(len(moves_list), 4))
            while len(random_moves_idx) < 4:
                random_moves_idx.append(0)

            hp_percent = 100
            item_id = random.choice(Ranges.ALL_ITEMS)
            team.extend(
                [random_species["id"], LVL] + random_moves_idx + [hp_percent, item_id]
            )

        return team

    def get_pkmn_name_from(self, pkmn_id: int) -> str:
        return self.pkmns.loc[pkmn_id, "speciesName"]

    def get_movepool_from(self, pkmn_id: int) -> List[int]:
        """
        Retrieve movepool list from pokemon dataframe
        Adds 0 which is the entry to say "no move"
        """
        return [0] + ast.literal_eval(self.pkmns.loc[pkmn_id, "moves"])

    def get_move_name_from(self, move_id: int) -> str:
        return self.moves.loc[move_id, "moveName"]

    @staticmethod
    def is_valid_id(id: int):
        if not Ranges.PKMN_ID_BOUNDS[0] <= id <= Ranges.PKMN_ID_BOUNDS[1]:
            log.error(
                f"Trying to create a pokemon with invalid id. Authorized range [{Ranges.PKMN_ID_BOUNDS[0]},{Ranges.PKMN_ID_BOUNDS[1]}], got {id}."
            )
            return False
        return True

    def is_in_pkmn_movepool(self, pkmn_id: int, move_ids: npt.NDArray) -> bool:
        movepool = self.get_movepool_from(pkmn_id)
        for move_id in move_ids:
            if move_id not in movepool:
                log.error(
                    f"Trying to create a pkmn :"
                    f"\n\tname {self.get_pkmn_name_from(pkmn_id)}"
                    f"\n\tid {pkmn_id}"
                    f"\n\tmoves {move_ids}"
                    f"\n\tHowever move {self.get_move_name_from(move_id)} with id {move_id} is not in its movepool."
                    f"\n\t{self.get_pkmn_name_from(pkmn_id)} move pool :  {movepool}"
                )
                return False
        return True

    @staticmethod
    def is_item_id_valid(id: int) -> bool:
        if not (
            id == 0
            or Ranges.ITEM_1[0] <= id <= Ranges.ITEM_1[1]
            or Ranges.ITEM_2[0] <= id <= Ranges.ITEM_2[1]
        ):
            log.error(
                f"Trying to create a pokemon with invalid item id."
                f"\n\tAuthorized values 0, [{Ranges.ITEM_1[0]}, {Ranges.ITEM_1[1]}] or [{Ranges.ITEM_2[0]}, {Ranges.ITEM_2[1]}]"
                f"\n\tGot {id}."
            )
            return False
        return True

    @staticmethod
    def is_hp_percent_valid(percent: int) -> bool:
        if not 0 <= percent <= 100:  # hp percent
            log.error(
                f"Invalid HP percentage : expected value comprised between [0;100], got {percent}."
            )
            return False
        return True

    def is_pkmn_valid(self, pkmn: npt.NDArray) -> bool:
        return (
            self.is_valid_id(pkmn[0])
            and self.is_in_pkmn_movepool(pkmn[0], pkmn[2:6])
            and self.is_hp_percent_valid(pkmn[6])
            and self.is_item_id_valid(pkmn[7])
        )

    def is_team_valid(self, team: npt.NDArray) -> bool:
        if len(team) != DataSize.TEAM:
            raise ValueError(
                "Wrong input length :\n"
                f"\tA team can only be generated with an array of size {DataSize.TEAM}.\n"
                f"\tGot {len(team)} data points : {team}."
            )

        log.debug("Creating a pkmn team with following pkmn:")
        team_pkmns = np.split(team, DataSize.PARTY_SIZE)
        for i, pkmn in enumerate(team_pkmns):
            if np.all(pkmn == 0):
                log.debug(f"PKMN {i} : Empty slot")
                continue
            log.debug(
                f"PKMN {i} :"
                f"\n\tid : {pkmn[0]}"
                f"\n\tlvl : {pkmn[1]}"
                f"\n\tmove ids : {pkmn[2:6]}"
                f"\n\thp_percent : {pkmn[6]}"
                f"\n\titem id : {pkmn[7]}"
            )
            if not self.is_pkmn_valid(pkmn):
                return False
        return True
