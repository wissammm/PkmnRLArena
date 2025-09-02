from pkmn_rl_arena import POKEMON_CSV_PATH, MOVES_CSV_PATH
from .battle_core import BattleCore

import pandas as pd

from typing import List
import random

class PkmnTeamFactory:
    def __init__(
        self,
        pkmn_path = POKEMON_CSV_PATH,
        moves_path = POKEMON_CSV_PATH,
        seed: int | None = None,
    ):
        # "id" 0 is test
        self.pkmn = pd.read_csv(POKEMON_CSV_PATH)
        self.pkmn = self.pkmn[self.pkmn["id"] != 0]
        self.moves = pd.read_csv(POKEMON_CSV_PATH)
        self.seed = seed

    def create_random_team(self) -> List[int]:
        """
        Create a random team from the provided CSV files.

        Returns:
            List[Pkmn]: A flat list of pkmns representing the team in the format:
                    [id, level, move0, move1, move2, move3, ...]
        """
        chosen_species = self.pkmn.sample(n=6)

        # Define item ranges
        item_range_1 = list(range(225, 178, -1))
        item_range_2 = list(range(175, 132, -1))
        all_items = item_range_1 + item_range_2

        team: List[int] = []
        for _, random_species in chosen_species.iterrows():
            moves_list = eval(random_species["moves"])
            random_moves_idx = random.sample(moves_list, min(len(moves_list), 4))
            while len(random_moves_idx) < 4:
                random_moves_idx.append(0)

            hp_percent = 100
            item_id = random.choice(all_items)
            team.extend(
                [random_species["id"], 10] + random_moves_idx + [hp_percent, item_id]
            )

        print(f"Created random team: {team}")
        return team

    @staticmethod
    def is_valid_id(id: int):
        if not 1 <= id <= 411:
            raise ValueError(
                f"Trying to get a pkmn name from an invalid ID. Id must be comprised within [1;411], got {id}"
            )

    def get_pokemon_rows(self, ids: List[int]):
        # error checking
        [PkmnTeamFactory.is_valid_id(id) for id in ids]
        # computation
        return self.pkmn["id" in ids]
