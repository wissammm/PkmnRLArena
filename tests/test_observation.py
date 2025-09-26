from pkmn_rl_arena.env.rendering import TYPES_ID
import unittest
import numpy as np
import picologging as logging
import random

import pkmn_rl_arena
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena import log
from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.battle_arena import BattleArena
from pkmn_rl_arena.data import pokemon_data
from pkmn_rl_arena.env.turn_type import TurnType

from pkmn_rl_arena.env.observation import Observation, ObsIdx


class TestObservation(unittest.TestCase):
    def setUp(self):
        log.setLevel(logging.DEBUG)
        core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.arena = BattleArena(core)

    def tearDown(self):
        self.arena.close()

    def test_observation_space(self):
        teams = {
            "player": [
                25, 10, 84, 84, 84, 84, 100, 0,  # Pikachu (ID 25) as active
                1, 10, 14, 14, 14, 14, 50, 0,        # Bulbasaur
            ],
            "enemy": [
                21, 10, 45, 45, 45, 45, 80, 0,    # SPEAROW (ID 21) as active
                4, 10, 5, 14, 25, 34, 60, 0,
            ],
        }

        options = {"teams": teams}
        observations, infos = self.arena.reset(options=options)
        self.assertEqual(self.arena.core.state.turn, TurnType.GENERAL)

        player_team_dump = self.arena.core.read_team_data("player")
        enemy_team_dump = self.arena.core.read_team_data("enemy")

        player_df = pokemon_data.to_pandas_team_dump_data(player_team_dump)
        enemy_df = pokemon_data.to_pandas_team_dump_data(enemy_team_dump)

        self.assertEqual(player_df[player_df["isActive"] == 1].iloc[0]["id"], 25)
        self.assertEqual(enemy_df[enemy_df["isActive"] == 1].iloc[0]["id"], 21)

        observation_factory = self.arena.observation_factory
        observations = observation_factory.from_game()
        self.assertIsInstance(observations, Observation)
        self.assertIn("player", observations._o)
        self.assertIn("enemy", observations._o)

        for agent in ["player", "enemy"]:
            obs = observations._o[agent]
            self.assertIsInstance(obs, np.ndarray)
            self.assertTrue(np.issubdtype(obs.dtype, np.integer))
            expected_shape = (6 * ObsIdx.NB_DATA_PKMN,)
            self.assertEqual(obs.shape, expected_shape)

        player_obs = observations._o["player"]
        enemy_obs = observations._o["enemy"]

        player_active_idx = 0
        enemy_active_idx = 0

        # Check species ID
        self.assertEqual(
            player_obs[player_active_idx + ObsIdx.RAW_DATA["species"]], 25
        )
        self.assertEqual(
            enemy_obs[enemy_active_idx + ObsIdx.RAW_DATA["species"]], 21
        )

        self.assertEqual(
            player_obs[player_active_idx + ObsIdx.RAW_DATA["is_active"]], 1
        )
        self.assertEqual(
            enemy_obs[enemy_active_idx + ObsIdx.RAW_DATA["is_active"]], 1
        )

        player_second_idx = ObsIdx.NB_DATA_PKMN
        enemy_second_idx = ObsIdx.NB_DATA_PKMN

        self.assertEqual(
            player_obs[player_second_idx + ObsIdx.RAW_DATA["species"]], 1
        )
        self.assertEqual(
            enemy_obs[enemy_second_idx + ObsIdx.RAW_DATA["species"]], 4
        )

        self.assertEqual(
            player_obs[player_second_idx + ObsIdx.RAW_DATA["is_active"]], 0
        )
        self.assertEqual(
            enemy_obs[enemy_second_idx + ObsIdx.RAW_DATA["is_active"]], 0
        )

        move_start = ObsIdx.RAW_DATA["moves_begin"]
        first_move_offset = move_start + ObsIdx.RAW_DATA["move_id_offset"]
        power_offset = move_start + ObsIdx.RAW_DATA["power_offset"]

        self.assertEqual(player_obs[player_active_idx + first_move_offset], 84)  # Move ID
        # power for Thunder Shock expected 40 in observation representation
        self.assertEqual(player_obs[player_active_idx + power_offset], 40)

        self.assertEqual(
            int(player_obs[ObsIdx.RAW_DATA["type_1"]]),
            13,
            "Pikachu type 1 should be ELECTRIK (13)",
        )
        self.assertEqual(
            int(player_obs[ObsIdx.RAW_DATA["type_2"]]),
            13,
            "Pikachu type 2 should be None (0)",
        )

        move_start = ObsIdx.RAW_DATA["moves_begin"]
        first_move_offset = move_start + ObsIdx.RAW_DATA["move_id_offset"]
        power_offset = move_start + ObsIdx.RAW_DATA["power_offset"]
        pp_offset = move_start + ObsIdx.RAW_DATA["pp_offset"] 

        self.assertEqual(player_obs[player_active_idx + first_move_offset], 84)  # Move ID
        self.assertEqual(player_obs[player_active_idx + power_offset], 40)
        self.assertGreaterEqual(player_obs[player_active_idx + pp_offset], 29, "First move should have at least 29 PP, thundershock=30pp")

        for i in range(2, 6):  # Pokémon 3 to 6
            pkmn_start = i * ObsIdx.NB_DATA_PKMN
            self.assertEqual(
                player_obs[pkmn_start + ObsIdx.RAW_DATA["species"]], 0
            )
            self.assertEqual(
                enemy_obs[pkmn_start + ObsIdx.RAW_DATA["species"]], 0
            )


if __name__ == "__main__":
    unittest.main()
