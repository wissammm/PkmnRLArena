from pkmn_rl_arena.env.observation import ObsIdx, Observation
from pkmn_rl_arena.env.pokemon_rl_core import PokemonRLCore
from pkmn_rl_arena.env.battle_state import TurnType
from pkmn_rl_arena.env.pkmn_team_factory import PkmnTeamFactory
import pkmn_rl_arena.data.parser
import pkmn_rl_arena.data.pokemon_data
import numpy as np

from pkmn_rl_arena import (
    ROM_PATH,
    BIOS_PATH,
    MAP_PATH,
    POKEMON_CSV_PATH,
    MOVES_CSV_PATH,
)
import unittest
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)


MAIN_STEPS = 64000


class TestPokemonRLCore(unittest.TestCase):
    def setUp(self):
        self.core = PokemonRLCore(ROM_PATH, BIOS_PATH, MAP_PATH)

    def test_advance_to_next_turn(self):
        # self.core.reset()
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.CREATE_TEAM)

    def test_create_team(self):
        team_factory = PkmnTeamFactory(POKEMON_CSV_PATH, MOVES_CSV_PATH)
        teams = {
            "player": team_factory.create_random_team(),
            "enemy": team_factory.create_random_team(),
        }

        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.CREATE_TEAM)

        self.core.battle_core.write_team_data(teams)

        addr_player = int(self.core.battle_core.parser.get_address("playerTeam"), 16)
        read_player_team = self.core.battle_core.gba.read_u32_list(addr_player, 8 * 6)
        print(teams["player"])
        print(read_player_team)
        # self.assertEqual(read_player_team, player_team, "Player team data mismatch")

        addr_enemy = int(self.core.battle_core.parser.get_address("enemyTeam"), 16)
        read_enemy_team = self.core.battle_core.gba.read_u32_list(addr_enemy, 8 * 6)

        player_team = teams["player"]
        enemy_team = teams["enemy"]
        # compare buffer read from gba with player_team
        for i in range(6):
            start = i * 8
            self.assertEqual(
                read_player_team[start],
                teams["player"][start],
                f"Player team ID mismatch at pokemon {i}, read {read_player_team[start]} vs {teams['player'][start]}",
            )
            self.assertEqual(
                read_player_team[start + 1],
                teams["player"][start + 1],
                f"Player team level mismatch at pokemon {i}",
            )
            self.assertEqual(
                read_player_team[start + 2 : start + 6],
                teams["player"][start + 2 : start + 6],
                f"Player team moves mismatch at pokemon {i}",
            )
            self.assertEqual(
                read_player_team[start + 7],
                teams["player"][start + 7],
                f"Player team item mismatch at pokemon {i}",
            )

        print(f"Read enemy team OK")

        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)

        player_team_dump_data = self.core.battle_core.read_team_data("player")
        enemy_team_dump_data = self.core.battle_core.read_team_data("enemy")

        print(player_team_dump_data)

        playerdf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            player_team_dump_data
        )
        enemydf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            enemy_team_dump_data
        )

        print(playerdf)

        for i in range(6):
            start = i * 8
            self.assertEqual(
                playerdf.iloc[i]["id"],
                teams["player"][start],
                f"Player team ID mismatch at pokemon {i}, read {playerdf.iloc[i]['id']} vs {teams['player'][start]}",
            )
            self.assertEqual(
                playerdf.iloc[i]["level"],
                teams["player"][start + 1],
                f"Player team level mismatch at pokemon {i}",
            )

        for i in range(6):
            start = i * 8
            self.assertEqual(
                enemydf.iloc[i]["id"],
                teams["enemy"][start],
                f"Enemy team ID mismatch at pokemon {i}",
            )
            self.assertEqual(
                enemydf.iloc[i]["level"],
                teams["enemy"][start + 1],
                f"Enemy team level mismatch at pokemon {i}",
            )

    def test_enemy_lost(self):
        # pokachu lvl 99 using shock wave (86) with 100% accyracy
        teams = {
            "player": [
                25,
                99,
                84,
                84,
                84,
                84,
                100,
                0,  # Pikachu with moves and 100% HP
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,  # Empty slots
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "enemy": [
                7,
                10,
                45,
                45,
                45,
                45,
                10,
                0,  # Squirtle use move 150 wich does nothing 10% HP
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,  # Empty slots
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }

        # This test case Squirtle have 100% chance to death
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.CREATE_TEAM)

        self.core.battle_core.write_team_data(teams)

        # Advance to the first turn
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)

        # Perform a move (e.g., player uses the first move)
        player_action = 0
        enemy_action = 0
        actions = {"player": player_action, "enemy": enemy_action}

        self.core.action_manager.write_actions(turn, actions)

        turn = self.core.battle_core.advance_to_next_turn()
        enemy_team_dump_data = self.core.battle_core.read_team_data("enemy")
        enemydf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            enemy_team_dump_data
        )
        # Print Hp of enemy POkemon active
        active_enemy = enemydf[enemydf["isActive"] == 1]
        enemy_updated_hp = active_enemy.iloc[0]["current_hp"]
        print(f"Enemy updated HP: {enemy_updated_hp}")

        self.assertEqual(turn, TurnType.DONE)
        # self.assertLess(enemy_updated_hp, enemy_initial_hp, "Enemy HP should decrease after a move is used.")

    def test_switch_pokemon(self):
        teams = {
            "player": [
                25,
                1,
                150,
                150,
                150,
                150,
                100,
                0,  # Pikachu  use move 150 wich does nothing
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "enemy": [
                7,
                99,
                74,
                53,
                54,
                55,
                10,
                0,  # Squirtle use move 74 wich does nothing 10% HP
                8,
                99,
                4,
                5,
                9,
                23,
                11,
                0,
                12,
                99,
                4,
                5,
                9,
                23,
                11,
                0,
                12,
                10,
                0,
                0,
                0,
                0,
                10,
                0,
                12,
                10,
                0,
                0,
                0,
                0,
                10,
                0,
                12,
                10,
                0,
                0,
                0,
                0,
                10,
                0,
            ],
        }  # This test case Squirtle have 100% chance to death
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.CREATE_TEAM)

        self.core.battle_core.write_team_data(teams)

        # Advance to the first turn
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)

        player_action = 0
        enemy_action = 5
        actions = {"player": player_action, "enemy": enemy_action}

        self.core.action_manager.write_actions(turn, actions)
        turn = self.core.battle_core.advance_to_next_turn()

        player_team_dump_data = self.core.battle_core.read_team_data("player")
        enemy_team_dump_data = self.core.battle_core.read_team_data("enemy")
        self.assertEqual(turn, TurnType.GENERAL)

        playerdf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            player_team_dump_data
        )
        enemydf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            enemy_team_dump_data
        )
        active_enemy = enemydf[enemydf["isActive"] == 1]
        print(enemydf)
        self.assertEqual(
            len(active_enemy),
            1,
            "There should be exactly one active Pokémon in the enemy team.",
        )
        self.assertEqual(
            active_enemy.iloc[0]["id"],
            8,
            "The active Pokémon in the enemy team should have ID 8.",
        )

    def test_switch_pokemon_when_one_fainted_enemy(self):
        teams = {
            "player": [
                25,
                99,
                84,
                84,
                84,
                84,
                100,
                0,  # Pikachu with moves and 100% HP
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,  # Empty slots
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "enemy": [
                7,
                10,
                45,
                45,
                45,
                45,
                10,
                0,  # Squirtle use move 150 wich does nothing 10% HP
                11,
                10,
                8,
                3,
                4,
                2,
                100,
                0,  # Empty slots
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }

        # This test case Squirtle have 100% chance to death
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.CREATE_TEAM)

        self.core.battle_core.write_team_data(teams)

        # Advance to the first turn
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)

        # Perform a move (e.g., player uses the first move)
        player_action = 0
        enemy_action = 0
        actions = {"player": player_action, "enemy": enemy_action}

        self.core.action_manager.write_actions(turn, actions)

        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.ENEMY)
        enemy_action = 5  # Switch with the [1] mon
        actions = {"enemy": enemy_action}
        self.core.action_manager.write_actions(turn, actions)
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)
        player_team_dump_data = self.core.battle_core.read_team_data("player")
        enemy_team_dump_data = self.core.battle_core.read_team_data("enemy")
        playerdf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            player_team_dump_data
        )
        enemydf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            enemy_team_dump_data
        )
        active_enemy = enemydf[enemydf["isActive"] == 1]
        self.assertEqual(
            len(active_enemy),
            1,
            "There should be exactly one active Pokémon in the enemy team.",
        )
        self.assertEqual(
            active_enemy.iloc[0]["id"],
            11,
            "The active Pokémon in the enemy team should have ID 11.",
        )

    def test_switch_pokemon_when_one_fainted_player(self):
        teams = {
            "player": [
                7,
                2,
                45,
                45,
                45,
                45,
                10,
                0,
                26,
                10,
                8,
                3,
                4,
                2,
                100,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "enemy": [
                25,
                10,
                84,
                84,
                84,
                84,
                100,
                0,  # Pikachu with moves and 100% HP
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }

        # This test case Pikachu has 100% chance to faint
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.CREATE_TEAM)

        self.core.battle_core.write_team_data(teams)

        # Advance to the first turn
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)

        # Both use first move (Pikachu will faint)
        player_action = 0
        enemy_action = 0
        actions = {"player": player_action, "enemy": enemy_action}

        self.core.action_manager.write_actions(turn, actions)

        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.PLAYER)
        player_action = 5  # Switch with the [1] mon (Bulbasaur)
        actions = {"player": player_action}
        self.core.action_manager.write_actions(turn, actions)
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)
        player_team_dump_data = self.core.battle_core.read_team_data("player")
        enemy_team_dump_data = self.core.battle_core.read_team_data("enemy")
        playerdf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            player_team_dump_data
        )
        enemydf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            enemy_team_dump_data
        )
        active_player = playerdf[playerdf["isActive"] == 1]
        self.assertEqual(
            len(active_player),
            1,
            "There should be exactly one active Pokémon in the player team.",
        )
        self.assertEqual(
            active_player.iloc[0]["id"],
            26,
            "The active Pokémon in the player team should have ID 26.",
        )

    def test_stats_change(self):
        teams = {
            "player": [
                8, 99, 45, 45, 45, 45, 10, 0,
                8, 99, 8, 3, 4, 2, 100,  0,    
                0, 10, 0, 0, 0, 0, 0,0,
                0, 10, 0, 0, 0, 0, 0,0,
                0, 10, 0, 0, 0, 0, 0,0,
                0, 10, 0, 0, 0, 0, 0,0
            ],
            "enemy": [
                7, 99, 45, 45, 45, 45, 100,0,  
                0, 10, 0, 0, 0, 0, 0,0,
                0, 10, 0, 0, 0, 0, 0,0,
                0, 10, 0, 0, 0, 0, 0,0,
                0, 10, 0, 0, 0, 0, 0,0,
                0, 10, 0, 0, 0, 0, 0,0
            ], 
        }
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.CREATE_TEAM)

        self.core.battle_core.write_team_data(teams)

        # Advance to the first turn
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)

        player_team_dump_data = self.core.battle_core.read_team_data("player")
        playerdf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            player_team_dump_data
        )
        
        enemy_team_dump_data = self.core.battle_core.read_team_data("enemy")
        enemydf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            enemy_team_dump_data
        )

        print("Player dataframe:")
        print(playerdf)
        print("Enemy dataframe:")
        print(enemydf)
        print("isActive column values (player):", playerdf["isActive"].values)
        print("isActive column values (enemy):", enemydf["isActive"].values)

        active_player = playerdf[playerdf["isActive"] == 1]
        active_enemy = enemydf[enemydf["isActive"] == 1]
        
        initial_player_attack = active_player.iloc[0]["baseAttack"]
        initial_enemy_attack = active_enemy.iloc[0]["baseAttack"]
        
        print(f"Initial Player Attack: {initial_player_attack}")
        print(f"Initial Enemy Attack: {initial_enemy_attack}")

        player_action = 0
        enemy_action = 0
        actions = {"player": player_action, "enemy": enemy_action}
        self.core.action_manager.write_actions(turn, actions)

        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)

        player_team_dump_data = self.core.battle_core.read_team_data("player")
        playerdf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            player_team_dump_data
        )
        active_player = playerdf[playerdf["isActive"] == 1]
        new_player_attack = active_player.iloc[0]["baseAttack"]
        
        enemy_team_dump_data = self.core.battle_core.read_team_data("enemy")
        enemydf = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(
            enemy_team_dump_data
        )
        active_enemy = enemydf[enemydf["isActive"] == 1]
        new_enemy_attack = active_enemy.iloc[0]["baseAttack"]

        print(f"Player Attack: {initial_player_attack} -> {new_player_attack}")
        print(f"Enemy Attack: {initial_enemy_attack} -> {new_enemy_attack}")
        
        self.assertLess(new_player_attack, initial_player_attack, 
                        "Player's attack stat should be lower after using a stat-lowering move")
        self.assertLess(new_enemy_attack, initial_enemy_attack, 
                        "Enemy's attack stat should be lower after using a stat-lowering move")




    # def test_special_moves():
    #     #ROAR FLEE FLY MULTIMOVE MULTIHIT ENCORE move 5 also
    #     pass
    # def test_status():
    #     pass

    # def test_all_moves():
    #     # # Test all moves
    #     pass

    def test_observation_space(self):
        teams = {
            "player": [
                25, 10, 84, 84, 84, 84, 100, 0,  # Pikachu (ID 25) as active
                1, 10, 1, 2, 3, 4, 50, 0,        # Bulbasaur as backup
                0, 0, 0, 0, 0, 0, 0, 0,          # Empty slots
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            "enemy": [
                7, 10, 45, 45, 45, 45, 80, 0,    # Squirtle (ID 7) as active
                4, 10, 5, 6, 7, 8, 60, 0,        # Charmander as backup
                0, 0, 0, 0, 0, 0, 0, 0,          # Empty slots
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
        }
        
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.CREATE_TEAM)
        self.core.battle_core.write_team_data(teams)
        
        turn = self.core.battle_core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)
        
        player_team_dump = self.core.battle_core.read_team_data("player")
        enemy_team_dump = self.core.battle_core.read_team_data("enemy")
        
        player_df = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(player_team_dump)
        enemy_df = pkmn_rl_arena.data.pokemon_data.to_pandas_team_dump_data(enemy_team_dump)
        
        self.assertEqual(player_df[player_df["isActive"] == 1].iloc[0]["id"], 25)
        self.assertEqual(enemy_df[enemy_df["isActive"] == 1].iloc[0]["id"], 7)

        observation_factory = self.core.observation_factory
        observations = observation_factory.from_game()
        self.assertIsInstance(observations, Observation)
        self.assertIn("player", observations._o)
        self.assertIn("enemy", observations._o)
        
        for agent in ["player", "enemy"]:
            obs = observations._o[agent] 
            self.assertIsInstance(obs, np.ndarray)
            self.assertEqual(obs.dtype, int)
            expected_shape = (6 * ObsIdx.NB_DATA_PKMN,)
            self.assertEqual(obs.shape, expected_shape)
        
        player_obs = observations._o["player"]
        enemy_obs = observations._o["enemy"]
        
        player_active_idx = 0
        enemy_active_idx = 0
        
        # Check species ID
        self.assertEqual(player_obs[player_active_idx + ObsIdx.RAW_DATA["species"]], 25)
        self.assertEqual(enemy_obs[enemy_active_idx + ObsIdx.RAW_DATA["species"]], 7)
        
        self.assertEqual(player_obs[player_active_idx + ObsIdx.RAW_DATA["is_active"]], 1)
        self.assertEqual(enemy_obs[enemy_active_idx + ObsIdx.RAW_DATA["is_active"]], 1)
        
        player_second_idx = ObsIdx.NB_DATA_PKMN  
        enemy_second_idx = ObsIdx.NB_DATA_PKMN
        
        self.assertEqual(player_obs[player_second_idx + ObsIdx.RAW_DATA["species"]], 1)
        self.assertEqual(enemy_obs[enemy_second_idx + ObsIdx.RAW_DATA["species"]], 4)
        
        self.assertEqual(player_obs[player_second_idx + ObsIdx.RAW_DATA["is_active"]], 0)
        self.assertEqual(enemy_obs[enemy_second_idx + ObsIdx.RAW_DATA["is_active"]], 0)
        
        move_start = ObsIdx.RAW_DATA["moves_begin"]
        first_move_offset = move_start + ObsIdx.RAW_DATA["move_id_offset"]
        power_offset = move_start + ObsIdx.RAW_DATA["power_offset"]
        
        self.assertEqual(player_obs[player_active_idx + first_move_offset], 84)  # Move ID
        self.assertEqual(player_obs[player_active_idx + power_offset], 40)  # Power of Thunder Shock
        
        for i in range(2, 6):  # Pokémon 3 to 6
            pkmn_start = i * ObsIdx.NB_DATA_PKMN
            self.assertEqual(player_obs[pkmn_start + ObsIdx.RAW_DATA["species"]], 0)
            self.assertEqual(enemy_obs[pkmn_start + ObsIdx.RAW_DATA["species"]], 0)


if __name__ == "__main__":
    unittest.main()
