from pkmn_rl_arena.env.observation import ObsIdx
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena import log
from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.battle_state import BattleState
from pkmn_rl_arena.env.battle_arena import BattleArena
from pkmn_rl_arena.env.pkmn_team_factory import DataSize
from pkmn_rl_arena.data import pokemon_data
from pkmn_rl_arena.env.turn_type import TurnType


from pettingzoo.test import parallel_api_test
import numpy as np
import picologging as logging

import copy
import random
import unittest


class TestArena(unittest.TestCase):
    def setUp(self):
        log.setLevel(logging.DEBUG)
        core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.arena = BattleArena(core)

    def tearDown(self):
        self.arena.close()

    def test_reset(self):
        observations, infos = self.arena.reset()
        self.assertEqual(
            self.arena.core.state, BattleState(id=0, step=0, turn=TurnType.GENERAL)
        )
        for agent, value in self.arena.terminations.items():
            self.assertFalse(
                value,
                f"Wrong value after env reset, expected {False} for env.termination, got {self.arena.terminations}.",
            )
        for agent, value in self.arena.truncations.items():
            self.assertFalse(
                value,
                f"Wrong value after env reset, expected {False} for env.truncations, got {self.arena.truncations}.",
            )

        for agent, value in self.arena.rewards.items():
            self.assertEqual(value, 0.0)
        for agent, value in self.arena._cumulative_rewards.items():
            self.assertEqual(value, 0.0)
        self.assertEqual(
            len(self.arena.reward_manager.obs),
            1,
            f"Length of observation list of the reward manager after reset should be 1, got {len(self.arena.reward_manager.obs)}.",
        )

    def test_step(self):
        observations, infos = self.arena.reset()
        previous_observations = observations
        states = [self.arena.core.state]

        for i in range(20):
            actions = {
                agent: random.choice(
                    self.arena.action_manager.get_valid_action_ids(agent)
                )
                for agent in self.arena.core.get_required_agents()
            }

            observations, rewards, terminations, truncations, infos = self.arena.step(
                actions
            )
            states.append(copy.deepcopy(self.arena.core.state))
            self.assertEqual(
                self.arena.core.state.step,
                i + 1,
                f"Invalid step number, step amount should be {i} (current step) + 1 (step executed in the current loop)  = {i + 2}, got {self.arena.core.state.step}.",
            )
            self.assertEqual(
                len(self.arena.reward_manager.obs),
                i + 2,
                f"Invalid length of observation state : observation should be i + 1(initial observation) + 1(step completed in the current loop) = {i + 1}, got {len(self.arena.reward_manager.obs)}.",
            )

            if not (
                (
                    previous_observations["player"]["observation"]
                    == observations["player"]["observation"]
                ).all()
                and (
                    previous_observations["enemy"]["observation"]
                    == observations["enemy"]["observation"]
                ).all()
            ):
                log.fatal(
                    f"No observation was updated for at step {self.arena.core.state.step}, at least one must be updated(if not both). For debugging : previous state : {states[-2]}, Current state : {states[-1]}."
                )
                for agent in self.arena.agents:
                    log.fatal(
                        f"\nCurrent observation for {agent}:\n {observations[agent]['observation']}\nPrevious observation :\n {previous_observations[agent]['observation']}"
                    )
                self.assertFalse(
                    True,
                    f"No observation was updated for at turn {i}, at least one must be updated(if not both). For debugging : previous state : {states[-2]}, Current state : {states[-1]}.",
                )

            previous_observations = observations

    # def test_render(self):
    #     self.arena.reset(seed=42)
    #     self.arena.render()


class TestPettingZooAPI(unittest.TestCase):
    def setUp(self):
        log.setLevel(logging.DEBUG)
        core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.arena = BattleArena(core)

    def test(self):
        parallel_api_test(self.arena, num_cycles=20)

    def tearDown(self):
        self.arena.close()


class TestResetOptions(unittest.TestCase):
    def setUp(self):
        log.setLevel(logging.DEBUG)
        core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.arena = BattleArena(core)

    def tearDown(self):
        self.arena.close()

    def test_load_savestate(self):
        """
        Testing :
            1. Generate random team data (8*6 params)
            2. Writes it to gba
            5. Read pkmn teams data generated by the game from pkmn params
            6. Ensures data matches grondtruth

        Functions called:
            - PkmnTeamFactory.Create_random_team()
            - BattleArena.reset() with a random generated team()
            - pkmn_data.to_pandas_team_dump_data()
            - BattleArena.create_teams()

        Ground Truth :
            - Team generated with team_factory
        """
        options = {"save_state": "boot_state", "teams": None}
        self.arena.reset(options=options)
        # state id of 0 means that BattleStateFactory.build() hasn't been called & has not created a new state
        self.assertEqual(
            self.arena.core.state, BattleState(id=0, step=0, turn=TurnType.GENERAL)
        )
        return

    def test_create_team(self):
        """
        Testing :
            1. Generate random team data (8*6 params)
            2. Writes it to gba
            5. Read pkmn teams data generated by the game from pkmn params
            6. Ensures data matches grondtruth

        Functions called:
            - PkmnTeamFactory.Create_random_team()
            - BattleArena.reset() with a random generated team()
            - pkmn_data.to_pandas_team_dump_data()
            - BattleArena.create_teams()

        Ground Truth :
            - Team generated with team_factory
        """
        options = {
            "teams": {
                "player": self.arena.team_factory.create_random_team(),
                "enemy": self.arena.team_factory.create_random_team(),
            },
        }
        self.arena.reset(options=options)
        self.assertEqual(self.arena.core.state.turn, TurnType.GENERAL)

        for agent in self.arena.possible_agents:
            ground_truth_team_params = options["teams"][agent]

            gba_read_team_data = self.arena.core.read_team_data(agent)
            gba_read_team_df = pokemon_data.to_pandas_team_dump_data(gba_read_team_data)

            for i in range(6):
                start = i * DataSize.PKMN
                self.assertEqual(
                    gba_read_team_df.iloc[i]["id"],
                    ground_truth_team_params[start],
                    f"{agent} team ID mismatch at pokemon {i}.",
                )
                self.assertEqual(
                    gba_read_team_df.iloc[i]["level"],
                    ground_truth_team_params[start + 1],
                    f"{agent} team level mismatch at pokemon {i}.",
                )

    def test_reset_with_invalid_pkmn_params(self):
        """
        Trying to create a squirtle with GROWL even if its not in squirtle movepool
        """
        options = {
            "save_state": "boot_state",
            "teams": {
                "player": [
                    7,
                    2,
                    45,
                    45,
                    45,
                    45,
                    10,
                    0,
                ],
                "enemy": None,
            },
        }

        with self.assertRaises(ValueError) as context_manager:
            self.arena.reset(options=options)
        self.assertEqual(
            str(context_manager.exception), 'Invalid reset param : "team".')
    
    def test_team_size(self):
        """
        Test that teams can be created with different sizes (1-6 Pokémon).
        Verifies:
        1. Team creation with different sizes
        2. Correct number of active Pokémon in the team
        3. Empty slots are properly filled
        4. Only one Pokémon is active per team
        """
        
        for team_size in range(1, 7):
            options = {
                "save_state": "boot_state", 
                "teams": None,
                "team_size": team_size
            }
            
            observations, infos = self.arena.reset(options=options)
            self.assertEqual(self.arena.core.state.turn, TurnType.GENERAL)
            
            for agent in self.arena.possible_agents:
                team_data = self.arena.core.read_team_data(agent)
                team_df = pokemon_data.to_pandas_team_dump_data(team_data)
                
                active_pokemon_count = len(team_df[team_df["isActive"] == 1])
                self.assertEqual(
                    active_pokemon_count, 
                    1,
                    f"Expected exactly 1 active Pokémon in {agent}'s team, got {active_pokemon_count}"
                )
                
                non_empty_pokemon = team_df[team_df["id"] > 0]
                self.assertEqual(
                    len(non_empty_pokemon), 
                    team_size,
                    f"Expected {team_size} Pokémon in {agent}'s team, got {len(non_empty_pokemon)}"
                )
                
                empty_slots = team_df[team_df["id"] == 0]
                self.assertEqual(
                    len(empty_slots), 
                    DataSize.PARTY_SIZE - team_size,
                    f"Expected {6 - team_size} empty slots in {agent}'s team, got {len(empty_slots)}"
                )
                    
            for agent in self.arena.agents:
                obs = observations[agent]["observation"]
                self.assertIsInstance(obs, np.ndarray)
                self.assertEqual(
                    obs.shape[0], 
                    DataSize.PARTY_SIZE * ObsIdx.NB_DATA_PKMN,
                    f"Observation shape should always be 6*NB_DATA_PKMN regardless of team size"
                )


class TestFightUnfold(unittest.TestCase):
    """
    The following tests writes directly actions using the action_manager to test

    Its purpose is to test the action manager as well as other lower level apis.

    NOTE: Should be refactored to not use BattleArena nor PkmnRLCore
    """

    def setUp(self):
        log.setLevel(logging.DEBUG)
        core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.arena = BattleArena(core)

    def tearDown(self):
        self.arena.close()

    def test_enemy_lost(self):
        # pikachu lvl 99 using shock wave (86) with 100% accyracy
        options = {
            "save_state": "boot_state",
            "teams": {
                # Pikachu with move thundershock & 100% HP
                "player": [
                    25,
                    99,
                    84,  # THUNDERSHOCK
                    0,
                    0,
                    0,
                    100,
                    0,
                ],
                # Lvl 10 Magikarp with move splash (no effect) & 10% HP
                "enemy": [
                    129,
                    10,
                    150,  # SPLASH
                    0,
                    0,
                    0,
                    10,
                    0,
                ],
            },
        }

        self.arena.reset(options=options)

        obs, rewards, terminations, truncations, infos = self.arena.step(
            actions={"player": 0, "enemy": 0}
        )

        for agent, term in terminations.items():
            self.assertTrue(self.arena.terminations[agent])
            self.assertEqual(
                self.arena.terminations[agent],
                term,
                "returned termination value is {term} not identical to self.arena.terminations[{agent}] = {self.arena.terminations[agent]}",
            )

        obs = self.arena.observation_factory.from_game()
        self.assertEqual(
            obs.who_won(),
            "player",
            f'should have returned "player" as enemy has no HP left. HP left for each pkmn : {obs.hp()}',
        )
        self.assertEqual(
            self.arena.core.state.turn,
            TurnType.DONE,
            "TurnType should be done as fight is over.",
        )

    def test_switch_pokemon(self):
        options = {
            "save_state": None,
            "teams": {
                "player": [
                    129,  # Magikarp lvl 1 with splash wich does nothing
                    1,  # lvl 1
                    150,  # splash
                    150,
                    150,
                    150,
                    100,  # 100% hp
                    0,
                ],
                "enemy": [
                    # Squirtle
                    7,
                    99,  # lvl 99
                    111,  # DEFENSE CURL
                    0,
                    0,
                    0,
                    10,  # 10 % hp
                    0,
                    # WARTORTLE
                    8,
                    99,
                    5,  # MEGAPUNCH
                    5,  # MEGAPUNCH
                    5,  # MEGAPUNCH
                    5,  # MEGAPUNCH
                    11,
                    0,
                ],
            },
        }

        self.arena.reset(seed=None, options=options)

        player_action = 0  # use move defense curl
        enemy_action = 5  #
        actions = {"player": player_action, "enemy": enemy_action}

        self.arena.action_manager.write_actions(actions)
        turn = self.arena.core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)

        enemy_team_dump_data = self.arena.core.read_team_data("enemy")

        enemydf = pokemon_data.to_pandas_team_dump_data(enemy_team_dump_data)
        active_enemy = enemydf[enemydf["isActive"] == 1]

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

    def test_invalid_action(self):
        options = {
            "save_state": "boot_state",
            "teams": {
                "player": [
                    # SQUIRTLE
                    7,
                    2,  # lvl 2
                    5,
                    5,
                    5,
                    5,
                    10,
                    0,
                    # RAICHU
                    26,
                    10,
                    5,
                    5,
                    5,
                    5,
                    100,
                    0,
                ],
                "enemy": [
                    25,  # pikachu
                    50,
                    84,  # Thunderschock
                    84,
                    84,
                    84,
                    100,  # %HP
                    0,
                ],
            },
        }

        # This test case Pikachu has 100% chance to faint
        self.arena.reset(options=options)

        # Both use first move (Pikachu will faint)
        player_action = 0
        enemy_action = 0
        actions = {"player": player_action, "enemy": enemy_action}

        self.arena.action_manager.write_actions(actions)

        self.arena.core.advance_to_next_turn()
        self.assertEqual(
            self.arena.core.state, BattleState(id=0, step=1, turn=TurnType.PLAYER)
        )
        player_action = 4
        actions = {"player": 4}  # Switch with the [1] mon (Raichu)
        written_actions = self.arena.action_manager.write_actions(actions)
        self.assertFalse(
            written_actions["player"],
            "Invalid action written successfully! This should not happen.",
        )

    def test_switch_pokemon_when_one_fainted_player(self):
        options = {
            "save_state": "boot_state",
            "teams": {
                "player": [
                    # SQUIRTLE
                    7,
                    2,  # lvl 2
                    5,
                    5,
                    5,
                    5,
                    10,
                    0,
                    # RAICHU
                    26,
                    10,
                    5,
                    5,
                    5,
                    5,
                    100,
                    0,
                ],
                "enemy": [
                    25,  # pikachu
                    50,
                    84,  # Thunderschock
                    84,
                    84,
                    84,
                    100,  # %HP
                    0,
                ],
            },
        }

        # This test case Pikachu has 100% chance to faint
        self.arena.reset(options=options)

        # Both use first move (Pikachu will faint)
        actions = {"player": 0, "enemy": 0}

        for agent, result in self.arena.action_manager.write_actions(actions).items():
            self.assertTrue(result, "Valid action not written this should not happen.")

        turn = self.arena.core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.PLAYER)

        actions = {"player": 5}  # Switch with the [1] mon (RAICHU)}
        for agent, result in self.arena.action_manager.write_actions(actions).items():
            self.assertTrue(result, "Valid action not written this should not happen.")

        turn = self.arena.core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)

        player_team_dump_data = self.arena.core.read_team_data("player")
        playerdf = pokemon_data.to_pandas_team_dump_data(player_team_dump_data)
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
        options = {
            "save_state": "boot_state",
            "teams": {
                "player": [
                    1,
                    99,
                    45,
                    45,
                    45,
                    45,
                    10,
                    0,
                ],
                "enemy": [
                    1,
                    99,
                    45,
                    45,
                    45,
                    45,
                    100,
                    0,
                ],
            },
        }

        # reset the arena with the specified teams
        self.arena.reset(options=options)
        self.assertEqual(self.arena.core.state.turn, TurnType.GENERAL)

        # read initial stats
        player_team_dump_data = self.arena.core.read_team_data("player")
        playerdf = pokemon_data.to_pandas_team_dump_data(player_team_dump_data)
        enemy_team_dump_data = self.arena.core.read_team_data("enemy")
        enemydf = pokemon_data.to_pandas_team_dump_data(enemy_team_dump_data)

        active_player = playerdf[playerdf["isActive"] == 1]
        active_enemy = enemydf[enemydf["isActive"] == 1]

        initial_player_attack = active_player.iloc[0]["baseAttack"]
        initial_enemy_attack = active_enemy.iloc[0]["baseAttack"]

        # both use their first move (stat-affecting in this test setup)
        actions = {"player": 0, "enemy": 0}
        self.arena.action_manager.write_actions(actions)
        turn = self.arena.core.advance_to_next_turn()
        self.assertEqual(turn, TurnType.GENERAL)

        # read stats after the move
        player_team_dump_data = self.arena.core.read_team_data("player")
        playerdf = pokemon_data.to_pandas_team_dump_data(player_team_dump_data)
        active_player = playerdf[playerdf["isActive"] == 1]
        new_player_attack = active_player.iloc[0]["baseAttack"]

        enemy_team_dump_data = self.arena.core.read_team_data("enemy")
        enemydf = pokemon_data.to_pandas_team_dump_data(enemy_team_dump_data)
        active_enemy = enemydf[enemydf["isActive"] == 1]
        new_enemy_attack = active_enemy.iloc[0]["baseAttack"]

        # assert stats changed as expected
        self.assertLess(
            new_player_attack,
            initial_player_attack,
            "Player's attack stat should be lower after using a stat-lowering move",
        )
        self.assertLess(
            new_enemy_attack,
            initial_enemy_attack,
            "Enemy's attack stat should be lower after using a stat-lowering move",
        )

    # def test_special_moves():
    #     #ROAR FLEE FLY MULTIMOVE MULTIHIT ENCORE move 5 also
    #     pass
    # def test_status():
    #     pass

    # def test_all_moves():
    #     # # Test all moves
    #     pass


if __name__ == "__main__":
    suite = unittest.TestSuite()
