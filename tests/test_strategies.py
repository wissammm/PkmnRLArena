from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.env.battle_arena import BattleArena
from pkmn_rl_arena.policy.strategy import Strategies
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena import log

import picologging as logging
import unittest


class TestSuperEffectiveStrategy(unittest.TestCase):
    def setUp(self):
        log.setLevel(logging.DEBUG)
        core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.arena = BattleArena(core)

    def tearDown(self):
        self.arena.close()

    def test_electric_super_effective(self):
        """
        Test that Electric moves (Thundershock) are detected as super-effective against Water type (Squirtle).
        """
        options = {
            "save_state": "boot_state",
            "teams": {
                # Pikachu (Electric) with Thundershock (Electric move)
                "player": [
                    25,      # Pikachu
                    50,
                    84,      # THUNDERSHOCK (Electric type)
                    84,
                    0,
                    0,
                    100,
                    0,
                ],
                # Squirtle (Water type)
                "enemy": [
                    7,       # Squirtle
                    50,
                    33,      # TACKLE (Normal move)
                    0,
                    0,
                    0,
                    100,
                    0,
                ],
            },
        }

        self.arena.reset(options=options)
        obs = self.arena.observation_factory.from_game()
        
        super_effective_moves = Strategies.get_super_effective_move_indices(obs, "player")
        
        self.assertIn(
            0, 
            super_effective_moves,
            "Thundershock (move index 0) should be super-effective against Water type Squirtle"
        )
        self.assertEqual(
            len(super_effective_moves),
            2,
            "2 moves should be super-effective"
        )
        log.debug(f"Super-effective move indices: {super_effective_moves}")


class TestPriorityKillStrategy(unittest.TestCase):
    def setUp(self):
        log.setLevel(logging.DEBUG)
        core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.arena = BattleArena(core)

    def tearDown(self):
        self.arena.close()

    def test_priority_kill_with_quick_attack(self):
        """
        Test that Quick Attack (priority move) is detected as a kill move against low HP opponent.
        Quick Attack has priority=1 and power=40.
        """
        options = {
            "save_state": "boot_state",
            "teams": {
                # Rattata with Quick Attack (priority move)
                "player": [
                    19,      # Rattata
                    50,
                    98,      # QUICK ATTACK (Normal type, priority=1, power=40)
                    33,      # TACKLE (Normal, no priority)
                    0,
                    0,
                    100,
                    0,
                ],
                # Low HP Pidgey (should be KO'able)
                "enemy": [
                    16,      # Pidgey
                    10,      # Low level
                    33,      # TACKLE
                    0,
                    0,
                    0,
                    5,       # Very low HP (5%)
                    0,
                ],
            },
        }

        self.arena.reset(options=options)
        obs = self.arena.observation_factory.from_game()
        
        priority_kill_moves = Strategies.get_priority_kill_moves(obs, "player")
        
        log.debug(f"Priority kill move indices: {priority_kill_moves}")
        
        self.assertGreater(
            len(priority_kill_moves),
            0,
            "Quick Attack should be detected as a priority kill move against low HP opponent"
        )
        
        # Quick Attack is at index 0
        self.assertIn(
            0,
            priority_kill_moves,
            "Quick Attack (index 0) should be in priority kill moves"
        )

    def test_no_priority_kill_when_opponent_healthy(self):
        """
        Test that priority moves are NOT suggested when opponent has high HP.
        Even with priority, the damage won't KO a healthy target.
        """
        options = {
            "save_state": "boot_state",
            "teams": {
                # Rattata with Quick Attack
                "player": [
                    19,      # Rattata
                    50,
                    98,      # QUICK ATTACK (priority=1)
                    0,
                    0,
                    0,
                    100,
                    0,
                ],
                # Healthy high-level Snorlax (lots of HP)
                "enemy": [
                    143,     # Snorlax (high HP Pokemon)
                    80,      # High level
                    33,      # TACKLE
                    0,
                    0,
                    0,
                    100,     # Full HP
                    0,
                ],
            },
        }

        self.arena.reset(options=options)
        obs = self.arena.observation_factory.from_game()
        
        priority_kill_moves = Strategies.get_priority_kill_moves(obs, "player")
        
        log.debug(f"Priority kill moves against healthy Snorlax: {priority_kill_moves}")
        
        self.assertEqual(
            len(priority_kill_moves),
            0,
            "No priority kill moves should be suggested against healthy high-HP opponent"
        )


class TestDefensiveSwitchStrategy(unittest.TestCase):
    def setUp(self):
        log.setLevel(logging.DEBUG)
        core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.arena = BattleArena(core)

    def tearDown(self):
        self.arena.close()

    def test_switch_to_water_vs_fire(self):
        """
        Test that Water Pokemon are suggested as defensive switches against Fire type opponent.
        Water resists Fire (0.5x damage).
        """
        options = {
            "save_state": "boot_state",
            "teams": {
                # Active: Grass type (weak to Fire), Bench: Water type (resists Fire)
                "player": [
                    1,       # Bulbasaur (Grass/Poison) - ACTIVE
                    50,
                    33,      # TACKLE
                    0,
                    0,
                    0,
                    100,
                    0,       # is_active = 1
                    7,       # Squirtle (Water) - BENCH
                    50,
                    33,      # TACKLE
                    0,
                    0,
                    0,
                    100,
                    0,     
                ],
                # Fire type opponent
                "enemy": [
                    4,       # Charmander (Fire type)
                    50,
                    52,      # EMBER (Fire move)
                    0,
                    0,
                    0,
                    100,
                    0,
                ],
            },
        }

        self.arena.reset(options=options)
        obs = self.arena.observation_factory.from_game()
        
        defensive_switches = Strategies.get_defensive_switch_indices(obs, "player")
        
        log.debug(f"Defensive switch indices: {defensive_switches}")
        
        self.assertGreater(
            len(defensive_switches),
            0,
            "Should suggest switching to Water Pokemon against Fire opponent"
        )
        
        # Squirtle is in party slot 1, so switch action index = 4 + 1 = 5
        self.assertIn(
            5,
            defensive_switches,
            "Switch to Squirtle (action 5) should be suggested against Fire type"
        )

    def test_no_defensive_switch_when_no_good_matchup(self):
        """
        Test that no defensive switches are suggested when no Pokemon have type advantage.
        """
        options = {
            "save_state": "boot_state",
            "teams": {
                # All Normal types (neutral to most types)
                "player": [
                    19,      # Rattata (Normal) - ACTIVE
                    50,
                    33,      # TACKLE
                    0,
                    0,
                    0,
                    100,
                    0,       
                    16,      # Pidgey (Normal/Flying) - BENCH
                    50,
                    33,      # TACKLE
                    0,
                    0,
                    0,
                    100,
                    0, 
                    
                ],
                # Fighting type opponent (super-effective vs Normal)
                "enemy": [
                    56,      # Mankey (Fighting type)
                    50,
                    7,       # KARATE CHOP (Fighting)
                    0,
                    0,
                    0,
                    100,
                    0,
                ],
            },
        }

        self.arena.reset(options=options)
        obs = self.arena.observation_factory.from_game()
        
        defensive_switches = Strategies.get_defensive_switch_indices(obs, "player")
        
        log.debug(f"Defensive switches when no good matchup: {defensive_switches}")
        
        # Normal types don't resist Fighting, so no defensive switches should be suggested
        self.assertEqual(
            len(defensive_switches),
            0,
            "No defensive switches should be suggested when no Pokemon have type advantage"
        )


if __name__ == "__main__":
    unittest.main()