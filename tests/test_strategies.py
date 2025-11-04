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


if __name__ == "__main__":
    unittest.main()