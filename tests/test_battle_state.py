from pkmn_rl_arena.env.battle_core import TurnType
import unittest
from pkmn_rl_arena.env.battle_core import BattleState, BattleStateFactory
from pkmn_rl_arena.env.save_state import SaveStateManager
from pkmn_rl_arena import logger
import picologging as logging


class TestBattleStateFactory(unittest.TestCase):
    def setUp(self):
        logger.setLevel(logging.DEBUG)
        self.factory = BattleStateFactory()
        self.states: list[BattleState] = [
            ("save_1", self.factory.build(current_turn=TurnType.CREATE_TEAM, step=0)),
            ("save_2", self.factory.build(current_turn=TurnType.DONE, step=10000)),
            ("save_3", self.factory.build(current_turn=TurnType.GENERAL, step=10)),
            ("save_4", self.factory.build(current_turn=TurnType.ENEMY, step=2)),
            ("save_5", self.factory.build(current_turn=TurnType.PLAYER, step=3)),
        ]

    def tearDown(self):
        pass

    def test_from_save_path(self):
        for idx, (name, state) in enumerate(self.states):
            logger.debug(f"Test : {name}, state: {state}")
            self.assertEqual(idx, state.id,"Each state id must be unique. This state run id is not unique.")
            save_path = SaveStateManager.buid_save_path(name, state)
            logger.debug(f"save_path = {save_path}")
            built_state = self.factory.from_save_path(save_path)
            self.assertEqual(state,built_state)
