import rustboyadvance_py
from pkmn_rl_arena import SAVE_PATH,ROM_PATH,BIOS_PATH
import pkmn_rl_arena.data.parser
import pkmn_rl_arena.data.pokemon_data

from .battle_state import TurnType

import os
from typing import List

class BattleCore:
    """
    Low-level battle engine interface.
    Handles GBA emulator, memory operations, and stop conditions.
    """

    def __init__(
        self,
        rom_path: str,
        bios_path: str,
        map_path: str,
        steps: int = 32000,
        setup: bool = True,
    ):
        self.rom_path = rom_path
        self.bios_path = bios_path
        self.map_path = map_path
        self.steps = steps
        # Initialize parser and GBA emulator
        self.parser = pkmn_rl_arena.data.parser.MapAnalyzer(map_path)
        self.gba = rustboyadvance_py.RustGba()
        self.gba.load(bios_path, rom_path)

        if setup:
            self.addrs = {}  # filled in fctn below
            self.addrs = self.setup_addresses()
            self.stop_ids = {}  # filled in fctn below
            self.setup_stops()

    def setup_addresses(self):
        """Setup memory addresses from the map file"""
        self.addrs = {
            "stopHandleTurnCreateTeam": int(
                self.parser.get_address("stopHandleTurnCreateTeam"), 16
            ),
            "stopHandleTurn": int(self.parser.get_address("stopHandleTurn"), 16),
            "stopHandleTurnPlayer": int(
                self.parser.get_address("stopHandleTurnPlayer"), 16
            ),
            "stopHandleTurnEnemy": int(
                self.parser.get_address("stopHandleTurnEnemy"), 16
            ),
            "stopHandleTurnEnd": int(self.parser.get_address("stopHandleTurnEnd"), 16),
            "monDataPlayer": int(self.parser.get_address("monDataPlayer"), 16),
            "monDataEnemy": int(self.parser.get_address("monDataEnemy"), 16),
            "playerTeam": int(self.parser.get_address("playerTeam"), 16),
            "enemyTeam": int(self.parser.get_address("enemyTeam"), 16),
            "legalMoveActionsPlayer": int(
                self.parser.get_address("legalMoveActionsPlayer"), 16
            ),
            "legalMoveActionsEnemy": int(
                self.parser.get_address("legalMoveActionsEnemy"), 16
            ),
            "legalSwitchActionsPlayer": int(
                self.parser.get_address("legalSwitchActionsPlayer"), 16
            ),
            "legalSwitchActionsEnemy": int(
                self.parser.get_address("legalSwitchActionsEnemy"), 16
            ),
            "actionDonePlayer": int(self.parser.get_address("actionDonePlayer"), 16),
            "actionDoneEnemy": int(self.parser.get_address("actionDoneEnemy"), 16),
        }
        return self.addrs

    def setup_stops(self):
        """Setup stop addresses for turn handling"""
        self.gba.add_stop_addr(
            self.addrs["stopHandleTurnCreateTeam"],
            1,
            True,
            "stopHandleTurnCreateTeam",
            0,
        )
        self.gba.add_stop_addr(
            self.addrs["stopHandleTurn"], 1, True, "stopHandleTurn", 1
        )
        self.gba.add_stop_addr(
            self.addrs["stopHandleTurnPlayer"], 1, True, "stopHandleTurnPlayer", 2
        )
        self.gba.add_stop_addr(
            self.addrs["stopHandleTurnEnemy"], 1, True, "stopHandleTurnEnemy", 3
        )
        self.gba.add_stop_addr(
            self.addrs["stopHandleTurnEnd"], 1, True, "stopHandleTurnEnd", 4
        )

        # Store stop IDs for different turn types
        self.stop_ids = {
            0: TurnType.CREATE_TEAM,
            1: TurnType.GENERAL,
            2: TurnType.PLAYER,
            3: TurnType.ENEMY,
            4: TurnType.DONE,
        }

    def add_stop_addr(self, addr: int, size: int, read: bool, name: str, stop_id: int):
        """Add a stop address to the GBA emulator"""
        self.gba.add_stop_addr(addr, size, read, name, stop_id)

    def run_to_next_stop(self, max_steps=2000000) -> int:
        """Run the emulator until we hit a stop condition"""
        stop_id = self.gba.run_to_next_stop(self.steps)

        # Keep running if we didn't hit a stop
        while stop_id == -1:
            max_steps -= 1
            if max_steps <= 0:
                raise TimeoutError(
                    "Reached maximum steps without hitting a stop condition"
                )
            stop_id = self.gba.run_to_next_stop(self.steps)

        return stop_id

    def advance_to_next_turn(self) -> TurnType:
        """Advance to the next turn and return current TurnType"""
        turntype = self.stop_ids[self.run_to_next_stop()]
        self._clear_stop_condition(turntype)
        return turntype

    def get_turn_type(self, stop_id: int) -> TurnType:
        """Convert stop ID to turn type"""
        return self.stop_ids.get(stop_id, TurnType.DONE)

    def read_team_data(self, agent: str) -> List[int]:
        """Read team data for specified agent"""
        if agent == "player":
            return self.gba.read_u32_list(self.addrs["monDataPlayer"], 35 * 6)
        elif agent == "enemy":
            return self.gba.read_u32_list(self.addrs["monDataEnemy"], 35 * 6)
        else:
            raise ValueError(f"Unknown agent: {agent}")

    def write_action(self, agent: str, action: int):
        """Write action for specified agent"""
        if agent == "player":
            self.gba.write_u16(self.addrs["actionDonePlayer"], action)
        elif agent == "enemy":
            self.gba.write_u16(self.addrs["actionDoneEnemy"], action)
        else:
            raise ValueError(f"Unknown agent: {agent}")

    def _clear_stop_condition(self, turn_type: TurnType):
        """Clear stop condition to continue execution"""
        match turn_type:
            case TurnType.CREATE_TEAM:
                self.gba.write_u16(self.addrs["stopHandleTurnCreateTeam"], 0)
            case TurnType.GENERAL:
                self.gba.write_u16(self.addrs["stopHandleTurn"], 0)
            case TurnType.PLAYER:
                self.gba.write_u16(self.addrs["stopHandleTurnPlayer"], 0)
            case TurnType.ENEMY:
                self.gba.write_u16(self.addrs["stopHandleTurnEnemy"], 0)
            case TurnType.DONE:
                self.gba.write_u16(self.addrs["stopHandleTurnEnd"], 0)
            case _ :
                raise ValueError(f"Unknown turntype : {turn_type}")

    def write_team_data(self, agent: str, data: List[int]):
        """Write team data for specified agent"""
        if agent == "player":
            self.gba.write_u32_list(self.addrs["playerTeam"], data)
        elif agent == "enemy":
            self.gba.write_u32_list(self.addrs["enemyTeam"], data)
        else:
            raise ValueError(f"Unknown agent: {agent}")

    def save_savestate(self, name: str) -> str:
        """Save the current state of the emulator"""
        os.makedirs(SAVE_PATH, exist_ok=True)
        save_path = os.path.join(SAVE_PATH, f"{name}.savestate")
        self.gba.save_savestate(save_path)
        return save_path

    def load_savestate(self, name: str) -> bool:
        """Load a saved state"""
        save_path = os.path.join(SAVE_PATH, f"{name}.savestate")
        if os.path.exists(save_path):
            self.gba.load_savestate(save_path, BIOS_PATH, ROM_PATH)
            return True
        else:
            print(f"Save state {save_path} does not exist.")
            return False
