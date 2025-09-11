import rustboyadvance_py
from pkmn_rl_arena import SAVE_PATH, ROM_PATH, BIOS_PATH, logger
import pkmn_rl_arena.data.parser
import pkmn_rl_arena.data.pokemon_data

from dataclasses import dataclass
from enum import Enum
import os
from typing import Dict, List


class TurnType(Enum):
    """Enumeration for different turn types"""

    CREATE_TEAM = "create_team"  # Initial team creation
    GENERAL = "general"  # Both players act simultaneously
    PLAYER = "player"  # Only player acts
    ENEMY = "enemy"  # Only enemy acts
    DONE = "done"  # Battle is finished


@dataclass
class BattleState:
    """
    Represents the current state of the battle
    NOTE :
        DO NOTE USE THIS CLASS CONSTRUCTOR.
        STATES MUST ONLY BE MANUFACTURED THROUGH StateFactory
    """

    id: int = 0  # unique id for this run
    step: int = 0  # nb step in this run
    turn: TurnType = TurnType.CREATE_TEAM  # current battle turntype


class BattleStateFactory:
    id_gen: int = 0  # unique episode id  generator
    id: int = 0  # unique id for this run

    def build_state(self, current_turn=TurnType.CREATE_TEAM, step=0) -> BattleState:
        id = BattleStateFactory.id_gen
        BattleStateFactory.id_gen += 1
        return BattleState(id=id, step=step, turn=current_turn)

    def from_save_path(save_path: str) -> BattleState:
        split_path = save_path.split("_")
        episode_step = split_path[-1]
        episode_step = int(episode_step[episode_step.find(":") :])
        turntype = split_path[-2]
        turntype = int(turntype[turntype.find(":") :])

        return BattleState(current_turn=turntype, step=episode_step)


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

        self.state = BattleState()
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

        self.state.turn = self.stop_ids[stop_id]
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
        match agent:
            case "player":
                return self.gba.read_u32_list(self.addrs["monDataPlayer"], 28 * 6)
            case "enemy":
                return self.gba.read_u32_list(self.addrs["monDataEnemy"], 28 * 6)
            case _:
                raise ValueError(f"Unknown agent: {agent}")

    def write_action(self, agent: str, action: int):
        """Write action for specified agent"""
        match agent:
            case "player":
                self.gba.write_u16(self.addrs["actionDonePlayer"], action)
            case "enemy":
                self.gba.write_u16(self.addrs["actionDoneEnemy"], action)
            case _:
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
            case _:
                raise ValueError(f"Unknown turntype : {turn_type}")

    def write_team_data(self, teams_data: Dict[str, List[int]]):
        """Write team data for specified agent"""
        authorized_agents = ["player", "enemy"]
        for agent, team in teams_data.items():
            if agent not in authorized_agents :
                raise ValueError(
                    f'Error: write_team_data : Invalid agent, expected either {authorized_agents}, got "{agent}".'
                )
            self.gba.write_u32_list(self.addrs[f"{agent}Team"], team)
        return

    def save_savestate(self, name: str) -> str:
        """Save the current state of the emulator"""
        os.makedirs(SAVE_PATH, exist_ok=True)
        save_path = os.path.join(SAVE_PATH, f"{name}.savestate")
        self.gba.save_savestate(save_path)
        return save_path

    def load_savestate(self, name: str) -> bool:
        """Load a saved state
        Args :
            name : str = Save state name.
                         The name must not be prefixed by SAVE_PATH
        """
        save_path = os.path.join(SAVE_PATH, f"{name}.savestate")
        if os.path.exists(save_path):
            logger.info(f"Attempting to load save state : {save_path}")
            self.gba.load_savestate(save_path, BIOS_PATH, ROM_PATH)
            self.setup_addresses()
            self.setup_stops()
            return True
        else:
            print(f"Save state {save_path} does not exist.")
            return False

    def is_episode_done(self) -> bool:
        """Check if battle is finished"""
        return self.state.turn == TurnType.DONE

    def get_current_turn(self) -> TurnType:
        """Get current turn type"""
        return self.state.turn

    def get_required_agents(self) -> List[str]:
        """Get list of agents required for current turn"""
        match self.state.turn:
            case TurnType.GENERAL:
                return ["player", "enemy"]
            case TurnType.PLAYER:
                return ["player"]
            case TurnType.ENEMY:
                return ["enemy"]
            case _:
                return []
