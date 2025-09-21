import rustboyadvance_py
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena import log
import pkmn_rl_arena.data.parser
import pkmn_rl_arena.data.pokemon_data
from pkmn_rl_arena.env.turn_type import TurnType
from dataclasses import dataclass
import os
from typing import Dict, List, Optional


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

    def __eq__(self, other) -> bool:
        """This is a helper for tests, it doesn't tests the id voluntarly ad it is unique."""
        return self.step == other.step and self.turn == other.turn


class BattleStateFactory:
    id_gen: int = 0  # unique episode id  generator
    id: int = 0  # unique id for this run

    def build(self, current_turn=TurnType.CREATE_TEAM, step=0) -> BattleState:
        if step < 0:
            raise ValueError(
                f"Attempting to create a step with negative step, step value must be strictly > 0, got {step}"
            )
        id = BattleStateFactory.id_gen
        BattleStateFactory.id_gen += 1
        return BattleState(id=id, step=step, turn=current_turn)

    @staticmethod
    def from_save_path(save_path: str) -> BattleState:
        """
        Creates state from save path, assuming save path is of shape :
        {save_name}_turntype:{turntype.value}_step:{step}_id:{id}.savestate
        """
        data = save_path.split(".")[-2].split("_")[-3:]

        turntype = TurnType(int(data[0][data[0].find(":") + 1 :]))
        step = int(data[1][data[1].find(":") + 1 :])
        id = int(data[2][data[2].find(":") + 1 :])

        return BattleState(id, step, turntype)


class BattleCore:
    """
    Low-level battle engine interface.
    Handles GBA emulator, memory operations, and stop conditions.
    """

    stop_address_names = [
        "stopHandleTurnCreateTeam",
        "stopHandleTurn",
        "stopHandleTurnPlayer",
        "stopHandleTurnEnemy",
        "stopHandleTurnEnd",
    ]
    data_address_names = ["monDataPlayer", "monDataEnemy"]
    team_address_names = ["playerTeam", "enemyTeam"]
    legal_actions_address_names = [
        "legalMoveActionsPlayer",
        "legalMoveActionsEnemy",
        "legalSwitchActionsPlayer",
        "legalSwitchActionsEnemy",
    ]
    action_done_address_names = ["actionDonePlayer", "actionDoneEnemy"]

    def __init__(
        self,
        rom_path: str,
        bios_path: str,
        map_path: str,
        steps: int = 32000,
        setup: bool = True,
        run_until_first_stop=True,
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

        if run_until_first_stop:
            self.advance_to_next_turn(count_step=False)

    def setup_addresses(self):
        """Setup memory addresses from the map file"""
        self.addrs = {
            addr_name: int(self.parser.get_address(addr_name), 16)
            for addr_name in self.stop_address_names
            + self.data_address_names
            + self.team_address_names
            + self.legal_actions_address_names
            + self.action_done_address_names
        }

        return self.addrs

    def setup_stops(self):
        """Setup stop addresses for turn handling"""
        for i, address_name in enumerate(self.stop_address_names):
            self.gba.add_stop_addr(self.addrs[address_name], 1, True, address_name, i)

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

    def run_to_next_stop(self, max_steps=2000000, count_step=True) -> int:
        """
        Run the emulator until we hit a stop condition and updates game state.
        Args:
            max_steps : number of steps to run in the gba before timeout
        Return:
            stop_id : -1 if uncatched error,
        Raises :
            TimeOutError : if max steps < nb of steps executed to run to next stop
        """
        stop_id = self.gba.run_to_next_stop(self.steps)

        # Keep running until we hit a stop
        while stop_id == -1:
            max_steps -= 1
            if max_steps <= 0:
                raise TimeoutError(
                    "Reached maximum steps without hitting a stop condition"
                )
            stop_id = self.gba.run_to_next_stop(self.steps)

        # update state
        self.state.turn = self.stop_ids[stop_id]
        if count_step:
            self.state.step += 1

        return stop_id

    def advance_to_next_turn(self, count_step=True) -> TurnType:
        """Advance to the next turn and return current TurnType"""
        self.run_to_next_stop(count_step=count_step)
        self._clear_stop_condition(self.state.turn)
        return self.state.turn

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
            if agent not in authorized_agents:
                raise ValueError(
                    f'Error: write_team_data : Invalid agent, expected either {authorized_agents}, got "{agent}".'
                )
            self.gba.write_u32_list(self.addrs[f"{agent}Team"], team)
        return

    def save_savestate(self, save_path: str) -> str:
        """Save the current state of the emulator in PATHS["SAVE"]"""
        os.makedirs(PATHS["SAVE"], exist_ok=True)
        save_path = os.path.join(PATHS["SAVE"], f"{save_path}")
        self.gba.save_savestate(save_path)
        return save_path

    def load_savestate(self, name: str) -> Optional[BattleState]:
        """Load a saved state
        Args :
            name : str = Save state name.
                         The name must not be prefixed by PATHS["SAVE"]
        """
        save_path = os.path.join(PATHS["SAVE"], name)
        if not os.path.exists(save_path):
            print(f"Save state {save_path} does not exist.")
            return None

        log.info(f"Loading following save state : {save_path}")
        self.gba.load_savestate(save_path, PATHS["BIOS"], PATHS["ROM"])
        self.setup_addresses()
        self.setup_stops()

        self.state = BattleStateFactory.from_save_path(save_path)
        log.debug(f"Successfully loaded save state, current state is now {self.state}.")
        return self.state

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
