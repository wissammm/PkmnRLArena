from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena import log
import pkmn_rl_arena.data.parser
import pkmn_rl_arena.data.pokemon_data
from pkmn_rl_arena.env.turn_type import TurnType
from pkmn_rl_arena.env.battle_state import BattleState, BattleStateFactory

import rustboyadvance_py

import os
from typing import Dict, List, Optional, Tuple


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
            ## all variables initialized here are filled in fctns below
             
            # Memory addresses that needs to be stored, they are listed by name as static attr of this class
            self.mem_addrs = {} 
            # list of tuples containing all the data required to create a stop address
            # see add_stop_addr method to understand the tuple
            self.stop_addrs  : List[Tuple[int,int,bool,str,int]]= []
            # For each stop address id, associate a TurnType to it
            self.stop_ids = {}
            self.mem_addrs = self.setup_addresses()
            self.setup_stops()

        if run_until_first_stop:
            self.advance_to_next_turn(count_step=False)

    def setup_addresses(self):
        """Setup memory addresses from the map file"""
        self.mem_addrs = {
            addr_names: int(self.parser.get_address(addr_names), 16)
            for addr_names in self.stop_address_names
            + self.data_address_names
            + self.team_address_names
            + self.legal_actions_address_names
            + self.action_done_address_names
        }

        return self.mem_addrs

    def setup_stops(self, init=True):
        """Setup stop addresses for turn handling"""
        if init:
            self.stop_addrs = [
                (self.mem_addrs[address_name], 1, True, address_name, i)
                for i, address_name in enumerate(self.stop_address_names)
            ]
            self.stop_ids = {
                0: TurnType.CREATE_TEAM,
                1: TurnType.GENERAL,
                2: TurnType.PLAYER,
                3: TurnType.ENEMY,
                4: TurnType.DONE,
            }

        self.gba.add_stop_addrs(self.stop_addrs)

    def add_stop_addrs(self, addrs: list[Tuple[int, int, bool, str, int]]):
        """Add a stop address to the GBA emulator"""
        self.gba.add_stop_addrs(addrs)

    def add_stop_addr(self, addr : int, value : int, is_active : bool, name : str, id : int):
        """Add a stop address to the GBA emulator"""
        self.gba.add_stop_addrs(addr , value , is_active , name , id  )

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
                return self.gba.read_u32_list(
                    self.mem_addrs["monDataPlayer"], 28 * 6
                )
            case "enemy":
                return self.gba.read_u32_list(self.mem_addrs["monDataEnemy"], 28 * 6)
            case _:
                raise ValueError(f"Unknown agent: {agent}")

    def write_action(self, agent: str, action: int):
        """Write action for specified agent"""
        match agent:
            case "player":
                self.gba.write_u16(self.mem_addrs["actionDonePlayer"], action)
            case "enemy":
                self.gba.write_u16(self.mem_addrs["actionDoneEnemy"], action)
            case _:
                raise ValueError(f"Unknown agent: {agent}")

    def _clear_stop_condition(self, turn_type: TurnType):
        """Clear stop condition to continue execution"""
        match turn_type:
            case TurnType.CREATE_TEAM:
                self.gba.write_u16(self.mem_addrs["stopHandleTurnCreateTeam"], 0)
            case TurnType.GENERAL:
                self.gba.write_u16(self.mem_addrs["stopHandleTurn"], 0)
            case TurnType.PLAYER:
                self.gba.write_u16(self.mem_addrs["stopHandleTurnPlayer"], 0)
            case TurnType.ENEMY:
                self.gba.write_u16(self.mem_addrs["stopHandleTurnEnemy"], 0)
            case TurnType.DONE:
                self.gba.write_u16(self.mem_addrs["stopHandleTurnEnd"], 0)
            case _:
                raise ValueError(f"Unknown turntype : {turn_type}")
    
    def clear_stop_condition_id(self, id: int):
        """Clear stop condition with id to continue execution"""
        if id in self.stop_ids:
            addr_name = self.stop_address_names[id]
            self.gba.write_u16(self.mem_addrs[addr_name], 0)
        else:
            raise ValueError(f"Unknown stop id : {id}")
    
    def clear_all_stop_conditions(self):
        """Clear all stop conditions to continue execution"""
        for addr_name in self.stop_address_names:
            self.gba.write_u16(self.mem_addrs[addr_name], 0)

    def write_team_data(self, teams_data: Dict[str, List[int]]):
        """Write team data for specified agent"""
        authorized_agents = ["player", "enemy"]
        for agent, team in teams_data.items():
            if agent not in authorized_agents:
                raise ValueError(
                    f'Error: write_team_data : Invalid agent, expected either {authorized_agents}, got "{agent}".'
                )
            self.gba.write_u32_list(self.mem_addrs[f"{agent}Team"], team)
        return

    def save_savestate(self, save_path: str) -> str:
        """Save the current state of the emulator in PATHS["SAVE"]"""
        os.makedirs(PATHS["SAVE"], exist_ok=True)
        save_path = os.path.join(PATHS["SAVE"], f"{save_path}")
        self.gba.save_savestate(save_path)
        return save_path

    def load_savestate(self, name: str, init: bool = False) -> Optional[BattleState]:
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
        self.gba.load_savestate(save_path, PATHS["BIOS"], PATHS["ROM"], self.stop_addrs)

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
