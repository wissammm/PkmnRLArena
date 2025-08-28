from .action import ActionManager
from .battle_core import BattleCore
from .battle_state import BattleState
from .observation import ObservationFactory
from .pkmn_team_factory import PkmnTeamFactory
from .save_state import SaveStateManager
from pkmn_rl_arena import POKEMON_CSV_PATH, MOVES_CSV_PATH, ROM_PATH, BIOS_PATH, MAP_PATH

from pkmn_rl_arena.logging import logger

from enum import Enum
from typing import Any, Dict, Optional
import functools

from gymnasium.spaces import Discrete
import numpy as np
from pettingzoo import AECEnv
from rich.console import Console
from rich.table import Table
from time import sleep


class RenderMode(Enum):
    """Enumeration for different turn types"""

    DISABLED = 1  # No rendering, just raw dogging gpu
    EPISODE_TERMINAL = 2  # Basic rendering enabed, returns a simple table for each episode with final team status and who won
    TURN_TERMINAL = 3  # Basic rendering enabed, returns a simple table for each turn with team status
    RUSTBOY = 4  # show a picture of each turn rendered in rustboy advance


class Arena(AECEnv):
    """
    This class describes the pokemon battle environment for MARL
    It handles :
        Observation Space
        Action Space
        Rewards

    The action is structured around episodes : a whole pokemon team battle
    """

    metadata = {
        "name": "pkmn_daycare_v0.1",
    }

    def __init__(
        self,
        battle_core: BattleCore,
        max_steps_per_episode: int = 1000,
        render_mode: RenderMode = RenderMode.DISABLED,
    ):
        # Initialize core components
        self.core = battle_core
        self.observation_factory = ObservationFactory(self.core)
        self.action_manager = ActionManager(self.core)
        self.battle_state = BattleState()
        self.team_factory = PkmnTeamFactory(POKEMON_CSV_PATH, MOVES_CSV_PATH)

        self.save_state_manager = SaveStateManager(self.core)

        # Environment configuration
        self.agents = ["player", "enemy"]
        self.action_space_size = 10
        self.observations = {agent: np.array([], dtype=int) for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.episode_steps = 0
        self.max_steps_per_episode = max_steps_per_episode  # Configurable
        self.infos = {}

        # render console
        self.console = Console()

        
        sleep(2)
        self.save_state_manager.save_state(
            "boot_state"
        )  # creating 1st save state directly might be not optimal. maybe add a wait to run until 1st stop
        logger.info(f"Created save_state : {self.save_state_manager.list_save_states()}")

    def load_save_state(self, options: Dict[str, str] | None = None):
        if options is None:
            raise ValueError(
                "Called load_save_state without any option. No save state will be loaded."
            )
        loaded = self.save_state_manager.load_state(options.get("save_state"))

        if not loaded:
            raise RuntimeError(f"Failed to load save state: {save_state}")

        return

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = {"save_state": "boot_state"},
    ):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()

        args:
            seed : to generate teams
            options : Dictionnary : "save_state" : path to save state
        """
        # TODO Implement seed args

        logger.info("Resetting env")
        if options is None:
            logger.debug("No options given")
        # Load save state if provided
        if not self.load_save_state(options):
            self.core = BattleCore(ROM_PATH, BIOS_PATH, MAP_PATH)

        # Reset managers
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.episode_steps = 0

        # reset battle
        self.battle_state = BattleState()

        # create new teams
        teams = {agent: self.team_factory.create_random_team() for agent in self.agents}
        self.core.write_team_data(teams)

        # Advance to first turn to  get initial observations
        self.core.advance_to_next_turn()
        self.observations = self.observation_factory.from_game()

        # clean rendering
        self.console.clear()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        self.infos = {a: {} for a in self.agents}

        return  # observations, infos

    def step(self, actions: Optional[Dict[str, int]]):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()

        Args:
            actions: Dictionary of actions for each agent

        Returns:
            observations: New observations for each agent
            rewards: Rewards for each agent (placeholder)
            done: Whether the episode is finished
            info: Additional information
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # if previous step terminated the episode then calling ded step
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            return self._was_dead_step(actions)

        # Get dummy infos (not used in this example)
        # Validate actions
        for agent, action in actions.items():
            if not ActionManager.is_valid_action(action):
                raise ValueError(f"Invalid action {action} for agent {agent}")

        # Write actions
        self.action_manager.write_actions(self.battle_state.current_turn, actions)
        self.battle_state.current_turn = self.core.advance_to_next_turn()

        # Get observations
        observation = self.observation_factory.from_game()
        self.observations = observation_factory.from_game()

        # Check termination conditions
        # TODO : DEFINE TRUCATIONS & TERMINATIONS
        # TODO : EXTRACT BATTLECORE TESTS
        if self.battle_state.is_battle_done():
            self.terminations = {agent: True for agent in self.agents}
            winner = observation.who_won()
            self.rewards[winner] = 1.0  # placeholder
        elif self.max_steps_per_episode < self.episode_steps:
            self.truncations = {agent: True for agent in self.agents}

        # Calculate rewards (placeholder)
        for agent in actions.items():
            self.rewards[agent] = 0.0
            self._cumulative_rewards[agent] += self.rewards[agent]

        self.episode_steps += 1

        # Prepare info
        # self.infos = {
        #     "current_turn": self.battle_state.get_current_turn(),
        #     "battle_done": self.battle_state.is_battle_done(),
        #     "ste_rewards": self.rewards,
        #     "episode_steps": self.episode_steps,
        #     "max_allowed_steps": self.max_steps_per_episode,
        # }

        self.infos = {}

        return  # self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def render(self):
        """
        Render the current state of the battle using the rich library.

        Args:
            observations: Dictionary containing observation DataFrames for 'player' and 'enemy'.
            csv_path: Path to the CSV file containing Pokémon data.
        """
        # Rendering : 3 options :
        # 1. print images of the game
        #    check Example in project root to see use of gba display and have a nice render
        # 2. Just printing status of each party
        # 3. No redering for faster computation

        # Create a table with two columns: Player and Enemy
        table = Table(
            title="Battle State", show_header=True, header_style="bold magenta"
        )
        table.add_column("Player", justify="center", style="cyan", no_wrap=True)
        table.add_column("Enemy", justify="center", style="red", no_wrap=True)

        # Get the current Pokémon for both player and enemy
        player_current = observations["player"][observations["player"]["isActive"] == 1]
        enemy_current = observations["enemy"][observations["enemy"]["isActive"] == 1]

        # Player's current Pokémon details
        player_current_details = ""
        if not player_current.empty:
            player_mon = player_current.iloc[0]
            player_name = get_pokemon_name(player_mon["id"])
            player_moves = player_mon["moves"]
            player_pp = [
                player_mon["move1_pp"],
                player_mon["move2_pp"],
                player_mon["move3_pp"],
                player_mon["move4_pp"],
            ]
            # Add HP information for the active Pokémon
            player_current_details = f"[bold]{player_name}[/bold] - HP: {player_mon['current_hp']}/{player_mon['max_hp']}\n"
            for move, pp in zip(player_moves, player_pp):
                player_current_details += f"Move {move}: PP {pp}\n"

        # Enemy's current Pokémon details
        enemy_current_details = ""
        if not enemy_current.empty:
            enemy_mon = enemy_current.iloc[0]
            enemy_name = get_pokemon_name(enemy_mon["id"])
            enemy_moves = enemy_mon["moves"]
            enemy_pp = [
                enemy_mon["move1_pp"],
                enemy_mon["move2_pp"],
                enemy_mon["move3_pp"],
                enemy_mon["move4_pp"],
            ]
            # Add HP information for the active Pokémon
            enemy_current_details = f"[bold]{enemy_name}[/bold] - HP: {enemy_mon['current_hp']}/{enemy_mon['max_hp']}\n"
            for move, pp in zip(enemy_moves, enemy_pp):
                enemy_current_details += f"Move {move}: PP {pp}\n"

        # Add current Pokémon details to the table
        table.add_row(player_current_details, enemy_current_details)

        # Player's team Pokémon names and HP
        player_team = observations["player"][observations["player"]["isActive"] != 1]
        player_team_details = ""
        for _, mon in player_team.iterrows():
            mon_name = get_pokemon_name(mon["id"])
            player_team_details += (
                f"{mon_name}: HP {mon['current_hp']}/{mon['max_hp']}\n"
            )

        # Enemy's team Pokémon names and HP
        enemy_team = observations["enemy"][observations["enemy"]["isActive"] != 1]
        enemy_team_details = ""
        for _, mon in enemy_team.iterrows():
            mon_name = get_pokemon_name(mon["id"])
            enemy_team_details += (
                f"{mon_name}: HP {mon['current_hp']}/{mon['max_hp']}\n"
            )

        # Add team details to the table
        table.add_row(player_team_details, enemy_team_details)

        # Print the table to the console
        self.console.print(table)

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return self.observation_manager.get_observations()[agent]

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/

        # (73 params / pkmn) * (6 pkmn / party) = 438
        return Discrete(438)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # We can seed the action space to make the environment deterministic.
        #
        # 4 moves + 5 pkmn switch = 9
        return Discrete(9)

    def close():
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass
