from .action import ActionManager
from .battle_core import BattleCore
from .battle_state import BattleState
from .observation import ObservationFactory, Observation
from .pkmn_team_factory import PkmnTeamFactory
from .save_state import SaveStateManager

from pkmn_rl_arena.utils.logging import logger

from typing import Optional, List, Dict
from pathlib import Path
import functools

from pettingzoo import AECEnv
from rl_new_pokemon_ai import PATHS
from gymnasium.spaces import Discrete
import numpy.typing as npt

from rich.console import Console
from rich.table import Table


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

    def __init__(self, battle_core: BattleCore, max_steps_per_episode: int = 1000):
        # Initialize core components
        self.core = battle_core
        self.observation_factory = ObservationFactory(self.core)
        self.action_manager = ActionManager(self.core)
        # self.turn_manager = TurnManager(self.core, self.action_manager)
        self.battle_state = BattleState()
        # self.episode_manager = EpisodeManager()
        self.save_state_manager = SaveStateManager(self.core)
        self.team_factory = PkmnTeamFactory(PATHS["PKMN_CSV"], PATHS["MOVES_CSV"])

        # Environment configuration
        self.possible_agents = ["player", "enemy"]
        self.action_space_size = 10
        self.observation: Dict[str, npt.NDArray[int] | None] = {
            "agent": None,
            "ennemy": None,
        }

        self.terminations = {agent: False for agent in self.agents}
        self.rewards = {"player": 0.0, "enemy": 0.0}
        self.episode_steps = 0
        self.max_steps_per_episode = max_steps_per_episode  # Configurable

        # render console
        self.console = Console()

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, str] | None = None,
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

        # Load save state if provided
        if options is None:
            raise ValueError(
                'options argument is empty, cannot load save_state from options["save_state"], exiting'
            )
        elif options.get("save_state", None) is None:
            raise ValueError(
                f'"save_state" key was not found in options argument. cannot load save state.\nKeys in option : {options.keys()}.\n\nexiting'
            )

        save_state = Path(options["save_state"])
        if not self.save_state_manager.has_state(save_state):
            raise ValueError(f"Save state not found at path : {save_state} exiting.")

        loaded = self.save_state_manager.load_state(save_state)
        if not loaded:
            raise RuntimeError(f"Failed to load save state: {save_state}")

        # Reset managers
        self.rewards = {"player": 0.0, "enemy": 0.0}
        self._cumulative_rewards = {"player": 0.0, "enemy": 0.0}
        self.episode_steps = 0
        self.terminations = {agent: False for agent in self.agents}

        # reset battle
        self.battle_state = BattleState()

        # create new teams
        teams = {
            "player": self.team_factory.create_random_team(self.core),
            "ennemy": self.team_factory.create_random_team(self.core),
        }
        self.core.write_team_data(teams)

        # Advance to first turn to  get initial observations
        self.core.advance_to_next_turn()
        observations = self.observation_manager.get_observations()

        # clean rendering
        self.console.clear()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, int]):
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

        # Get dummy infos (not used in this example)
        # Validate actions
        for agent, action in actions.items():
            if not ActionManager.is_valid_action(action):
                raise ValueError(f"Invalid action {action} for agent {agent}")

        # Write actions
        self.action_manager.write_actions(self.battle_state.current_turn, actions)
        self.core.clear_stop_condition(self.battle_state.current_turn)
        self.battle_state.current_turn = self.core.advance_to_next_turn()

        # Check termination conditions
        # TODO : DEFINE TRUCATIONS & TERMINATIONS
        # TODO : EXTRACT BATTLECORE TESTS
        if (
            self.battle_state.is_battle_done()
            or self.max_steps_per_episode < self.episode_steps
        ):
            terminations = {agent: True for agent in self.agents}

        # Get observations
        self.observation = self.observation_manager.get_observations()

        # Calculate rewards (placeholder)
        self.rewards = {"player": 0.0, "enemy": 0.0}
        self._cumulative_rewards += self.rewards

        self.episode_steps += 1

        # Prepare info
        infos = {
            "current_turn": self.battle_state.get_current_turn(),
            "battle_done": self.battle_state.is_battle_done(),
            "ste_rewards": self.rewards,
            "episode_steps": self.episode_steps,
            "max_allowed_steps": self.max_steps_per_episode,
        }

        return observations, rewards, terminations, truncations, infos

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
