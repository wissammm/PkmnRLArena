from pkmn_rl_arena.env.pkmn_team_factory import DataSize
from .action import ActionManager, ACTION_SPACE_SIZE
from .battle_core import BattleCore, TurnType, BattleState
from .observation import ObservationFactory, OBS_SPACE_SIZE
from .pkmn_team_factory import PkmnTeamFactory
from .save_state import SaveStateManager
from pkmn_rl_arena import (
    POKEMON_CSV_PATH,
    MOVES_CSV_PATH,
    ROM_PATH,
    BIOS_PATH,
    MAP_PATH,
)

from pkmn_rl_arena.logging import log

from enum import Enum
from typing import Any, Dict, Optional
import functools


import numpy as np
from numpy import typing as npt

from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import AgentSelector

from rich.console import Console
from rich.table import Table


class RenderMode(Enum):
    """Enumeration for different rendering mode"""

    DISABLED = 1  # No rendering, just raw dogging gpu
    EPISODE_TERMINAL = 2  # Basic rendering enabed, returns a simple table for each episode with final team status and who won
    TURN_TERMINAL = 3  # Basic rendering enabed, returns a simple table for each turn with team status
    RUSTBOY = 4  # show a picture of each turn rendered in rustboy advance


class BattleArena(ParallelEnv):
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

    ##########################################################################
    #
    # CONSTRUCTOR
    #
    ##########################################################################
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
        self.possible_agents = ["player", "enemy"]
        self.agents = self.possible_agents
        # self.reset_options = {"required": ["save_state", "teams"], "optional": []}
        self.action_space_size = 10
        self.observations = {
            agent: {"observation": np.array([], dtype=int), "action_mask": []}
            for agent in self.agents
        }
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.rewards = {agent: 0.0 for agent in self.agents}
        self.episode_steps = 0
        self.max_steps_per_episode = max_steps_per_episode  # Configurable
        self.infos = {}

        # render console
        self.console = Console()

        self.battle_state.current_turn = self.core.advance_to_next_turn()
        if not self.battle_state.current_turn == TurnType.CREATE_TEAM:
            raise RuntimeError(
                "Upon creating BattleCore and calling advance_to_next_turn(), turntype should be advance to next turn"
            )
        self.save_state_manager.save_state("boot_state")
        log.info(f"Created save_state : {self.save_state_manager.save_states}")

    ##########################################################################
    #
    # RESET & OPTIONS FCTN
    #
    ##########################################################################
    # def check_options_valid(self, options: Dict[str, Any]):
    #     for option in options.keys():
    #         if not (
    #             option in self.reset_options["required"]
    #             or option in self.reset_options["optional"]
    #         ):
    #             raise ValueError(
    #                 f"Invalid reset option found : {option}.\n Expected options : {self.reset_options['required']} and {self.reset_options['optional']}"
    #             )
    #     return

    def load_save_state(self, options: Dict[str, str]):
        """In charge of trying to load a save state.
        Args:
            options: Dict[str,str] : if dict

        """
        if options.get("save_state") is None:
            log.debug(
                'No save state name given in options["save_state"], creating a new battle core.'
            )
            self.core = BattleCore(ROM_PATH, BIOS_PATH, MAP_PATH)
            self.battle_state.current_turn = self.core.advance_to_next_turn()
            return

        loaded = self.save_state_manager.load_state(options.get("save_state"))
        if not loaded:
            raise RuntimeError("Failed to load save state.")
        self.battle_state.current_turn = self.core.get_turn_type()
        return

    def create_teams(self, options: Dict[str, Any]) -> Dict[str, list[int]]:
        if options.get("teams") is None:
            return {
                agent: self.team_factory.create_random_team() for agent in self.agents
            }

        teams = options["teams"]
        for agent, team in teams.items():
            log.info(f"Creating {agent} team.")
            if team is None:
                teams[agent] = self.team_factory.create_random_team()
                continue

            if len(team) % DataSize.PKMN != 0:
                raise ValueError(
                    f"Pkmn team creation : Incorrect param count."
                    f"\nA pkmn takes {DataSize.PKMN} params to be created, but received  {len(team)} params."
                    f"\nWhich accounts for : {int(len(team) / DataSize.PKMN)} pkmns and {len(team) / DataSize.PKMN} leftover params."
                )

            while len(team) / DataSize.PKMN < DataSize.PARTY_SIZE:
                team.extend([0, 0, 0, 0, 0, 0, 0, 0])  # Empty pkmn slot
            if not self.team_factory.is_team_valid(np.array(team)):
                raise ValueError('Invalid reset param : "team".')

            teams[agent] = team

        return teams

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = {"save_state": "boot_state", "teams": None},
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

        log.debug(f"Resetting env with options {options}")
        if options is None:
            raise ValueError(
                f"No options given, for env reset, required options : {self.reset_options}"
            )
        # self.check_options_valid(options)

        # Reset managers
        self.agents = self.possible_agents
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.episode_steps = 0
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        ##### reset battle
        # Load save state if provided otherwise creates a new battlecore
        self.load_save_state(options)
        # create new teams
        teams = self.create_teams(options)
        self.core.write_team_data(teams)

        # Advance to first turn to  get initial observations
        self.battle_state.current_turn = self.core.advance_to_next_turn()
        assert self.battle_state.current_turn == TurnType.GENERAL, (
            f"Expected turntype.GENERAL, got {self.battle_state.current_turn}"
        )
        observations = self.observation_factory.from_game().o
        observations = self.observation_factory.from_game()
        self.observations = {
            "player": {
                "observation": observations.o["player"],
                "action_mask": self.action_manager.get_action_mask("player"),
            },
            "enemy": {
                "observation": observations.o["enemy"],
                "action_mask": self.action_manager.get_action_mask("enemy"),
            },
        }

        # clean rendering
        self.console.clear()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        self.infos = {a: {} for a in self.agents}

        return self.observations, self.infos

    ##########################################################################
    #
    # STEP
    #
    ##########################################################################
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
        log.debug(f"{self.core.state}")
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
            if not ActionManager.is_valid_action(agent, action):
                raise ValueError(f"Invalid action {action} for agent {agent}")

        # Write actions
        self.action_manager.write_actions(self.battle_state.current_turn, actions)
        self.battle_state.current_turn = self.core.advance_to_next_turn()

        # Get observations
        observation = self.observation_factory.from_game()
        self.observations = {
            "player": {
                "observation": observation.o["player"],
                "action_mask": self.action_manager.get_action_mask("player"),
            },
            "enemy": {
                "observation": observation.o["enemy"],
                "action_mask": self.action_manager.get_action_mask("enemy"),
            },
        }

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

        self.infos = {}

        return (
            self.observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    ##########################################################################
    #
    # RENDER
    #
    ##########################################################################

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
        pass

    def _get_observations(self):
        obs = self.observation_factory.from_game()
        return {
            "player": {
                "observation": obs.o["player"],
                "action_mask": self.action_manager.get_action_mask("player"),
            },
            "enemy": {
                "observation": obs.o["enemy"],
                "action_mask": self.action_manager.get_action_mask("enemy"),
            },
        }

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/

        # (73 params / pkmn) * (6 pkmn / party) = 438
        return Discrete(OBS_SPACE_SIZE)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # We can seed the action space to make the environment deterministic.
        #
        # 4 moves + 5 pkmn switch = 9
        return Discrete(ACTION_SPACE_SIZE)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        self.save_state_manager.remove_save_states()
        return
