from pkmn_rl_arena.env.pkmn_team_factory import DataSize
from .action import ActionManager, ACTION_SPACE_SIZE
from .battle_core import BattleCore
from .battle_state import TurnType
from .observation import Observation, ObservationFactory, ObsIdx
from .pkmn_team_factory import PkmnTeamFactory
from .reward.manager import RewardManager
from .reward.functions import reward_functions
from .save_state import SaveStateManager
from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena.env.rendering import GameRendering

from pkmn_rl_arena.logging import log

from collections.abc import Callable
from enum import Enum
from typing import Any, Dict, Optional, Tuple
import functools

import numpy as np
from numpy import typing as npt

from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv


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
        reward_function: Callable[[str, list[Observation]], float] = reward_functions[
            0
        ],
        max_steps_per_episode: int = 1000,
        render_mode: RenderMode = RenderMode.DISABLED,
    ):
        # Initialize core components
        self.core = battle_core
        self.observation_factory = ObservationFactory(self.core)
        self.action_manager = ActionManager(self.core)
        self.team_factory = PkmnTeamFactory(PATHS["POKEMON_CSV"], PATHS["MOVES_CSV"])
        self.reward_manager = RewardManager(reward_function)
        self.save_state_manager = SaveStateManager(self.core)

        # Environment configuration
        self.possible_agents = ["player", "enemy"]
        self.agents = self.possible_agents
        self.action_space_size = ACTION_SPACE_SIZE
        self.observations = {
            agent: {"observation": np.array([], dtype=int), "action_mask": []}
            for agent in self.agents
        }
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.rewards = {agent: 0.0 for agent in self.agents}
        self.max_steps_per_episode = max_steps_per_episode  # Configurable
        self.infos = {}

        # render console
        self.game_renderer = GameRendering(self.team_factory, self.agents)

        if self.core.state.turn != TurnType.CREATE_TEAM:
            raise RuntimeError(
                f"Env creation : Upon creating BattleCore and calling advance_to_next_turn(), turntype should be {TurnType.CREATE_TEAM}. Got {self.core.state.turn}."
            )
        log.debug(f"CURRENT STATE = {self.core.state}")
        self.save_state_manager.save_state("boot_state")
        log.info(f"Created save_state : {self.save_state_manager.save_states}")

    ##########################################################################
    #
    # RESET & OPTIONS FCTN
    #
    ##########################################################################
    def load_save_state(self, options: Dict[str, str]):
        """In charge of trying to load a save state.
        Args:
            options: Dict[str,str] : if dict

        """
        if options.get("save_state") is None:
            log.debug(
                'No save state name given in options["save_state"], creating a new battle core.'
            )
            self.core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
            self.action_manager.battle_core = self.core
            self.observation_factory.battle_core = self.core
            self.team_factory.battle_core = self.core
            self.save_state_manager.battle_core = self.core
        else:
            returned_state = self.save_state_manager.load_state(options["save_state"])
            self.core = self.save_state_manager.core
            self.observation_factory.core = self.save_state_manager.core
            self.action_manager.core = self.save_state_manager.core
            self.team_factory.core = self.save_state_manager.core
            self.reward_manager.core = self.save_state_manager.core
            self.save_state_manager.core = self.save_state_manager.core
            if returned_state is None:
                raise RuntimeError(
                    f"Failed to load save state {options.get('save_state')}"
                )
            if options["save_state"] == "boot_state":
                assert self.core.state.step == 0, (
                    f'Loaded "boot_state", expected state\'s step number to be 0, got following state: : {self.core.state}.'
                )
        assert self.core.state.turn == TurnType.CREATE_TEAM, (
            f"Its required to reset with a state whose turntype value is {TurnType.CREATE_TEAM}, got {self.core.state.turn}."
        )
        return

    def create_teams(self, options: Dict[str, Any]) -> Dict[str, list[int]]:
        if options.get("teams") is None:
            log.debug('No team provided in options["teams"], creating random teams.')
            return {
                agent: self.team_factory.create_random_team() for agent in self.agents
            }

        teams = options["teams"]
        for agent, team in teams.items():
            log.info(f"Creating {agent} team.")
            if team is None:
                log.info(f"No team provided for {agent}, creating a random one.")
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
    ) -> Tuple[
        Dict[str, Dict[str, npt.NDArray[int]]],  # observations
        Dict[str, Any],  # infos
    ]:
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

        # Reset managers
        self.agents = self.possible_agents
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}

        ##### reset battle
        # Load save state if provided otherwise creates a new battlecore
        self.load_save_state(options)

        # create new teams
        teams = self.create_teams(options)
        self.core.write_team_data(teams)
        self.core.advance_to_next_turn(count_step=False)

        observations = self.observation_factory.from_game()
        self.observations = {
            "player": {
                "observation": observations.get_normalized_agent_data("player"),
                "action_mask": self.action_manager.get_action_mask("player"),
            },
            "enemy": {
                "observation": observations.get_normalized_agent_data("enemy"),
                "action_mask": self.action_manager.get_action_mask("enemy"),
            },
        }

        self.reward_manager.reset()
        self.reward_manager.add_observation(observations)

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        self.infos = {a: {} for a in self.agents}
        self.reward = {a: 0 for a in self.agents}

        # create new rendering
        self.game_renderer.console.clear()
        self.render(observations, self.reward)

        return self.observations, self.infos

    def _was_dead_step(self, actions):
        # No agents left after episode ends
        # We can improve this method by keeping the last valid observation, reward, info
        self.agents = []
        return {}, {}, {}, {}, {}

    ##########################################################################
    #
    # STEP
    #
    ##########################################################################
    def step(
        self, actions: Optional[Dict[str, int]]
    ) -> Tuple[
        Dict[str, Dict[str, npt.NDArray[int]]],  # observations
        Dict[str, float],  # rewards
        Dict[str, bool],  # terminations
        Dict[str, bool],  # truncations
        Dict[str, Any],  # infos
    ]:
        """
        step(action) takes in an action for all agents and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
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
        for agent in self.agents:
            if self.truncations[agent] or self.terminations[agent]:
                return self._was_dead_step(actions)

        # Write actions
        self.action_manager.write_actions(actions)
        self.core.advance_to_next_turn()

        # Get observations
        observations = self.observation_factory.from_game()
        self.observations = {
            "player": {
                "observation": observations.get_normalized_agent_data("player"),
                "action_mask": self.action_manager.get_action_mask("player"),
            },
            "enemy": {
                "observation": observations.get_normalized_agent_data("enemy"),
                "action_mask": self.action_manager.get_action_mask("enemy"),
            },
        }

        self.reward_manager.add_observation(observations)

        for agent, obs in self.observations.items():
            self.rewards[agent] = self.reward_manager.compute_reward(agent)
            self._cumulative_rewards[agent] += self.rewards[agent]

        if self.core.is_episode_done():
            self.terminations = {agent: True for agent in self.agents}
        elif self.max_steps_per_episode < self.core.state.step:
            self.truncations = {agent: True for agent in self.agents}

        self.render(observations, self.rewards)

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

    def render(self, observation: Observation, reward: Dict[str, float]):
        """
        Render the current state of the battle using the rich library.
        """
        self.game_renderer.refresh(observation, reward)

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

        return Discrete(ObsIdx.OBS_SIZE)

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
        # self.save_state_manager.remove_save_states()
        return
