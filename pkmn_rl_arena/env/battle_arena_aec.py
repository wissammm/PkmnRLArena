from pkmn_rl_arena.env.pkmn_team_factory import DataSize
from .action import ActionManager, ACTION_SPACE_SIZE
from .battle_core import BattleCore, CoreContext
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

from gymnasium.spaces import Discrete, Box, Dict as GymDict

from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector


class RenderMode(Enum):
    """Enumeration for different rendering mode"""

    DISABLED = 1
    EPISODE_TERMINAL = 2
    TURN_TERMINAL = 3
    RUSTBOY = 4


class BattleArenaAEC(AECEnv):
    """
    Pokemon battle environment for MARL using AEC (Agent Environment Cycle).
    Agents act sequentially based on the game's turn requirements.
    """

    metadata = {
        "name": "pkmn_daycare_aec_v0.1",
        "render_modes": ["disabled", "episode_terminal", "turn_terminal", "rustboy"],
    }

    def __init__(
        self,
        battle_core: BattleCore,
        reward_function: Callable[[str, list[Observation]], float] = reward_functions[0],
        max_steps_per_episode: int = 1000,
        render_mode: RenderMode = RenderMode.DISABLED,
    ):
        super().__init__()
        
        # Initialize core components
        self.ctxt = CoreContext(battle_core)
        self.observation_factory = ObservationFactory(self.ctxt)
        self.action_manager = ActionManager(self.ctxt)
        self.team_factory = PkmnTeamFactory(PATHS["POKEMON_CSV"], PATHS["MOVES_CSV"])
        self.reward_manager = RewardManager(reward_function)
        self.save_state_manager = SaveStateManager(self.ctxt)

        # Environment configuration
        self.possible_agents = ["player", "enemy"]
        self.agents = []
        self.action_space_size = ACTION_SPACE_SIZE
        
        # AEC-specific
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = None
        self._action_buffer = {}
        
        self.observations = {
            agent: {
                "categorical": np.zeros(ObsIdx.CATEGORICAL_SIZE, dtype=np.int64),
                "continuous": np.zeros(ObsIdx.CONTINUOUS_SIZE, dtype=np.float32),
                "action_mask": np.zeros(ACTION_SPACE_SIZE, dtype=np.float32),
            }
            for agent in self.possible_agents
        }
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        
        self.max_steps_per_episode = max_steps_per_episode
        self.infos = {agent: {} for agent in self.possible_agents}

        # Rendering
        self.game_renderer = GameRendering(self.team_factory, self.possible_agents)
        self.render_mode = render_mode

        if self.ctxt.core.state.turn != TurnType.CREATE_TEAM:
            raise RuntimeError(
                f"Env creation: Upon creating BattleCore, turntype should be {TurnType.CREATE_TEAM}. "
                f"Got {self.ctxt.core.state.turn}."
            )
        log.debug(f"CURRENT STATE = {self.ctxt.core.state}")
        self.save_state_manager.save_state("boot_state")
        log.info(f"Created save_state: {self.save_state_manager.save_states}")

    ##########################################################################
    # RESET
    ##########################################################################
    def load_save_state(self, options: Dict[str, str]):
        if options.get("save_state") is None:
            log.debug("No save state given, creating new battle core.")
            self.ctxt.set_core(BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"]))
        else:
            returned_state = self.save_state_manager.load_state(options["save_state"])
            if returned_state is None:
                raise RuntimeError(f"Failed to load save state {options.get('save_state')}")
            if options["save_state"] == "boot_state":
                assert self.ctxt.core.state.step == 0, (
                    f'Loaded "boot_state", expected step=0, got {self.ctxt.core.state}.'
                )
        assert self.ctxt.core.state.turn == TurnType.CREATE_TEAM, (
            f"Reset requires turntype {TurnType.CREATE_TEAM}, got {self.ctxt.core.state.turn}."
        )

    def create_teams(self, options: Dict[str, Any], team_size: int = 6) -> Dict[str, list[int]]:
        if options.get("teams") is None:
            log.debug("No team provided, creating random teams.")
            return {
                agent: self.team_factory.create_random_team(size_of_team=team_size)
                for agent in self.possible_agents
            }

        teams = options["teams"]
        for agent, team in teams.items():
            log.info(f"Creating {agent} team.")
            if team is None:
                log.info(f"No team for {agent}, creating random one.")
                teams[agent] = self.team_factory.create_random_team(size_of_team=team_size)
                continue

            if len(team) % DataSize.PKMN != 0:
                raise ValueError(
                    f"Pkmn team creation: Incorrect param count for {agent}'s team. "
                    f"Expected multiple of {DataSize.PKMN}, got {len(team)}."
                )

            while len(team) / DataSize.PKMN < DataSize.PARTY_SIZE:
                team.extend([0] * DataSize.PKMN)
            if not self.team_factory.is_team_valid(np.array(team)):
                raise ValueError(f'Invalid team for {agent}.')

            teams[agent] = team

        return teams

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ):
        """Reset environment and return initial observations."""
        if options is None:
            options = {"save_state": "boot_state", "teams": None, "team_size": 6}
        
        log.debug(f"Resetting env with options {options}")

        team_size = options.get("team_size", 6)
        if not 1 <= team_size <= 6:
            raise ValueError(f"Team size must be 1-6, got {team_size}")

        # Reset state
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {"team_size": team_size} for agent in self.possible_agents}
        self._action_buffer = {}

        # Load state & create teams
        self.load_save_state(options)
        teams = self.create_teams(options, team_size)
        self.ctxt.core.write_team_data(teams)
        self.ctxt.core.advance_to_next_turn(count_step=False)

        # Initialize observations
        observations = self.observation_factory.from_game()
        self.observations = self._get_observations()
        raw_obs_obj = self.observation_factory.from_game()

        self.reward_manager.reset()
        self.reward_manager.add_observation(observations)

        # Set agent selector for first turn
        required_agents = self._get_required_agents()
        if required_agents:
            self._agent_selector = agent_selector(required_agents)
            self.agent_selection = self._agent_selector.reset()
        else:
            self.agents = []
            self.agent_selection = None

        # Rendering
        if self.render_mode != RenderMode.DISABLED:
            self.game_renderer.stop()
            self.game_renderer.start(raw_obs_obj, self.rewards, self.ctxt.core.state)
        
        return self.observations, self.infos

    ##########################################################################
    # STEP
    ##########################################################################
    def step(self, action: int):
        """Execute action for current agent."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._action_buffer[agent] = action
        
        required_agents = self._get_required_agents()
        all_actions_ready = all(ag in self._action_buffer for ag in required_agents)
        
        if all_actions_ready:
            self._execute_turn()
            self._action_buffer = {}
            
            # Check termination
            if self.ctxt.core.is_episode_done():
                self.terminations = {agent: True for agent in self.possible_agents}
                self.agents = []
                self.agent_selection = None
                return
            elif self.max_steps_per_episode < self.ctxt.core.state.step:
                self.truncations = {agent: True for agent in self.possible_agents}
                self.agents = []
                self.agent_selection = None
                return
            
            # Update agent selector for new turn
            required_agents = self._get_required_agents()
            if required_agents:
                self._agent_selector = agent_selector(required_agents)
                self.agent_selection = self._agent_selector.reset()
            else:
                self.agents = []
                self.agent_selection = None
        else:
            # Cycle to next required agent
            if self.agents:
                self.agent_selection = self._agent_selector.next()

    def _execute_turn(self):
        """Execute a full turn with buffered actions."""
        self.action_manager.write_actions(self._action_buffer)
        self.ctxt.core.advance_to_next_turn()

        # Get new observations
        observations = self.observation_factory.from_game()
        self.observations = self._get_observations()
        raw_obs_obj = self.observation_factory.from_game()
        self.reward_manager.add_observation(observations)

        # Reset and compute rewards
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        for agent in self.possible_agents:
            reward = self.reward_manager.compute_reward(agent)
            self.rewards[agent] = reward
            self._cumulative_rewards[agent] += reward

        # Update infos
        for agent in self.possible_agents:
            self.infos[agent] = {
                "turn": self.ctxt.core.state.turn.name,
                "step": self.ctxt.core.state.step,
                "cumulative_reward": self._cumulative_rewards[agent],
            }

        self.render(raw_obs_obj, self.rewards)

    def _was_dead_step(self, action):
        """Handle step for terminated/truncated agent."""
        if all(self.terminations.values()) or all(self.truncations.values()):
            self.agents = []
            self.agent_selection = None
            return
        
        self.rewards[self.agent_selection] = 0.0
        
        active_agents = [
            ag for ag in self.agents 
            if not (self.terminations[ag] or self.truncations[ag])
        ]
        
        if active_agents:
            self._agent_selector = agent_selector(active_agents)
            self.agent_selection = self._agent_selector.reset()
        else:
            self.agents = []
            self.agent_selection = None

    def _get_required_agents(self):
        """Get agents required for current turn."""
        turn_type = self.ctxt.core.state.turn
        if turn_type == TurnType.GENERAL:
            return ["player", "enemy"]
        elif turn_type == TurnType.PLAYER:
            return ["player"]
        elif turn_type == TurnType.ENEMY:
            return ["enemy"]
        else:
            return []

    ##########################################################################
    # OBSERVATION & ACTION SPACE
    ##########################################################################
    def observe(self, agent: str):
        """Return observation for specified agent."""
        return self.observations.get(
            agent,
            {
                "categorical": np.zeros(ObsIdx.CATEGORICAL_SIZE, dtype=np.int64),
                "continuous": np.zeros(ObsIdx.CONTINUOUS_SIZE, dtype=np.float32),
                "action_mask": np.zeros(ACTION_SPACE_SIZE, dtype=np.float32),
            },
        )

    def _get_observations(self):
        obs = self.observation_factory.from_game()

        formatted_obs = {}
        for agent in self.possible_agents:
            embed_data = obs.get_embedding_data(agent)

            formatted_obs[agent] = {
                "categorical": embed_data["categorical"].astype(np.int64),
                "continuous": embed_data["continuous"].astype(np.float32),
                "action_mask": self.action_manager.get_action_mask(agent).astype(np.float32),
            }
        return formatted_obs

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return GymDict(
            {
                "categorical": Box(
                    low=0, high=ObsIdx.MAX_FLAGS,
                    shape=(ObsIdx.CATEGORICAL_SIZE,), dtype=np.int64,
                ),
                "continuous": Box(
                    low=0, high=1,
                    shape=(ObsIdx.CONTINUOUS_SIZE,), dtype=np.float32
                ),
                "action_mask": Box(
                    low=0, high=1,
                    shape=(ACTION_SPACE_SIZE,), dtype=np.float32
                ),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(ACTION_SPACE_SIZE)

    ##########################################################################
    # RENDER
    ##########################################################################
    def render(self, observation: Observation, reward: Dict[str, float]):
        """Render current state."""
        if self.render_mode == RenderMode.DISABLED:
            return
        self.game_renderer.refresh(observation, reward, self.ctxt.core.state)

    def close(self):
        """Clean up resources."""
        pass