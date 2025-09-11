from pkmn_rl_arena import ROM_PATH, BIOS_PATH, MAP_PATH, POKEMON_CSV_PATH, SAVE_PATH
from .action import ActionManager
from .battle_core import BattleCore, TurnType
from .episode import EpisodeManager
from .observation import ObservationFactory
from .save_state import SaveStateManager
from .turn_manager import TurnManager

import random
import sys
import os
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from rich.console import Console
from rich.table import Table
import shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

random.seed(124)


def clear_save_path():
    """Delete all files and folders inside SAVE_PATH."""
    if os.path.exists(SAVE_PATH):
        for filename in os.listdir(SAVE_PATH):
            file_path = os.path.join(SAVE_PATH, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")


class PokemonRLCore:
    """
    Main class that coordinates all components.
    This is the primary interface for the RL environment.
    """

    def __init__(
        self, rom_path: str, bios_path: str, map_path: str, max_steps: int = 200000
    ):
        # Initialize core components
        self.battle_core = BattleCore(rom_path, bios_path, map_path, max_steps)
        self.observation_factory = ObservationFactory(self.battle_core)
        self.action_manager = ActionManager(self.battle_core)
        self.turn_manager = TurnManager(self.battle_core, self.action_manager)
        self.episode_manager = EpisodeManager()
        self.save_state_manager = SaveStateManager(self.battle_core)

        # Environment configuration
        self.agents = ["player", "enemy"]
        self.action_space_size = 10

    def reset(
        self, save_state: Optional[str] = "state_before_create_team"
    ) -> Dict[str, pd.DataFrame]:
        """Reset the environment"""
        # Load save state if provided
        if save_state is not None and self.save_state_manager.has_state(save_state):
            loaded = self.save_state_manager.load_state(save_state)
            if not loaded:
                raise RuntimeError(f"Failed to load save state: {save_state}")
        else:
            self.episode_manager.reset_episode()
            self.turn_manager.state = BattleState()
            turn = self.turn_manager.advance_to_next_turn()
            if turn != TurnType.CREATE_TEAM:
                raise RuntimeError("Expected to start with CREATE_TEAM turn")

            self.save_state_manager.save_state(save_state)

        # Reset managers
        self.episode_manager.reset_episode()
        self.turn_manager.state = BattleState()

        # Reset team
        turn = self.turn_manager.advance_to_next_turn()

        if turn == TurnType.CREATE_TEAM:
            player_team = self._create_random_team(POKEMON_CSV_PATH)
            enemy_team = self._create_random_team(POKEMON_CSV_PATH)

            self.battle_core.write_team_data("player", player_team)
            self.battle_core.write_team_data("enemy", enemy_team)
            self.battle_core.clear_stop_condition(turn)

        else:
            raise RuntimeError("Expected to start with CREATE_TEAM turn")

        # Advance to first turn
        self.turn_manager.advance_to_next_turn()

        # Get initial observations
        return self.observation_factory.from_game()

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float], bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            actions: Dictionary of actions for each agent

        Returns:
            observations: New observations for each agent
            rewards: Rewards for each agent (placeholder)
            done: Whether the episode is finished
            info: Additional information
        """
        # Validate actions
        for agent, action in actions.items():
            if not self.action_manager.is_valid_action(action):
                raise ValueError(f"Invalid action {action} for agent {agent}")

        # Process turn
        turn_completed = self.turn_manager.process_turn(actions)

        if turn_completed:
            # Advance to next turn
            self.turn_manager.advance_to_next_turn()

        # Get new observations
        observations = self.observation_factory.from_game()

        # Calculate rewards (placeholder)
        rewards = {"player": 0.0, "enemy": 0.0}

        # Check if episode is done
        battle_done = self.turn_manager.is_battle_done()
        episode_done = self.episode_manager.is_episode_done(battle_done)

        # Update episode
        self.episode_manager.update_episode(rewards)

        # Prepare info
        info = {
            "current_turn": self.turn_manager.get_current_turn(),
            "battle_done": battle_done,
            "episode_info": self.episode_manager.get_episode_info(),
        }

        return observations, rewards, episode_done, info

    def get_current_turn_type(self) -> TurnType:
        """Get current turn type"""
        return self.turn_manager.get_current_turn()

    def get_required_agents(self) -> List[str]:
        """Get agents required for current turn"""
        return self.turn_manager._get_required_agents()

    def is_waiting_for_action(self) -> bool:
        """Check if environment is waiting for actions"""
        return self.turn_manager.state.waiting_for_action

    def render(self, observations: Dict[str, pd.DataFrame], csv_path: str):
        """
        Render the current state of the battle using the rich library.

        Args:
            observations: Dictionary containing observation DataFrames for 'player' and 'enemy'.
            csv_path: Path to the CSV file containing Pokémon data.
        """
        # Load Pokémon data from the CSV file
        pokemon_data = pd.read_csv(csv_path)

        # Initialize the rich console
        console = Console()

        # Create a table with two columns: Player and Enemy
        table = Table(
            title="Battle State", show_header=True, header_style="bold magenta"
        )
        table.add_column("Player", justify="center", style="cyan", no_wrap=True)
        table.add_column("Enemy", justify="center", style="red", no_wrap=True)

        # Helper function to get Pokémon name by ID
        def get_pokemon_name(pokemon_id):
            row = pokemon_data[pokemon_data["id"] == pokemon_id]
            return row["speciesName"].values[0] if not row.empty else "Unknown"

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
        console.print(table)


# Example usage
if __name__ == "__main__":
    # Configuration
    clear_save_path()
    # Initialize core
    rl_core = PokemonRLCore(ROM_PATH, BIOS_PATH, MAP_PATH)

    # Reset environment
    observations = rl_core.reset()

    # Example game loop
    for step in range(300):
        if step == 10:
            rl_core.reset()
            print("---------------------------")
            print()
            print("New save state")
            print("---------------------------")

        # Get current turn info
        current_turn = rl_core.get_current_turn_type()
        required_agents = rl_core.get_required_agents()

        print(
            f"Step {step}: Turn type = {current_turn}, Required agents = {required_agents}"
        )

        # Generate random actions for required agents
        actions = {}
        for agent in required_agents:
            legal_actions = rl_core.action_manager.get_action_mask(agent)
            if legal_actions:
                actions[agent] = random.choice(
                    legal_actions
                )  # Choose a random legal action
            else:
                print(f"No legal actions available for {agent}")

        # Execute step
        observations, rewards, done, info = rl_core.step(actions)

        rl_core.battle_core.save_savestate(f"step_{step}")

        rl_core.render(observations, POKEMON_CSV_PATH)

        print(f"Rewards: {rewards}, Done: {done}")

        if done:
            print("Episode finished!")
            break
