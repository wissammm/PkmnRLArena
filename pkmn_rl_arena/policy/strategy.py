from pkmn_rl_arena.env.observation import Observation, ObsIdx
from pkmn_rl_arena.policy.types import TYPES_ID, effectiveness
from pkmn_rl_arena.policy.battle_helpers import BattleHelpers, MoveEffects
from typing import List, Tuple, Optional
import numpy as np

class Strategies:
    """Various strategies for selecting actions based on observations."""

    @staticmethod
    def get_super_effective_move_indices(observation: Observation, attacker_agent: str) -> List[int]:
        """
        Returns indices (0-3) of moves that are super-effective (2x damage) against the opponent's active Pokemon.
        
        Args:
            observation: Current battle observation
            attacker_agent: Agent making the attack ("player" or "enemy")
            
        Returns:
            List of move indices (0-3) that are super-effective. Empty list if none.
        """
        opponent_agent = "enemy" if attacker_agent == "player" else "player"
        
        attacker_data = observation.get_agent_data(attacker_agent)
        opponent_data = observation.get_agent_data(opponent_agent)
        
        attacker_pokemon = np.split(attacker_data, 6)
        opponent_pokemon = np.split(opponent_data, 6)
        
        attacker_active = BattleHelpers.get_active_pokemon(attacker_pokemon)
        opponent_active = BattleHelpers.get_active_pokemon(opponent_pokemon)
        
        if attacker_active is None or opponent_active is None:
            return []
        
        opponent_type1, opponent_type2 = BattleHelpers.get_pokemon_types(opponent_active)
        
        super_effective_indices = []
        
        for i in range(ObsIdx.MAX_PKMN_MOVES):
            move_data = BattleHelpers.get_move_data(attacker_active, i)
            
            # Skip if move has no power or no PP
            if move_data["power"] == 0 or move_data["pp"] == 0 or move_data["type"] == "NONE":
                continue
            
            # Calculate effectiveness
            total_effectiveness = BattleHelpers.calculate_type_effectiveness(
                move_data["type"], opponent_type1, opponent_type2
            )
            
            # Check if super-effective (2x or 4x damage)
            if total_effectiveness >= 2.0:
                super_effective_indices.append(i)
        
        return super_effective_indices
    
    @staticmethod
    def get_priority_kill_moves(observation: Observation, attacker_agent: str) -> List[int]:
        """
        Returns indices (0-3) of priority moves that can KO the opponent's active Pokemon.
        
        Args:
            observation: Current battle observation
            attacker_agent: Agent making the attack ("player" or "enemy")
            
        Returns:
            List of move indices (0-3) that are priority moves capable of KOing opponent.
        """
        opponent_agent = "enemy" if attacker_agent == "player" else "player"
        
        attacker_data = observation.get_agent_data(attacker_agent)
        opponent_data = observation.get_agent_data(opponent_agent)
        
        attacker_pokemon = np.split(attacker_data, 6)
        opponent_pokemon = np.split(opponent_data, 6)
        
        attacker_active = BattleHelpers.get_active_pokemon(attacker_pokemon)
        opponent_active = BattleHelpers.get_active_pokemon(opponent_pokemon)
        
        if attacker_active is None or opponent_active is None:
            return []
        
        priority_kill_indices = []
        
        for i in range(ObsIdx.MAX_PKMN_MOVES):
            move_data = BattleHelpers.get_move_data(attacker_active, i)
            
            # Skip if move has no power, no PP, or no priority
            if move_data["power"] == 0 or move_data["pp"] == 0 or move_data["priority"] <= 0:
                continue
            
            # Check if move can KO
            if BattleHelpers.can_ko(attacker_active, opponent_active, i):
                priority_kill_indices.append(i)
        
        return priority_kill_indices
    
    @staticmethod
    def get_defensive_switch_indices(observation: Observation, attacker_agent: str) -> List[int]:
        """
        Returns indices (4-9) of Pokemon switches that have type advantage/resistance against opponent.
        
        Args:
            observation: Current battle observation
            attacker_agent: Agent considering the switch ("player" or "enemy")
            
        Returns:
            List of action indices (4-9) representing advantageous switches.
        """
        opponent_agent = "enemy" if attacker_agent == "player" else "player"
        
        attacker_data = observation.get_agent_data(attacker_agent)
        opponent_data = observation.get_agent_data(opponent_agent)
        
        attacker_pokemon = np.split(attacker_data, 6)
        opponent_pokemon = np.split(opponent_data, 6)
        
        opponent_active = BattleHelpers.get_active_pokemon(opponent_pokemon)
        
        if opponent_active is None:
            return []
        
        opponent_type1, opponent_type2 = BattleHelpers.get_pokemon_types(opponent_active)
        
        defensive_switches = []
        
        for pkmn_idx, pkmn in enumerate(attacker_pokemon):
            is_active = int(pkmn[ObsIdx.RAW_DATA["is_active"]])
            hp = int(pkmn[ObsIdx.RAW_DATA["HP"]])
            
            if is_active == 1 or hp == 0:
                continue
            
            pkmn_type1, pkmn_type2 = BattleHelpers.get_pokemon_types(pkmn)
            
            # Calculate resistance (how much damage opponent does to this Pokemon)
            total_resistance = 1.0
            
            if opponent_type1 != "NONE":
                total_resistance *= BattleHelpers.calculate_type_effectiveness(
                    opponent_type1, pkmn_type1, pkmn_type2
                )
            
            if opponent_type2 != "NONE":
                total_resistance *= BattleHelpers.calculate_type_effectiveness(
                    opponent_type2, pkmn_type1, pkmn_type2
                )
            
            # If this Pokemon resists opponent (takes 0.5x or less damage)
            if total_resistance <= 0.5:
                defensive_switches.append(4 + pkmn_idx)
        
        return defensive_switches
    
    @staticmethod
    def should_switch_to_preserve_pokemon(observation: Observation, attacker_agent: str) -> bool:
        """
        Determines if active Pokemon should switch out to preserve it.
        
        Args:
            observation: Current battle observation
            attacker_agent: Agent considering preservation switch
            
        Returns:
            True if should switch to preserve active Pokemon
        """
        opponent_agent = "enemy" if attacker_agent == "player" else "player"
        
        attacker_data = observation.get_agent_data(attacker_agent)
        opponent_data = observation.get_agent_data(opponent_agent)
        
        attacker_pokemon = np.split(attacker_data, 6)
        opponent_pokemon = np.split(opponent_data, 6)
        
        attacker_active = BattleHelpers.get_active_pokemon(attacker_pokemon)
        opponent_active = BattleHelpers.get_active_pokemon(opponent_pokemon)
        
        if attacker_active is None or opponent_active is None:
            return False
        
        # Condition 1: HP must be low
        hp_percentage = BattleHelpers.get_hp_percentage(attacker_active)
        if hp_percentage >= 30:
            return False
        
        # Condition 2: Check if opponent has type advantage
        attacker_type1, attacker_type2 = BattleHelpers.get_pokemon_types(attacker_active)
        opponent_type1, opponent_type2 = BattleHelpers.get_pokemon_types(opponent_active)
        
        type_disadvantage = 1.0
        
        if opponent_type1 != "NONE":
            type_disadvantage *= BattleHelpers.calculate_type_effectiveness(
                opponent_type1, attacker_type1, attacker_type2
            )
        
        if opponent_type2 != "NONE":
            type_disadvantage *= BattleHelpers.calculate_type_effectiveness(
                opponent_type2, attacker_type1, attacker_type2
            )
        
        if type_disadvantage < 2.0:
            return False
        
        # Condition 3: Check if we have better alternatives
        defensive_switches = Strategies.get_defensive_switch_indices(observation, attacker_agent)
        
        return len(defensive_switches) > 0
    
    @staticmethod
    def get_setup_move_indices(observation: Observation, attacker_agent: str) -> List[int]:
        """
        Returns indices (0-3) of stat-boosting moves when it's safe to set up.
        
        Args:
            observation: Current battle observation
            attacker_agent: Agent considering setup moves
            
        Returns:
            List of move indices (0-3) for stat-boosting moves when safe.
        """
        opponent_agent = "enemy" if attacker_agent == "player" else "player"
        
        attacker_data = observation.get_agent_data(attacker_agent)
        opponent_data = observation.get_agent_data(opponent_agent)
        
        attacker_pokemon = np.split(attacker_data, 6)
        opponent_pokemon = np.split(opponent_data, 6)
        
        attacker_active = BattleHelpers.get_active_pokemon(attacker_pokemon)
        opponent_active = BattleHelpers.get_active_pokemon(opponent_pokemon)
        
        if attacker_active is None or opponent_active is None:
            return []
        
        # Check if it's safe to set up
        our_hp_percent = BattleHelpers.get_hp_percentage(attacker_active)
        opponent_hp_percent = BattleHelpers.get_hp_percentage(opponent_active)
        
        # Not safe if we're low HP or opponent is healthy
        if our_hp_percent < 70 or opponent_hp_percent > 30:
            return []
        
        stat_boost_effects = MoveEffects.get_stat_boost_effects()
        
        setup_moves = []
        
        for i in range(ObsIdx.MAX_PKMN_MOVES):
            move_data = BattleHelpers.get_move_data(attacker_active, i)
            
            # Stat-boosting moves typically have 0 power
            if move_data["pp"] > 0 and move_data["power"] == 0 and move_data["effect"] in stat_boost_effects:
                setup_moves.append(i)
        
        return setup_moves
    
    @staticmethod
    def get_status_move_indices(observation: Observation, attacker_agent: str) -> List[int]:
        """
        Returns indices (0-3) of status-inflicting moves when advantageous.
        
        Args:
            observation: Current battle observation
            attacker_agent: Agent considering status moves
            
        Returns:
            List of move indices (0-3) for status-inflicting moves.
        """
        opponent_agent = "enemy" if attacker_agent == "player" else "player"
        
        attacker_data = observation.get_agent_data(attacker_agent)
        opponent_data = observation.get_agent_data(opponent_agent)
        
        attacker_pokemon = np.split(attacker_data, 6)
        opponent_pokemon = np.split(opponent_data, 6)
        
        attacker_active = BattleHelpers.get_active_pokemon(attacker_pokemon)
        opponent_active = BattleHelpers.get_active_pokemon(opponent_pokemon)
        
        if attacker_active is None or opponent_active is None:
            return []
        
        # Check opponent HP
        opponent_hp_percent = BattleHelpers.get_hp_percentage(opponent_active)
        
        # Don't use status moves on low HP opponents
        if opponent_hp_percent < 70:
            return []
        
        # Check if opponent already has status
        if BattleHelpers.has_status_condition(opponent_active):
            return []
        
        status_effect_codes = MoveEffects.get_status_effects()
        
        status_moves = []
        
        for i in range(ObsIdx.MAX_PKMN_MOVES):
            move_data = BattleHelpers.get_move_data(attacker_active, i)
            
            if move_data["pp"] > 0 and move_data["effect"] in status_effect_codes:
                status_moves.append(i)
        
        return status_moves