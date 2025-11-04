from pkmn_rl_arena.env.observation import Observation, ObsIdx
from pkmn_rl_arena.policy.types import TYPES_ID, effectiveness
from typing import List
import numpy as np

class Strategies:
    """Various strategies for selecting actions based on observations."""

    def get_super_effective_move_indices(observation: Observation, attacker_agent: str) -> List[int]:
        """
        Returns indices (0-3) of moves that are super-effective (2x damage) against the opponent's active Pokemon.
        
        Args:
            observation: Current battle observation
            attacker_agent: Agent making the attack ("player" or "enemy")
            
        Returns:
            List of move indices (0-3) that are super-effective. Empty list if none.
        """
        # Determine opponent agent
        opponent_agent = "enemy" if attacker_agent == "player" else "player"
        
        # Get attacker's data
        attacker_data = observation.get_agent_data(attacker_agent)
        opponent_data = observation.get_agent_data(opponent_agent)
        
        # Split into individual Pokemon data
        attacker_pokemon = np.split(attacker_data, 6)
        opponent_pokemon = np.split(opponent_data, 6)
        
        # Find active Pokemon for both sides
        attacker_active = None
        opponent_active = None
        
        for pkmn in attacker_pokemon:
            if pkmn[ObsIdx.RAW_DATA["is_active"]] == 1:
                attacker_active = pkmn
                break
        
        for pkmn in opponent_pokemon:
            if pkmn[ObsIdx.RAW_DATA["is_active"]] == 1:
                opponent_active = pkmn
                break
        
        if attacker_active is None or opponent_active is None:
            return []
        
        # Get opponent types
        opponent_type1_id = int(opponent_active[ObsIdx.RAW_DATA["type_1"]])
        opponent_type2_id = int(opponent_active[ObsIdx.RAW_DATA["type_2"]])
        
        opponent_type1 = TYPES_ID.get(opponent_type1_id, "NONE")
        opponent_type2 = TYPES_ID.get(opponent_type2_id, "NONE")
        
        # Check each move for super-effectiveness
        super_effective_indices = []
        
        for i in range(ObsIdx.MAX_PKMN_MOVES):
            move_start = ObsIdx.RAW_DATA["moves_begin"] + (i * ObsIdx.RAW_DATA["move_slot_stride"])
            
            # Get move type
            move_type_id = int(attacker_active[move_start + ObsIdx.RAW_DATA["type_offset"]])
            move_type = TYPES_ID.get(move_type_id, "NONE")
            
            # Get move power (skip status moves with 0 power)
            move_power = int(attacker_active[move_start + ObsIdx.RAW_DATA["power_offset"]])
            move_pp = int(attacker_active[move_start + ObsIdx.RAW_DATA["pp_offset"]])
            
            # Skip if move has no power or no PP
            if move_power == 0 or move_pp == 0 or move_type == "NONE":
                continue
            
            # Calculate effectiveness against opponent's type(s)
            eff1 = effectiveness(move_type, opponent_type1) if opponent_type1 != "NONE" else 1.0
            eff2 = effectiveness(move_type, opponent_type2) if opponent_type2 != "NONE" else 1.0
            
            total_effectiveness = eff1 * eff2
            
            # Check if super-effective (2x or 4x damage)
            if total_effectiveness >= 2.0:
                super_effective_indices.append(i)
        
        return super_effective_indices
    # Quick attack 
    # switch to super effective
    # Stats boost 
    # Apply status
    