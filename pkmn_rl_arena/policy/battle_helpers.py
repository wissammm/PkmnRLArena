from pkmn_rl_arena.env.observation import Observation, ObsIdx
from pkmn_rl_arena.policy.types import TYPES_ID, effectiveness
from typing import List, Tuple, Optional
import numpy as np

class MoveEffects:
    """Move effect constants from battle_move_effects.h"""
    
    # Status effect moves (primary status conditions)
    EFFECT_SLEEP = 1
    EFFECT_POISON = 66
    EFFECT_PARALYZE = 67
    EFFECT_TOXIC = 33
    EFFECT_WILL_O_WISP = 167  # Burn
    
    # Status effect moves with damage
    EFFECT_POISON_HIT = 2
    EFFECT_BURN_HIT = 4
    EFFECT_FREEZE_HIT = 5
    EFFECT_PARALYZE_HIT = 6
    
    # Stat boost moves (+1 stage)
    EFFECT_ATTACK_UP = 10
    EFFECT_DEFENSE_UP = 11
    EFFECT_SPEED_UP = 12
    EFFECT_SPECIAL_ATTACK_UP = 13
    EFFECT_SPECIAL_DEFENSE_UP = 14
    EFFECT_ACCURACY_UP = 15
    EFFECT_EVASION_UP = 16
    
    # Stat boost moves (+2 stages)
    EFFECT_ATTACK_UP_2 = 50  # Swords Dance
    EFFECT_DEFENSE_UP_2 = 51
    EFFECT_SPEED_UP_2 = 52
    EFFECT_SPECIAL_ATTACK_UP_2 = 53  # Nasty Plot
    EFFECT_SPECIAL_DEFENSE_UP_2 = 54
    EFFECT_ACCURACY_UP_2 = 55
    EFFECT_EVASION_UP_2 = 56
    
    # Special setup moves
    EFFECT_CALM_MIND = 211  # +1 SpAtk, +1 SpDef
    EFFECT_DRAGON_DANCE = 212  # +1 Atk, +1 Speed
    EFFECT_BULK_UP = 208  # +1 Atk, +1 Def
    EFFECT_COSMIC_POWER = 206  # +1 Def, +1 SpDef
    EFFECT_BELLY_DRUM = 142  # Max Attack, 50% HP cost
    
    # Stat drop moves
    EFFECT_ATTACK_DOWN = 18
    EFFECT_DEFENSE_DOWN = 19
    EFFECT_SPEED_DOWN = 20
    EFFECT_SPECIAL_ATTACK_DOWN = 21
    EFFECT_SPECIAL_DEFENSE_DOWN = 22
    EFFECT_ACCURACY_DOWN = 23
    EFFECT_EVASION_DOWN = 24
    
    @classmethod
    def get_status_effects(cls) -> List[int]:
        """Returns list of effect codes that inflict status conditions"""
        return [
            cls.EFFECT_SLEEP,
            cls.EFFECT_POISON,
            cls.EFFECT_PARALYZE,
            cls.EFFECT_TOXIC,
            cls.EFFECT_WILL_O_WISP,
            cls.EFFECT_POISON_HIT,
            cls.EFFECT_BURN_HIT,
            cls.EFFECT_FREEZE_HIT,
            cls.EFFECT_PARALYZE_HIT,
        ]
    
    @classmethod
    def get_stat_boost_effects(cls) -> List[int]:
        """Returns list of effect codes that boost user's stats"""
        return [
            # +1 stage boosts
            cls.EFFECT_ATTACK_UP,
            cls.EFFECT_DEFENSE_UP,
            cls.EFFECT_SPEED_UP,
            cls.EFFECT_SPECIAL_ATTACK_UP,
            cls.EFFECT_SPECIAL_DEFENSE_UP,
            cls.EFFECT_ACCURACY_UP,
            cls.EFFECT_EVASION_UP,
            # +2 stage boosts
            cls.EFFECT_ATTACK_UP_2,
            cls.EFFECT_DEFENSE_UP_2,
            cls.EFFECT_SPEED_UP_2,
            cls.EFFECT_SPECIAL_ATTACK_UP_2,
            cls.EFFECT_SPECIAL_DEFENSE_UP_2,
            cls.EFFECT_ACCURACY_UP_2,
            cls.EFFECT_EVASION_UP_2,
            # Multi-stat boosts
            cls.EFFECT_CALM_MIND,
            cls.EFFECT_DRAGON_DANCE,
            cls.EFFECT_BULK_UP,
            cls.EFFECT_COSMIC_POWER,
            cls.EFFECT_BELLY_DRUM,
        ]


class BattleHelpers:
    """Helper functions for battle calculations and state queries"""
    
    @staticmethod
    def get_active_pokemon(pokemon_list: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Finds and returns the active Pokemon from a list of Pokemon data.
        
        Args:
            pokemon_list: List of 6 Pokemon arrays
            
        Returns:
            Active Pokemon array or None if not found
        """
        for pkmn in pokemon_list:
            if pkmn[ObsIdx.RAW_DATA["is_active"]] == 1:
                return pkmn
        return None
    
    @staticmethod
    def get_pokemon_types(pokemon: np.ndarray) -> Tuple[str, str]:
        """
        Extracts both types from a Pokemon.
        
        Args:
            pokemon: Pokemon data array
            
        Returns:
            Tuple of (type1, type2) as strings (e.g., ("FIRE", "FLY"))
        """
        type1_id = int(pokemon[ObsIdx.RAW_DATA["type_1"]])
        type2_id = int(pokemon[ObsIdx.RAW_DATA["type_2"]])
        
        type1 = TYPES_ID.get(type1_id, "NONE")
        type2 = TYPES_ID.get(type2_id, "NONE")
        
        return type1, type2
    
    @staticmethod
    def get_hp_percentage(pokemon: np.ndarray) -> float:
        """
        Calculates HP percentage of a Pokemon.
        
        Args:
            pokemon: Pokemon data array
            
        Returns:
            HP percentage (0-100)
        """
        current_hp = int(pokemon[ObsIdx.RAW_DATA["HP"]])
        max_hp = int(pokemon[ObsIdx.RAW_DATA["max_HP"]])
        
        if max_hp == 0:
            return 0.0
        
        return (current_hp / max_hp) * 100
    
    @staticmethod
    def calculate_type_effectiveness(move_type: str, defender_type1: str, defender_type2: str) -> float:
        """
        Calculates total type effectiveness multiplier.
        
        Args:
            move_type: Type of the attacking move
            defender_type1: First type of defender
            defender_type2: Second type of defender
            
        Returns:
            Total effectiveness multiplier (0.25x, 0.5x, 1x, 2x, or 4x)
        """
        eff1 = effectiveness(move_type, defender_type1) if defender_type1 != "NONE" else 1.0
        eff2 = effectiveness(move_type, defender_type2) if defender_type2 != "NONE" else 1.0
        
        return eff1 * eff2
    
    @staticmethod
    def has_stab(move_type: str, pokemon_type1: str, pokemon_type2: str) -> bool:
        """
        Checks if move gets STAB (Same Type Attack Bonus).
        
        Args:
            move_type: Type of the move
            pokemon_type1: First type of Pokemon
            pokemon_type2: Second type of Pokemon
            
        Returns:
            True if move type matches Pokemon type (1.5x damage)
        """
        return move_type in [pokemon_type1, pokemon_type2]
    
    @staticmethod
    def get_move_data(pokemon: np.ndarray, move_index: int) -> dict:
        """
        Extracts all data for a specific move.
        
        Args:
            pokemon: Pokemon data array
            move_index: Move slot (0-3)
            
        Returns:
            Dictionary with move data (power, pp, type, priority, effect, etc.)
        """
        move_start = ObsIdx.RAW_DATA["moves_begin"] + (move_index * ObsIdx.RAW_DATA["move_slot_stride"])
        
        move_type_id = int(pokemon[move_start + ObsIdx.RAW_DATA["type_offset"]])
        
        return {
            "power": int(pokemon[move_start + ObsIdx.RAW_DATA["power_offset"]]),
            "pp": int(pokemon[move_start + ObsIdx.RAW_DATA["pp_offset"]]),
            "type": TYPES_ID.get(move_type_id, "NONE"),
            "type_id": move_type_id,
            "priority": int(pokemon[move_start + ObsIdx.RAW_DATA["priority_offset"]]),
            "effect": int(pokemon[move_start + ObsIdx.RAW_DATA["effect_offset"]]),
            "accuracy": int(pokemon[move_start + ObsIdx.RAW_DATA["accuracy_offset"]]),
        }
    
    @staticmethod
    def calculate_damage(
        attacker: np.ndarray,
        defender: np.ndarray,
        move_index: int,
        is_critical: bool = False
    ) -> Tuple[int, int]:
        """
        Calculates damage range for a move using Gen 3 damage formula.
        
        Args:
            attacker: Attacker Pokemon data
            defender: Defender Pokemon data
            move_index: Move slot to use (0-3)
            is_critical: Whether to calculate critical hit damage
            
        Returns:
            Tuple of (min_damage, max_damage)
        """
        move_data = BattleHelpers.get_move_data(attacker, move_index)
        
        if move_data["power"] == 0 or move_data["pp"] == 0:
            return (0, 0)
        
        # Get stats
        attacker_level = int(attacker[ObsIdx.RAW_DATA["level"]])
        attacker_attack = int(attacker[ObsIdx.RAW_DATA["stats_begin"]])  # Physical ATK
        attacker_sp_attack = int(attacker[ObsIdx.RAW_DATA["stats_begin"] + 3])  # Special ATK
        
        defender_defense = int(defender[ObsIdx.RAW_DATA["stats_begin"] + 1])  # Physical DEF
        defender_sp_defense = int(defender[ObsIdx.RAW_DATA["stats_begin"] + 4])  # Special DEF
        
        # Simplified: assume physical for most moves (can be enhanced with split detection)
        attack_stat = attacker_attack
        defense_stat = defender_defense
        
        # Get types for STAB and effectiveness
        attacker_type1, attacker_type2 = BattleHelpers.get_pokemon_types(attacker)
        defender_type1, defender_type2 = BattleHelpers.get_pokemon_types(defender)
        
        # Calculate STAB
        stab = 1.5 if BattleHelpers.has_stab(move_data["type"], attacker_type1, attacker_type2) else 1.0
        
        # Calculate type effectiveness
        type_eff = BattleHelpers.calculate_type_effectiveness(move_data["type"], defender_type1, defender_type2)
        
        # Base damage (Gen 3 formula)
        damage = ((2 * attacker_level / 5 + 2) * move_data["power"] * attack_stat / defense_stat / 50 + 2)
        
        # Apply modifiers
        if is_critical:
            damage *= 2
        
        damage = damage * stab * type_eff
        
        # Random factor (0.85 - 1.0)
        damage_min = int(damage * 0.85)
        damage_max = int(damage * 1.0)
        
        return (damage_min, damage_max)
    
    @staticmethod
    def can_ko(attacker: np.ndarray, defender: np.ndarray, move_index: int) -> bool:
        """
        Checks if a move can guarantee a KO (minimum damage >= opponent HP).
        
        Args:
            attacker: Attacker Pokemon data
            defender: Defender Pokemon data
            move_index: Move slot to check
            
        Returns:
            True if move guarantees KO
        """
        damage_min, _ = BattleHelpers.calculate_damage(attacker, defender, move_index)
        defender_hp = int(defender[ObsIdx.RAW_DATA["HP"]])
        
        return damage_min >= defender_hp
    
    @staticmethod
    def get_alive_pokemon_count(pokemon_list: List[np.ndarray]) -> int:
        """
        Counts number of alive Pokemon in a team.
        
        Args:
            pokemon_list: List of 6 Pokemon arrays
            
        Returns:
            Number of Pokemon with HP > 0
        """
        count = 0
        for pkmn in pokemon_list:
            hp = int(pkmn[ObsIdx.RAW_DATA["HP"]])
            if hp > 0:
                count += 1
        return count
    
    @staticmethod
    def get_fastest_pokemon(pokemon_list: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Finds the fastest alive Pokemon in the team.
        
        Args:
            pokemon_list: List of 6 Pokemon arrays
            
        Returns:
            Fastest alive Pokemon or None
        """
        fastest = None
        max_speed = -1
        
        for pkmn in pokemon_list:
            hp = int(pkmn[ObsIdx.RAW_DATA["HP"]])
            if hp > 0:
                speed = int(pkmn[ObsIdx.RAW_DATA["stats_begin"] + 2])  # SPEED stat
                if speed > max_speed:
                    max_speed = speed
                    fastest = pkmn
        
        return fastest
    
    @staticmethod
    def will_outspeed(pokemon1: np.ndarray, pokemon2: np.ndarray) -> bool:
        """
        Checks if pokemon1 is faster than pokemon2.
        
        Args:
            pokemon1: First Pokemon data
            pokemon2: Second Pokemon data
            
        Returns:
            True if pokemon1 has higher speed stat
        """
        speed1 = int(pokemon1[ObsIdx.RAW_DATA["stats_begin"] + 2])
        speed2 = int(pokemon2[ObsIdx.RAW_DATA["stats_begin"] + 2])
        
        return speed1 > speed2
    
    @staticmethod
    def has_status_condition(pokemon: np.ndarray) -> bool:
        """
        Checks if Pokemon has a status condition.
        
        Args:
            pokemon: Pokemon data array
            
        Returns:
            True if Pokemon is poisoned, burned, paralyzed, asleep, or frozen
        """
        status = int(pokemon[ObsIdx.RAW_DATA["status"]])
        return status != 0
    
    @staticmethod
    def get_available_switches(pokemon_list: List[np.ndarray]) -> List[int]:
        """
        Gets list of valid switch action indices (Pokemon that are alive and not active).
        
        Args:
            pokemon_list: List of 6 Pokemon arrays
            
        Returns:
            List of action indices (4-9) for valid switches
        """
        available_switches = []
        
        for idx, pkmn in enumerate(pokemon_list):
            is_active = int(pkmn[ObsIdx.RAW_DATA["is_active"]])
            hp = int(pkmn[ObsIdx.RAW_DATA["HP"]])
            
            if is_active == 0 and hp > 0:
                # Switch action index = 4 + pokemon slot
                available_switches.append(4 + idx)
        
        return available_switches
