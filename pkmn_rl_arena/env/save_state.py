from pkmn_rl_arena import SAVE_PATH
from pkmn_rl_arena.logging import logger

from .battle_core import BattleCore

import os
from typing import List


class SaveStateManager:
    """
    Manages emulator save states for quick save/load functionality.
    """

    def __init__(self, battle_core: BattleCore):
        self.battle_core = battle_core
        self.save_dir = SAVE_PATH
        self.save_states = []
        os.makedirs(self.save_dir, exist_ok=True)

    def save_state(self, name: str):
        """Save current state with the given name."""
        self.battle_core.save_savestate(name)
        self.save_states.append(name)

    def load_state(self, name: str | None) -> bool:
        """Load a saved state by name. Returns True if successful, False otherwise."""
        if name is None:
            logger.warn(
                f'Called load_save_state but the entry options save state name is {None}. Loading first state saved.'
            )
            if len(self.save_states) == 0 :
                logger.fatal("No save state available to load. Exiting.")
                exit(1)
            name = self.save_states[0]

        if not self.has_state(name):
            raise ValueError(f"Save state not found at path : {name} exiting.")

        return self.battle_core.load_savestate(name)

    def list_save_states(self) -> List[str]:
        """List all available save state names (without extension)."""
        if not os.path.exists(self.save_dir):
            return []
        files = os.listdir(self.save_dir)
        return [f[:len(SAVE_PATH)] for f in files if f.endswith(".savestate")]

    def has_state(self, name: str) -> bool:
        """Check if a save state with the given name exists."""
        return os.path.exists(os.path.join(self.save_dir, f"{name}.savestate"))

    def remove_save_states(self) :
        """Delete all save states."""
        pass
        # os.de
