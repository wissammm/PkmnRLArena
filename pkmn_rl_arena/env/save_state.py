from pathlib import Path
from pkmn_rl_arena import SAVE_PATH
from pkmn_rl_arena.logging import logger

from .battle_core import BattleCore

import os
from typing import List

from regex import re


class SaveStateManager:
    """
    Manages emulator save states for quick save/load functionality.
    """

    regex = {
        "stamp": re.compile(".+_turntype:[[:alpha]]_step:\d+.*"),
        "file_ext": re.compile(".+\.savestate"),
    }

    def __init__(self, battle_core: BattleCore):
        self.battle_core = battle_core
        self.save_dir = SAVE_PATH
        self.save_states = []
        self.state_wild_card = "_turntype:*_step:*.savestate"
        os.makedirs(self.save_dir, exist_ok=True)

    def save_state(self, name: str):
        """
        Save current state with the given name.
        Note:
            The save state name will be appended current turntype & episode_steps
            i.e.:
                name = boot_state
                state = { turntype = Turntype.CREATE_TEAM , episode_steps = 0}
                final name = boot_state_turntype:Turntype.CREATE_TEAM_step:0.savestate"
        """
        name += f"_turntype:{self.battle_core.state.turn}_step:{self.battle_core.state.step}"
        self.save_states.append(name)
        self.battle_core.save_savestate(name)

    def load_state(self, name: str | None) -> bool:
        """
        Loads a saved state specified via name argument.
        extension : _turntype:<turn>_step:<nb_step>.savestate
        Args:
            name : 3 possible values :
                1. None: Retrieves latests savestate and loads it
                2. Name without extension : loads 1st matching save state in self.save_states
                3. Name with extension : loads save state

        Returns :
            True if successful, False otherwise.
        Raises :
            If state given by arg does not exist
        """
        # case 1
        if name is None:
            logger.warn(
                f"Called load_save_state but the entry options save state name is {name}. Loading first state saved."
            )
            if len(self.save_states) == 0:
                logger.fatal("No save state available to load. Exiting.")
                exit(1)
            return self.battle_core.load_savestate(self.save_states[0])

        # case 2
        if not re.match(SaveStateManager.regex["file_ext"], name):
            logger.debug(f"Given state without file ext, attempting to load 1st state whose name matches regex {name}.+")
            for save in self.save_states:
                if re.match(f"{name}.+", save):
                    found
                    return self.battle_core.load_savestate(save)
            if name not in self.save_states:
                raise ValueError(f"Save state not found at path : {name} exiting.")
        # case 3
        else:
            return self.battle_core.load_savestate(name)

    def remove_save_states(self):
        """Delete all save states."""
        for path in self.save_states:
            os.remove(path)
        return
