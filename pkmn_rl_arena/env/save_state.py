from pkmn_rl_arena.paths import PATHS
from pkmn_rl_arena.logging import log
from pathlib import Path

from .battle_core import BattleState, CoreContext

import os
import re
from typing import Optional


class SaveStateManager:
    """
    Manages emulator save states for quick save/load functionality.
    """

    save_file_ext = "savestate"

    regex_file_ext = re.compile(f".+\.{save_file_ext}")

    def __init__(self, core_context: CoreContext):
        self.ctxt: CoreContext = core_context
        self.save_dir: Path = PATHS["SAVE"]
        self.save_states: list[Path] = []
        self.state_wild_card = "_turntype:*_step:*.savestate"
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def buid_save_path(name: str, state: BattleState):
        return f"{name}_turntype:{state.turn.value}_step:{state.step}_id:{state.id}.{SaveStateManager.save_file_ext}"

    def save_state(self, name: str):
        """
        Save current state with the given name.
        Note:
            The save state name will be appended current turntype & episode_steps
            i.e.:
                name = boot_state
                state = { turntype = Turntype.CREATE_TEAM , episode_steps = 0}
                final name = boot_state_turntype:Turntype.CREATE_TEAM_step:0.savestate"
        Return:
            save_path of save_state
        """
        save_path = SaveStateManager.buid_save_path(name, self.ctxt.core.state)
        self.save_states.append(save_path)
        self.ctxt.core.save_savestate(save_path)
        return save_path

    def _resolve_name(self, save_name: str | None) -> Path:

        # case 3
        if save_name is not None and re.match(SaveStateManager.regex_file_ext, save_name): 
            return Path(save_name)
        elif save_name is None:
        # case 1
            log.warn(
                f"Called load_save_state but the entry options save state name is {save_name}. Loading first state saved."
            )
            if len(self.save_states) == 0:
                log.fatal("No save state available to load. Exiting.")
                exit(1)
            return self.save_states[0]
        # case 2
        else :
            log.debug(
                f"Given state without file ext, attempting to load 1st state whose name matches regex {save_name}.+"
            )

            for save_state in self.save_states:
                if re.match(f"{save_name}.+", str(save_state)):
                    return save_state

            raise ValueError(f"Save state not found at path : {save_name} exiting.")


    def load_state(self, name: str | None) -> Optional[BattleState]:
        """
        Loads a saved state specified via name argument.
        extension : _turntype:<turn>_step:<nb_step>.savestate
        Args:
            name : 3 possible values :
                1. None: Retrieves latests savestate and loads it
                2. Name without extension : loads 1st matching save state in self.savestates
                3. Name with extension : loads save state

        Returns :
            True if successful, False otherwise.
        Raises :
            If state given by arg does not exist
        """

        path_to_load: Path = self._resolve_name(name)
        log.debug(f"Attempting to load following save state: {path_to_load}")
        return self.ctxt.core.load_savestate(path_to_load, init=False)

    def remove_save_states(self):
        """Delete all save states."""
        for path in self.save_states:
            os.remove(os.path.join(PATHS["SAVE"], path))
        return
