from pkmn_rl_arena import log

from dataclasses import dataclass
from enum import Enum


class TurnType(Enum):
    """Enumeration for different turn types"""

    CREATE_TEAM = 0  # Initial team creation
    GENERAL = 1  # Both players act simultaneously
    PLAYER = 2  # Only player acts
    ENEMY = 3  # Only enemy acts
    DONE = 4  # Battle is finished


@dataclass
class BattleState:
    """
    Represents the current state of the battle
    NOTE :
        DO NOTE USE THIS CLASS CONSTRUCTOR.
        STATES MUST ONLY BE MANUFACTURED THROUGH StateFactory
    """

    id: int = 0  # unique id for this run
    step: int = 0  # nb step in this run
    turn: TurnType = TurnType.CREATE_TEAM  # current battle turntype

    def __eq__(self, other) -> bool:
        """This is a helper for tests, it doesn't tests the id voluntarly ad it is unique."""
        return self.step == other.step and self.turn == other.turn


class BattleStateFactory:
    id_gen: int = 0  # unique episode id  generator
    id: int = 0  # unique id for this run

    def build(self, current_turn=TurnType.CREATE_TEAM, step=0) -> BattleState:
        if step < 0:
            raise ValueError(
                f"Attempting to create a step with negative step, step value must be strictly > 0, got {step}"
            )
        id = BattleStateFactory.id_gen
        BattleStateFactory.id_gen += 1
        return BattleState(id=id, step=step, turn=current_turn)

    @staticmethod
    def from_save_path(save_path: str) -> BattleState:
        """
        Creates state from save path, assuming save path is of shape :
        {save_name}_turntype:{turntype.value}_step:{step}_id:{id}.savestate
        """
        log.debug(f"Creating battlestate from {save_path}")
        data = save_path.split(".")[-2].split("_")[-3:]

        turntype = TurnType(int(data[0][data[0].find(":") + 1 :]))
        step = int(data[1][data[1].find(":") + 1 :])
        id = int(data[2][data[2].find(":") + 1 :])

        return BattleState(id, step, turntype)

