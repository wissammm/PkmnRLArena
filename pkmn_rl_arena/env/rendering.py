from pandas.core.interchange.dataframe_protocol import Column
from pkmn_rl_arena.env.observation import Observation, ObsIdx
from pkmn_rl_arena.env.pkmn_team_factory import DataSize, PkmnTeamFactory

import numpy as np
from rich.console import Console, RenderHook, Group
from rich.table import Table
from rich.progress import track, Progress, ProgressBar

from rich.text import Text


from typing import Dict
from numpy import typing as npt


# class GameRendering(RenderHook):
class GameRendering:
    def __init__(self, team_factory: PkmnTeamFactory, agents=["player", "enemy"]):
        self.console = Console()
        self.team_factory = team_factory
        self.agents = agents
        self.main_table = Table()
        self.main_table.add_column("Player", justify="center", header_style="Red")
        self.main_table.add_column("Enemy", justify="center", header_style="Blue")
        self.agent_tables = {agent: Table() for agent in self.agents}

    def refresh(
        self,
        obs: Observation,
        reward: Dict[str, float],
    ):
        table = Table()
        for agent, table in self.agent_tables.items():
            table.rows.clear()  # or rebuild contents

            stats_changes = obs.stat_changes()
            active_pkmn = obs.active_pkmn()

            for i, pkmn in enumerate(np.split(obs.agent(agent), DataSize.PARTY_SIZE)):
                # Left column
                pkmn_id = pkmn[ObsIdx.RAW_DATA["species"]]
                if pkmn_id == 0:
                    continue
                name = self.team_factory.get_pkmn_name_from(pkmn_id)
                types = (
                    pkmn[ObsIdx.RAW_DATA["type_1"]],
                    pkmn[ObsIdx.RAW_DATA["type_2"]],
                )

                status = stats_changes[agent]
                status_str = ""
                for status_change in status[i]:
                    status_str += f"{status_change} "
                status_str = status_str[:-1]
                left_col = Group(
                    Text(name), Text(f"{types[0]},{types[1]}"), Text(status_str)
                )

                # right column
                max_hp = int(pkmn[ObsIdx.RAW_DATA["max_HP"]])
                hp = int(pkmn[ObsIdx.RAW_DATA["HP"]])
                hp_bar = ProgressBar(total=max_hp, completed=hp, width=20, style="green")
                right_renderables  = [hp_bar]
                for move_offset in range(
                    ObsIdx.RAW_DATA["moves_begin"],
                    ObsIdx.NB_DATA_PKMN,
                    ObsIdx.RAW_DATA["move_slot_stride"],
                ):
                    name = self.team_factory.get_move_name_from(
                        pkmn[move_offset + ObsIdx.RAW_DATA["move_id_offset"]]
                    )
                    type = pkmn[move_offset + ObsIdx.RAW_DATA["type_offset"]]
                    pp = pkmn[move_offset + ObsIdx.RAW_DATA["pp_offset"]]

                    right_renderables.append(
                        Text(f"{name}\t\t{type}\t\tpp : {pp}")
                    )
                right_col = Group(*right_renderables)
                table.add_row(
                    left_col, right_col,
                )

        self.main_table.add_row(
            self.agent_tables["player"], self.agent_tables["enemy"]
        )

        self.console.print(self.main_table)

        return
