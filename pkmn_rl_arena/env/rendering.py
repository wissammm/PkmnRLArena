from pkmn_rl_arena import log
from pkmn_rl_arena.env.battle_state import BattleState
from pkmn_rl_arena.env.observation import Observation, ObsIdx
from pkmn_rl_arena.env.pkmn_team_factory import DataSize, PkmnTeamFactory

import numpy as np
from rich.console import Console, RenderHook, Group
from rich.progress import track, Progress, ProgressBar
from rich.table import Table
from rich.panel import Panel

from rich.text import Text


from typing import Dict
from numpy import typing as npt

from rich.live import Live

TYPES_ID = {
    0: "NONE",
    1: "NORMAL",
    2: "FIGHTING",
    3: "FLYING",
    4: "POISON",
    5: "GROUND",
    6: "ROCK",
    7: "BUG",
    8: "GHOST",
    9: "STEEL",
    10: "MYSTERY",
    11: "FIRE",
    12: "WATER",
    13: "GRASS",
    14: "ELECTRIC",
    15: "PSYCHIC",
    16: "ICE",
    17: "DRAGON",
    18: "DARK",
    19: "NUMBER_OF_MON_TYPES",
}


class GameRendering:
    def __init__(self, team_factory: PkmnTeamFactory, agents=["player", "enemy"]):
        self.console = Console()
        self.team_factory = team_factory
        self.agents = agents
        self.live = None

    def _build_table(self, obs: Observation, reward: Dict[str, float]) -> Table:
        main_table = Table(expand=True)
        main_table.add_column("Player", justify="center", header_style="Red")
        main_table.add_column("Enemy", justify="center", header_style="Blue")

        agent_tables = {
            a: Table(expand=True, show_header=False, show_edge=False, show_lines=True)
            for a in self.agents
        }

        stats = obs.stats()
        lvls = obs.lvl()

        for agent, table in agent_tables.items():
            table.add_column("Pokémon", justify="left", ratio=4, no_wrap=False)
            table.add_column("Stats / Moves", justify="left", ratio=5, no_wrap=False)

            for i, pkmn in enumerate(np.split(obs.agent(agent), DataSize.PARTY_SIZE)):
                pkmn_id = pkmn[ObsIdx.RAW_DATA["species"]]
                if pkmn_id == 0:
                    continue

                # name + types + status
                name = self.team_factory.get_pkmn_name_from(pkmn_id)
                types = (
                    TYPES_ID[int(pkmn[ObsIdx.RAW_DATA["type_1"]])],
                    TYPES_ID[int(pkmn[ObsIdx.RAW_DATA["type_2"]])],
                )

                status_str = " ".join(str(stats[agent][i]))
                left_col = Group(
                    Text(f"{name} LVL: {lvls[agent][i]}", style="bold"),
                    Text(f"{types[0]}, {types[1]}"),
                    Text(status_str),
                )

                # HP bar + moves
                max_hp = int(pkmn[ObsIdx.RAW_DATA["max_HP"]])
                hp = int(pkmn[ObsIdx.RAW_DATA["HP"]])
                hp_percent = hp / max_hp
                match hp_percent:
                    case _ if hp_percent >= 0.5:
                        bar_style = "green"
                    case _ if hp_percent >= 0.2:
                        bar_style = "yellow"
                    case _:
                        bar_style = "red"
                hp_bar = ProgressBar(
                    total=max_hp,
                    completed=hp,
                    width=20,
                    complete_style=bar_style,  # <- filled part
                    finished_style=bar_style,  # <- when bar is at 100%
                    style="grey37",  # <- background (empty slots)
                )
                moves_renderable = []
                for move_offset in range(
                    ObsIdx.RAW_DATA["moves_begin"],
                    ObsIdx.NB_DATA_PKMN,
                    ObsIdx.RAW_DATA["move_slot_stride"],
                ):
                    move_name = self.team_factory.get_move_name_from(
                        pkmn[move_offset + ObsIdx.RAW_DATA["move_id_offset"]]
                    )
                    move_type = pkmn[move_offset + ObsIdx.RAW_DATA["type_offset"]]
                    pp = pkmn[move_offset + ObsIdx.RAW_DATA["pp_offset"]]
                    moves_renderable.append(
                        Text(f"\n{move_name}\t{move_type}\tpp: {pp}")
                    )
                right_col = Group(hp_bar, *moves_renderable)

                table.add_row(left_col, right_col)

        main_table.add_row(agent_tables["player"], agent_tables["enemy"])

        return main_table

    def start(self, obs, reward):
        """Start live rendering once"""
        self.live = Live(
            self._build_table(obs, reward),
            console=self.console,
            auto_refresh=False,
            transient=False,
        )
        self.live.start()

    def stop(self):
        """Stop live rendering"""
        if self.live:
            self.live.stop()
            self.live = None

    def refresh(self, obs: Observation, reward: Dict[str, float], state: BattleState):
        """Trigger manual refresh"""
        if self.live:
            # Panel(Text("state : {}"))
            self.live.update(self._build_table(obs, reward))
            self.live.refresh()
