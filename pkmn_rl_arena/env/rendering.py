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
    255: "NONE",
    0: "NORM",
    1: "FIGHT",
    2: "FLY",
    3: "PSN",
    4: "GRND",
    5: "ROCK",
    6: "BUG",
    7: "GHOST",
    8: "STEEL",
    9: "MYSTERY",
    10: "FIRE",
    11: "WTR",
    12: "GRASS",
    13: "ELEK",
    14: "PSY",
    15: "ICE",
    16: "DRAGON",
    17: "DARK",
    18: "NUMBER_OF_MON_TYPES",
}


class GameRendering:
    def __init__(self, team_factory: PkmnTeamFactory, agents=["player", "enemy"]):
        self.console = Console()
        self.team_factory = team_factory
        self.agents = agents
        self.live = None

    def _build_table(self, obs: Observation, reward: Dict[str, float]) -> Table:
        main_table = Table(expand=True, padding=(0, 1))
        main_table.add_column(
            f"Player\tReward {reward['player']}", justify="center", header_style="Red"
        )
        main_table.add_column(
            f"Enemy\tReward : {reward['enemy']}", justify="center", header_style="Blue"
        )

        agent_tables = {
            a: Table(
                expand=True,
                show_header=False,
                show_edge=False,
                show_lines=True,
                padding=(0, 1),
            )
            for a in self.agents
        }

        stats = obs.stats()
        lvls = obs.lvl()
        active_pkmn = obs.active_pkmn()

        for agent, table in agent_tables.items():
            # table.add_column("Pokémon", justify="left", ratio=4, no_wrap=False)
            # table.add_column("Stats / Moves", justify="left", ratio=5, no_wrap=False)

            for i, pkmn in enumerate(np.split(obs.agent(agent), DataSize.PARTY_SIZE)):
                pkmn_id = pkmn[ObsIdx.RAW_DATA["species"]]
                if pkmn_id == 0:
                    continue
                pkmn_table = Table(show_header=False, show_lines=False, show_edge=False)

                ############################
                # NAME & active indicator
                is_active = True if i == active_pkmn[agent] else False
                active_indicator = Text("● ", style="red" if is_active else "grey50")

                name = self.team_factory.get_pkmn_name_from(pkmn_id)
                types = (
                    int(pkmn[ObsIdx.RAW_DATA["type_1"]]),
                    int(pkmn[ObsIdx.RAW_DATA["type_2"]]),
                )
                types_str = "Types : "
                if types[0] != types[1]:
                    types_str += (
                        f"{str(TYPES_ID[types[0]]):^6} / {str(TYPES_ID[types[1]]):^6}"
                    )
                else:
                    types_str += f"{str(TYPES_ID[types[0]]):^6}"

                ############################
                # HP bar
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
                    width=None,
                    complete_style=bar_style,  # <- filled part
                    finished_style=bar_style,  # <- when bar is at 100%
                    style="grey37",  # <- background (empty slots)
                )

                ############################
                # Stats
                stats_table = Table(show_edge=False,show_footer=False,show_lines=False,box=None)
                stats_table.add_column(header="Atk")
                stats_table.add_column(header="Def")
                stats_table.add_column(header="Speed")
                stats_table.add_column(header="Sp.atk")
                stats_table.add_column(header="Sp.Def")
                stats_table.add_row(*[f"{str(stat):<4}" for stat in stats[agent][i]])
                pkmn_table.add_row(
                    Group(
                        active_indicator
                        + Text(
                            f"{name:<12}{types_str} Lvl {str(lvls[agent][i]):<5}",
                            style="bold",
                        ),
                        stats_table,
                        hp_bar,
                    )
                )

                ############################
                # moves
                moves_renderable = []
                for mv_offset in range(
                    ObsIdx.RAW_DATA["moves_begin"],
                    ObsIdx.NB_DATA_PKMN,
                    ObsIdx.RAW_DATA["move_slot_stride"],
                ):
                    mv_id = pkmn[mv_offset + ObsIdx.RAW_DATA["move_id_offset"]]
                    if mv_id == 0:
                        continue
                    mv_name = self.team_factory.get_move_name_from(mv_id)
                    mv_type = TYPES_ID[pkmn[mv_offset + ObsIdx.RAW_DATA["type_offset"]]]
                    pp = pkmn[mv_offset + ObsIdx.RAW_DATA["pp_offset"]]
                    max_pp = self.team_factory.moves.loc[mv_id, "pp"]
                    pp_str = f"{pp}/{max_pp}"
                    dmg = pkmn[mv_offset + ObsIdx.RAW_DATA["power_offset"]]
                    moves_renderable.append(
                        Text(
                            f"\n{mv_name:<14} {mv_type:<7}Dmg : {str(dmg):<2} pp: {pp_str}"
                        )
                    )

                moves_table = Table(
                    show_edge=False, show_header=False, show_footer=False
                )

                match len(moves_renderable):
                    case 1:
                        moves_table.add_row(moves_renderable[0], "")
                    case 2:
                        moves_table.add_row(moves_renderable[0], moves_renderable[1])
                    case 3:
                        moves_table.add_row(moves_renderable[0], moves_renderable[1])
                        moves_table.add_row(moves_renderable[2], "")
                    case 4:
                        moves_table.add_row(moves_renderable[0], moves_renderable[1])
                        moves_table.add_row(moves_renderable[2], moves_renderable[3])
                pkmn_table.add_row(moves_table)
                table.add_row(pkmn_table)

        main_table.add_row(agent_tables["player"], agent_tables["enemy"])

        return main_table

    def _build_state_panel(self, state: BattleState):
        return Panel(Text(f"{state.id} step : {state.step} turn type : {state.turn}"))

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
            self.live.update(
                Group(self._build_state_panel(state), self._build_table(obs, reward))
            )
            self.live.refresh()
