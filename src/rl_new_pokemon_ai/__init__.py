import os
from pathlib import Path
from logging import setup_colored_logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROM_PATH = os.path.join(BASE_DIR, "../pokeemerald_ai_rl/pokeemerald_modern.elf")
BIOS_PATH = os.path.join(BASE_DIR, "../rustboyadvance-ng-for-rl/gba_bios.bin")
MAP_PATH = os.path.join(BASE_DIR, "../pokeemerald_ai_rl/pokeemerald_modern.map")
POKEMON_CSV_PATH = os.path.join(BASE_DIR, "../data/csv_data/pokemon_data.csv")
PKMN_MOVES_PATH = Path(BASE_DIR / Path("../data/csv_data/moves_data.csv"))
SAVE_PATH = os.path.join(BASE_DIR, "../savestate")
