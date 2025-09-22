import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "ROM": os.path.join(BASE_DIR, "../pokeemerald_ai_rl/pokeemerald_modern.elf"),
    "BIOS": os.path.join(BASE_DIR, "../rustboyadvance-ng-for-rl/gba_bios.bin"),
    "MAP": os.path.join(BASE_DIR, "../pokeemerald_ai_rl/pokeemerald_modern.map"),
    "POKEMON_CSV": os.path.join(BASE_DIR, "../data/csv_data/pokemon_data.csv"),
    "MOVES_CSV": os.path.join(BASE_DIR, "../data/csv_data/moves_data.csv"),
    "SAVE": os.path.join(BASE_DIR, "../savestate"),
    "GBA": os.path.join(BASE_DIR, "export/gba"),
    "PARAMETERS": os.path.join(BASE_DIR, "export/templates/parameters.jinja"),
}
