import pandas as pd
import os
import re

def parse_pokemon_scrap(path):
    data = []
    with open(path) as f:
        lines = f.readlines()

    current = {}
    moves = []
    for line in lines:
        line = line.strip()
        if line.startswith("speciesName:"):
            if current:
                current['moves'] = moves
                data.append(current)
                current = {}
                moves = []
            current['speciesName'] = line.split(":", 1)[1].strip()
        elif line.startswith("id:"):
            current['id'] = int(line.split(":", 1)[1].strip())
        elif line.startswith("baseHP:"):
            current['baseHP'] = int(line.split(":", 1)[1].strip())
        elif line.startswith("baseAttack:"):
            current['baseAttack'] = int(line.split(":", 1)[1].strip())
        elif line.startswith("baseDefense:"):
            current['baseDefense'] = int(line.split(":", 1)[1].strip())
        elif line.startswith("baseSpeed:"):
            current['baseSpeed'] = int(line.split(":", 1)[1].strip())
        elif line.startswith("baseSpAttack:"):
            current['baseSpAttack'] = int(line.split(":", 1)[1].strip())
        elif line.startswith("baseSpDefense:"):
            current['baseSpDefense'] = int(line.split(":", 1)[1].strip())
        elif line.startswith("type0:"):
            current['type0'] = int(line.split(":", 1)[1].strip())
        elif line.startswith("type1:"):
            current['type1'] = int(line.split(":", 1)[1].strip())
        elif line.startswith("ability0:"):
            current['ability0'] = int(line.split(":", 1)[1].strip())
        elif line.startswith("ability1:"):
            current['ability1'] = int(line.split(":", 1)[1].strip())
        elif re.search(r"move\d+:\s*(\d+)", line):
            m = re.search(r"move\d+:\s*(\d+)", line)
            move_id = int(m.group(1))
            moves.append(move_id)
    if current:
        current['moves'] = moves
        data.append(current)

    df = pd.DataFrame(data)
    return df

def parse_moves_file(path):
    data = []
    current = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[INFO] GBA Debug: moveName:"):
                if current:
                    data.append(current)
                    current = {}
                move_name = line.split("moveName:", 1)[1].strip()
                current["moveName"] = move_name
            elif re.match(r"^\w+:", line):
                key, value = line.split(":", 1)
                value = value.strip()
                try:
                    value = int(value)
                except ValueError:
                    pass
                current[key] = value
        if current:
            data.append(current)
    df = pd.DataFrame(data)
    return df
