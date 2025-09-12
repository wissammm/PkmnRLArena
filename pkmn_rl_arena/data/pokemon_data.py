import pandas as pd

def to_pandas_mon_dump_data(array):
    """Convert Pokemon data array to pandas DataFrame with named columns"""
    data = {
        'id': array[0],
        'isActive': array[1],
        'baseAttack': array[2],
        'baseDefense': array[3],
        'baseSpeed': array[4],
        'baseSpAttack': array[5],
        'baseSpDefense': array[6],
        'ability_num': array[7],
        'type0': array[8],
        'type1': array[9],
        'current_hp': array[10],
        'level': array[11],
        'friendship': array[12],
        'max_hp': array[13],
        'held_item': array[14],
        'pp_bonuses': array[15],
        'personality': array[16],
        'status1': array[17],
        'status2': array[18],
        'status3': array[19],
        'move1': array[20],
        'move1_pp': array[21],
        'move2': array[22],
        'move2_pp': array[23],
        'move3': array[24],
        'move3_pp': array[25],
        'move4': array[26],
        'move4_pp': array[27],
    }

    return pd.DataFrame([data])
    
    return pd.DataFrame([data])

def to_pandas_team_dump_data(array):
    """Convert a Pokémon team data array to a pandas DataFrame"""
    team_data = []
    for i in range(6): 
        start = i * 28
        end = start + 28
        mon_data = {
            'id': array[start + 0],
            'isActive': array[start + 1],
            'baseAttack': array[start + 2],
            'baseDefense': array[start + 3],
            'baseSpeed': array[start + 4],
            'baseSpAttack': array[start + 5],
            'baseSpDefense': array[start + 6],
            'ability_num': array[start + 7],
            'type0': array[start + 8],
            'type1': array[start + 9],
            'current_hp': array[start + 10],
            'level': array[start + 11],
            'friendship': array[start + 12],
            'max_hp': array[start + 13],
            'held_item': array[start + 14],
            'pp_bonuses': array[start + 15],
            'personality': array[start + 16],
            'status1': array[start + 17],
            'status2': array[start + 18],
            'status3': array[start + 19],
            'move1': array[start + 20],
            'move1_pp': array[start + 21],
            'move2': array[start + 22],
            'move2_pp': array[start + 23],
            'move3': array[start + 24],
            'move3_pp': array[start + 25],
            'move4': array[start + 26],
            'move4_pp': array[start + 27],
        }
        team_data.append(mon_data)
    
    return pd.DataFrame(team_data)
