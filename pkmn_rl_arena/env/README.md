# Pkmn RL Arena

This sub pkg is the PettingZoo environment to help our RL agents to learn.

# Env
defined by 2 players :
1. Player
2. Ennemy

Each player has a team containing up to 6 pokemons.
Each pokemon is defined described by 35 values 

# Action & observation spaces 
## Action space
At each turn our agent has 9 possible actions:
| IDX  | Action                    |
|------|---------------------------|
|0 - 3 | Moves                     |
|4 - 9 | Switch with another pokemon

# PKMN DESCRITPION
Pokemons are described in 28 values by the game : 
1 (dst[0])  | species (MON_DATA_SPECIES)
2 (dst[1])  | isActive (0/1)
3 (dst[2])  | attack (MON_DATA_ATK)
4 (dst[3])  | defense (MON_DATA_DEF)
5 (dst[4])  | speed (MON_DATA_SPEED)
6 (dst[5])  | special attack (MON_DATA_SPATK)
7 (dst[6])  | special defense (MON_DATA_SPDEF)
8 (dst[7])  | ability number (MON_DATA_ABILITY_NUM)
9 (dst[8])  | type1 (from species)
10 (dst[9]) | type2 (from species)
11 (dst[10])| current HP (MON_DATA_HP)
12 (dst[11])| level (MON_DATA_LEVEL)
13 (dst[12])| friendship (MON_DATA_FRIENDSHIP)
14 (dst[13])| max HP (MON_DATA_MAX_HP)
15 (dst[14])| held item (MON_DATA_HELD_ITEM)
16 (dst[15])| PP bonuses (MON_DATA_PP_BONUSES)
17 (dst[16])| personality (MON_DATA_PERSONALITY)
18 (dst[17])| status (MON_DATA_STATUS)
19 (dst[18])| status2 (reserved / currently 0)
20 (dst[19])| status3 (reserved / currently 0)
21 (dst[20])| move1 id (MON_DATA_MOVE1)
22 (dst[21])| move1 pp (MON_DATA_PP1)
23 (dst[22])| move2 id (MON_DATA_MOVE2)
24 (dst[23])| move2 pp (MON_DATA_PP2)
25 (dst[24])| move3 id (MON_DATA_MOVE3)
26 (dst[25])| move3 pp (MON_DATA_PP3)
27 (dst[26])| move4 id (MON_DATA_MOVE4)
28 (dst[27])| move4 pp (MON_DATA_PP4)




                          
Of these we remove SPECIES & IV's and we add moves data : 
|IDX | value                                    |
|----|------------------------------------------|
| 1  | pkmn_ID                                  |
| 2  | IS_ACTIVE                                |
|    | STATS :                                  |
| 3  |   ATK                                    |
| 4  |   DEF                                    |
| 5  |   SPEED                                  |
| 6  |   SPATK                                  |
| 7  |   SPDEF                                  |
| 8  | ABILITY                                  |
| 9  | ABILITY(placeholder)                     |
| 10 | SPECIE_1                                 |
| 11 | SPECIE_2                                 |
|    | STATUS:                                  |
| 12 |   HP                                     |
| 13 |   LEVEL                                  |
| 14 |   FRIENDSHIP                             |
| 15 |   MAX_HP                                 |
| 16 |   HELD_ITEM                              |
| 17 |   PP_BONUSES                             |
| 18 |   PERSONALITY                            |
| 19 |   STATUS_1                               | 
| 20 |   STATUS_2                               | 
| 21 |   STATUS_3                               | 
|    | MOVE_1                                   |
| 22 |   id                                     |
| 23 |   MAX_PP                                 |
| 24 |   CURR_PP                                |
| 25 |   effect                                 |
| 26 |   power                                  |
| 27 |   type                                   |
| 28 |   accuracy                               |
| 29 |   curr_pp                                |
| 30 |   max_pp                                 |
| 31 |   priority                               |
| 32 |   secondaryeffectchance                  |
| 33 |   target                                 |
| 34 |   flags                                  |
|    | MOVE_2                                   |
| 35 |   id                                     |
| 36 |   MAX_PP                                 |
| 37 |   CURR_PP                                |
| 38 |   effect                                 |
| 39 |   power                                  |
| 40 |   type                                   |
| 41 |   accuracy                               |
| 42 |   curr_pp                                |
| 43 |   max_pp                                 |
| 44 |   priority                               |
| 45 |   secondaryeffectchance                  |
| 46 |   target                                 |
| 47 |   flags                                  |
|    | MOVE_3                                   |
| 48 |   id                                     |
| 49 |   MAX_PP                                 |
| 50 |   CURR_PP                                |
| 51 |   effect                                 |
| 52 |   power                                  |
| 53 |   type                                   |
| 54 |   accuracy                               |
| 55 |   curr_pp                                |
| 56 |   max_pp                                 |
| 57 |   priority                               |
| 58 |   secondaryeffectchance                  |
| 59 |   target                                 |
| 60 |   flags                                  |
|    | MOVE_4                                   |
| 61 |   id                                     |
| 62 |   MAX_PP                                 |
| 63 |   CURR_PP                                |
| 64 |   effect                                 |
| 65 |   power                                  |
| 66 |   type                                   |
| 67 |   accuracy                               |
| 68 |   curr_pp                                |
| 69 |   max_pp                                 |
| 70 |   priority                               |
| 71 |   secondaryeffectchance                  |
| 72 |   target                                 |
| 73 |   flags                                  |
