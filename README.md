# PkmnRLArena

Reinforcement learning environment and tools for Pokémon Emerald : 
    - Modified Emulator to have python librairy 
    - PettingZoo environnement for MARL
    - Quantization 
    - Neural Network graph manipulation 
    - Export 

##  Installation 
### Linux 
1. **Clone the repository**
git clone --recursive
2. **Set up a Python virtual environment**
3. **Install Python dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Build and install the rust emulator extension**
    ```
    cd rustboyadvance-ng-for-rl/platform/rustboyadvance-py
    maturin develop --release --features elf_support 
    ```

5. **Compile the rom**

To compile the custom pokeemerald rom, go to see directly the INSTALL.md inside pokeemerald_ai_rl. Replace make modern by 
```bash
make modern DINFO=1 DOBSERVED_DATA=1 DSKIP_TEXT=1 DSKIP_GRAPHICS=1 NO_DEBUG_PRINT=1 -j
```
 - Fork of [rustboyadvance](https://github.com/michelhe/rustboyadvance-ng) a gba emulator written in rust :crab:.
   We added python bindings & a few functions added to help bridge the debugger with `pokemon_ai_for_rl`.
 - PettingZoo environnement for Multi-Agent Reinforcement Learning
 - Quantization
 - Neural Network graph manipulation 
 - Export of the network onto a gba flash card, effectively replacing the AI of the OG game with your AI.

## Project structure
```
pkmn_rl_arena/
├──  agbcc                              # Library allowing to compile gba game with cc compiler
├──  example                            # TO DEFINE / TO ORDER
├── rustboyadvance-ng-for-rl/           # Rust GBA emulator with Python bindings
├── pokeemerald_ai_rl/                  # Custom Pokémon Emerald ROM modified for RL training & build scripts
├── data/                               # Data files (CSV, etc.)
├── pkmn_rl_arena/                      # Main Python source code
│     ├── data/ 
│     ├── env/                          # All usefull files to train the model
│     ├── export/                       # To export an onnx in c 
│     │   ├── exporters/
│     │   │   └── layers/
│     │   └── templates/
│     └── quantize/                     # Quantize the model for the GBA
│         ├── __init__.py
│         └── quantize.py
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
└── .python-version
```
- **rustboyadvance-ng-for-rl/**: Rust-based GBA emulator with Python bindings (PyO3)
- **pokeemerald_ai_rl/**: Custom Pokémon Emerald ROM and build instructions
- **pkmn_rl_arena/**: Main Python code (environment, data, quantization, export)
- **data/**: Data files (CSV, etc.)
- **tests/**: Unit and integration tests 

##  Installation 
Requirements :
 - [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) required to build rust code, to do so, you must install [rust](https://www.rust-lang.org/)
 ```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
 ```
 - [uv](https://docs.astral.sh/uv/getting-started/installation/)  Facultative but highly recommended
 ```bash
 curl -LsSf https://astral.sh/uv/install.sh | sh
 ```

### Linux 
1. **Clone the repository & its submodules**
```bash
git clone https://github.com/wissammm/rl_new_pokemon_ai.git  --recurse-submodule 
```
2. **Set up a Python virtual environment**
```bash
uv venv # use python venv if uv is not installed
source .venv/bin/activate
```
3. **Install Python dependencies**
```sh
uv pip compile pyproject.toml 
```

#### Install dependencies

SDL2 : 
```bash
sudo apt install libsdl2-dev libsdl2-image-dev
```

#### Compile pokemon_ai_for_rl
To compile the custom pokeemerald rom, go to see directly the [pokeemerald_ai_rl's INSTALL.md](./pokeemerald_ai_rl/INSTALL.md).
> [!WARNING]
> 1. In [the build section](./pokeemerald_ai_rl/INSTALL.md#Build-pokeemerald) in each path replace `pokeemerald` with `pokeemard_ai_for_rl`
> 2. Replace the final `make modern` command by 
> ```bash
> make modern DINFO=1 DOBSERVED_DATA=1 DSKIP_TEXT=1 DSKIP_GRAPHICS=1 NO_DEBUG=1 -j
> ```

#### Install the rust emulator python extension
**Build and install the rust emulator extension**
```
cd rustboyadvance-ng-for-rl/platform/rustboyadvance-py
maturin develop --features elf_support --release -j6
```

#### Build & run the gba debugger with pokemon emerald loaded on it

1. Download a GBA bios & put it in the root of `rustboyadvance-ng-for-rl`. A bios dl link can be found [in the README.md of the project](README).
```bash
./run_rom.sh
```

## License
This project is licensed under the MIT License. NOT FOR "pret/pokeemerald" SCIENTIFIC USE ONLY
See the [LICENSE](LICENSE) file for details.
