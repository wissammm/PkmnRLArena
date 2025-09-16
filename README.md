# PkmnRLArena

Reinforcement learning environment and tools for Pokémon Emerald : 
    - Modified Emulator to have python librairy 
    - PettingZoo environnement for MARL
    - Quantization 
    - Neural Network graph manipulation 
    - Export 


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
### Docker (recommended)
Using Docker ensures a fully reproducible environment without needing to manually install system dependencies.

> [!NOTE]  
> You need to have [Docker](https://docs.docker.com/get-docker/) installed on your system before continuing.

```bash
git clone --recurse-submodules https://github.com/wissammm/PkmnRLArena.git
cd PkmnRLArena
docker build -t pokemon-rl .
```

Then run docker : 
```bash
docker run -it --rm pokemon-rl /bin/bash
```

### Debian/Ubuntu


**Clone the repository & its submodules**
```bash
git clone https://github.com/wissammm/rl_new_pokemon_ai.git  --recurse-submodule 
```

1. Install system packages
```bash
sudo apt update
sudo apt install -y build-essential curl git libsdl2-dev libsdl2-image-dev \
  python3 python3-pip python3-venv wget binutils-arm-none-eabi libpng-dev gdebi-core
```

2. Install Rust (needed for the Rust emulator bindings)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# then reopen shell or `source $HOME/.cargo/env`
```

3. (Recommended) Install uv for deterministic Python installs
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# reopen shell or `export PATH="$HOME/.local/bin:$PATH"`
```

4. devkitPro / devkitARM
- Recommended: use the devkitpro/devkitarm Docker image for reproducible builds.
- To install locally on Debian/Ubuntu, run the official installer script:
```bash
wget https://apt.devkitpro.org/install-devkitpro-pacman
chmod +x ./install-devkitpro-pacman
sudo ./install-devkitpro-pacman
# then install gba toolchain
sudo dkp-pacman -Sy --noconfirm gba-dev
# source environment variables
source /etc/profile.d/devkit-env.sh
```

7. Download GBA BIOS to the emulator folder
```bash
wget -O rustboyadvance-ng-for-rl/gba_bios.bin \
  https://raw.githubusercontent.com/Nebuleon/ReGBA/master/bios/gba_bios.bin
```

8. Create venv and install Python dependencies (uses uv.lock if available)
```bash
uv venv .venv        # or: python3 -m venv .venv
source .venv/bin/activate
uv sync              # installs packages from uv.lock / pyproject.toml
```

9. Build & install the Rust emulator Python extension
```bash
cd rustboyadvance-ng-for-rl/platform/rustboyadvance-py
maturin develop --features elf_support --release -j6
cd ../../../
```

10. Build agbcc and install into the pokeemerald build tree
```bash
cd agbcc
./build.sh
./install.sh ../pokeemerald_ai_rl
cd ..
```

11. Build the custom pokeemerald ROM
```bash
cd pokeemerald_ai_rl
make modern DINFO=1 DOBSERVED_DATA=1 DSKIP_TEXT=1 DSKIP_GRAPHICS=1 NO_DEBUG_PRINT=1 -j$(nproc)
cd ..
```

#### Build & run the gba debugger with pokemon emerald loaded on it

1. Download a GBA bios & put it in the root of `rustboyadvance-ng-for-rl`.
```bash
./run_rom.sh
```

## License
This project is licensed under the MIT License. NOT FOR "pret/pokeemerald" SCIENTIFIC USE ONLY
See the [LICENSE](LICENSE) file for details.
