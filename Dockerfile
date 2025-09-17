# Use a recent devkitPro image (recommended)
FROM docker.io/devkitpro/devkitarm:20250728

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEVKITPRO=/opt/devkitpro
ENV PATH=$DEVKITPRO/tools/bin:$PATH

# Install system packages your build needs
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libsdl2-dev \
    libsdl2-image-dev \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    binutils-arm-none-eabi \
    libpng-dev \
    gdebi-core \
    && rm -rf /var/lib/apt/lists/*

# Install RUST / CARGO & UV
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# clone repo
WORKDIR /app
RUN git config --global url."https://github.com/".insteadOf git@github.com:
RUN git clone --recurse-submodules https://github.com/wissammm/PkmnRLArena.git .

# retrieve GBA BIOS
RUN wget -O ./rustboyadvance-ng-for-rl/bios.bin https://raw.githubusercontent.com/Nebuleon/ReGBA/master/bios/gba_bios.bin
RUN mv /app/rustboyadvance-ng-for-rl/bios.bin /app/rustboyadvance-ng-for-rl/gba_bios.bin

COPY pyproject.toml uv.lock /app/
RUN uv venv .venv
ENV PATH="/app/.venv/bin:${PATH}" 
RUN uv sync

WORKDIR /app/rustboyadvance-ng-for-rl/platform/rustboyadvance-py
RUN python -m maturin develop --features elf_support --release -j6

WORKDIR /app

WORKDIR /app/agbcc
RUN ./build.sh
RUN ./install.sh ../pokeemerald_ai_rl

WORKDIR /app/pokeemerald_ai_rl
RUN make modern DINFO=1 DOBSERVED_DATA=1 DSKIP_TEXT=1 DSKIP_GRAPHICS=1 NO_DEBUG_PRINT=1 -j

WORKDIR /app

ENV PYTHONPATH=/app


CMD ["python", "-m", "pkmn_rl_arena"]
