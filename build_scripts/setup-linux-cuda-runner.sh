#!/bin/bash
set -e

echo "=== ApraPipes Linux CUDA Self-Hosted Runner Setup ==="
echo "This script sets up a Linux x64 self-hosted runner with CUDA support for ApraPipes CI"
echo "This script is idempotent and can be re-run safely"
echo ""

# --- Docker ---
echo "[1/7] Setting up Docker..."
if command -v docker &> /dev/null; then
    echo "  Docker already installed: $(docker --version)"
else
    sudo apt-get update
    sudo apt-get install -y docker.io
fi
sudo systemctl enable docker
sudo systemctl start docker

# Add current user to docker group if not already
if ! groups | grep -q docker; then
    sudo usermod -aG docker "$USER"
    echo "  Added $USER to docker group (re-login required for effect)"
fi

# --- NVIDIA Container Toolkit ---
echo "[2/7] Setting up NVIDIA Container Toolkit..."
if dpkg -l | grep -q nvidia-container-toolkit; then
    echo "  NVIDIA Container Toolkit already installed"
else
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# --- Build Dependencies (prep-cmd packages) ---
echo "[3/7] Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
    ca-certificates curl zip unzip tar \
    autoconf automake build-essential gcc g++ make ninja-build \
    git-core git-lfs \
    libass-dev libfreetype6-dev libvorbis-dev libmp3lame-dev libsdl2-dev \
    libva-dev libvdpau-dev \
    flex bison nasm yasm zlib1g-dev gperf \
    libx11-dev libgles2-mesa-dev libglu1-mesa-dev xorg-dev \
    libxinerama-dev libxcursor-dev \
    texinfo wget dos2unix \
    libgnutls28-dev libtool libssl-dev \
    python3-jinja2

# libxcb packages (expanded from libxcb*-dev)
sudo apt-get install -y \
    libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev libxcb-render0-dev \
    libxcb-shape0-dev libxcb-xkb-dev libxcb-icccm4-dev libxcb-image0-dev \
    libxcb-keysyms1-dev libxcb-randr0-dev libxcb-util0-dev libxcb-cursor-dev || true

# libncurses packages
sudo apt-get install -y libncurses-dev libncursesw5-dev || true

# libsoup (may vary by Ubuntu version)
sudo apt-get install -y libsoup2.4-dev libsoup-gnome2.4-dev || sudo apt-get install -y libsoup-3.0-dev || true

# --- CMake and Python Tools ---
echo "[4/7] Installing CMake and Python tools..."
if command -v cmake &> /dev/null; then
    echo "  CMake already installed: $(cmake --version | head -1)"
else
    sudo apt-get install -y cmake python3-pip python3-dev
fi

# --- PowerShell ---
echo "[5/7] Setting up PowerShell..."
if command -v pwsh &> /dev/null; then
    echo "  PowerShell already installed: $(pwsh --version)"
else
    wget -q "https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb" -O /tmp/packages-microsoft-prod.deb
    sudo dpkg -i /tmp/packages-microsoft-prod.deb
    rm /tmp/packages-microsoft-prod.deb
    sudo apt-get update
    sudo apt-get install -y powershell
fi

# --- GitHub CLI ---
echo "[6/7] Setting up GitHub CLI..."
if command -v gh &> /dev/null; then
    echo "  GitHub CLI already installed: $(gh --version | head -1)"
else
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null
    sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y gh
fi

# --- Verification ---
echo "[7/7] Verifying installation..."
echo ""
echo "=== Verification Results ==="
echo "Docker:          $(docker --version 2>/dev/null || echo 'NOT FOUND')"
echo "NVIDIA Toolkit:  $(dpkg -l | grep nvidia-container-toolkit | awk '{print $3}' || echo 'NOT FOUND')"
echo "PowerShell:      $(pwsh --version 2>/dev/null || echo 'NOT FOUND')"
echo "GitHub CLI:      $(gh --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "Git:             $(git --version 2>/dev/null || echo 'NOT FOUND')"
echo "Git LFS:         $(git lfs version 2>/dev/null || echo 'NOT FOUND')"
echo "CMake:           $(cmake --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "Ninja:           $(ninja --version 2>/dev/null || echo 'NOT FOUND')"
echo "GCC:             $(gcc --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "Python:          $(python3 --version 2>/dev/null || echo 'NOT FOUND')"
echo "Pip:             $(pip3 --version 2>/dev/null || echo 'NOT FOUND')"
echo ""

echo "=== Testing CUDA ==="
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA Driver:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'FAILED')"
    echo "GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'FAILED')"
else
    echo "nvidia-smi: NOT FOUND - CUDA drivers may not be installed"
fi

if [ -f /usr/local/cuda/bin/nvcc ]; then
    echo "CUDA Compiler:   $(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep 'release' || echo 'FAILED')"
else
    echo "nvcc: NOT FOUND - CUDA toolkit may not be installed at /usr/local/cuda"
fi

echo ""
echo "=== Testing Docker GPU Access ==="
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "GPU access in Docker: OK"
else
    echo "GPU access in Docker: FAILED (you may need to re-login or check nvidia-docker setup)"
fi

echo ""
echo "=== Setup Complete ==="
echo "If you were added to the docker group, please log out and back in."
echo ""
echo "=== Next Steps ==="
echo "1. Install GitHub Actions runner (if not already done):"
echo "   - Download: https://github.com/actions/runner/releases"
echo "   - Configure with repo token"
echo "2. Add 'linux-cuda' label to the runner in GitHub settings"
echo "3. Start the runner as a service"
