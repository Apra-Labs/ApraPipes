#!/bin/bash
# Setup script for ApraPipes CI self-hosted runner (Ubuntu 22.04/24.04)
# This script installs all required dependencies for the CI-Linux-CUDA build
# Idempotent - can be re-run safely
#
# Prerequisites:
# - Ubuntu 22.04 or 24.04
# - NVIDIA GPU with drivers installed
# - CUDA 11.8 toolkit installed at /usr/local/cuda-11.8
# - sudo access

set -e

echo "=== ApraPipes Linux CUDA Self-Hosted Runner Setup ==="
echo "This script sets up a Linux x64 self-hosted runner with CUDA 11.8 support"
echo "Idempotent - can be re-run safely"
echo ""

# --- Docker ---
echo "[1/8] Setting up Docker..."
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
echo "[2/8] Setting up NVIDIA Container Toolkit..."
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

# --- Build Dependencies ---
echo "[3/8] Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
    ca-certificates curl zip unzip tar \
    autoconf autoconf-archive automake build-essential gcc g++ make ninja-build \
    gcc-11 g++-11 \
    git-core git-lfs \
    cmake pkg-config \
    libass-dev libfreetype6-dev libvorbis-dev libmp3lame-dev libsdl2-dev \
    libva-dev libvdpau-dev \
    flex bison nasm yasm zlib1g-dev gperf \
    libx11-dev libgles2-mesa-dev libglu1-mesa-dev xorg-dev \
    libxinerama-dev libxcursor-dev \
    texinfo wget dos2unix \
    libgnutls28-dev libtool libssl-dev \
    python3-jinja2 python3-pip python3-dev

# libxcb packages (expanded from libxcb*-dev)
sudo apt-get install -y \
    libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev libxcb-render0-dev \
    libxcb-shape0-dev libxcb-xkb-dev libxcb-icccm4-dev libxcb-image0-dev \
    libxcb-keysyms1-dev libxcb-randr0-dev libxcb-util0-dev libxcb-cursor-dev || true

# libncurses packages
sudo apt-get install -y libncurses-dev libncursesw5-dev || true

# libsoup (may vary by Ubuntu version)
sudo apt-get install -y libsoup2.4-dev libsoup-gnome2.4-dev || sudo apt-get install -y libsoup-3.0-dev || true

# --- CUDNN for CUDA 11.8 ---
echo "[4/8] Installing CUDNN for CUDA 11.8..."
if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm /tmp/cuda-keyring.deb
    sudo apt-get update
fi

# Pin to CUDA 11.8 compatible version
if ! dpkg -l | grep -q "libcudnn8-dev.*cuda11.8"; then
    sudo apt-get install -y libcudnn8=8.9.7.29-1+cuda11.8 libcudnn8-dev=8.9.7.29-1+cuda11.8 || true
    # Hold the packages to prevent accidental upgrade to CUDA 12.x versions
    sudo apt-mark hold libcudnn8 libcudnn8-dev || true
else
    echo "  CUDNN 11.8 already installed"
fi

# --- CUDA Environment ---
echo "[5/8] Setting up CUDA environment..."
if [ ! -f /etc/profile.d/cuda.sh ]; then
    cat << 'CUDA_ENV' | sudo tee /etc/profile.d/cuda.sh
export PATH=/usr/local/cuda/bin:$PATH
export CUDAToolkit_ROOT=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
CUDA_ENV
    sudo chmod +x /etc/profile.d/cuda.sh
    echo "  Created /etc/profile.d/cuda.sh"
else
    echo "  CUDA environment already configured"
fi

# --- PowerShell ---
echo "[6/8] Setting up PowerShell..."
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
echo "[7/8] Setting up GitHub CLI..."
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
echo "[8/8] Verifying installation..."
echo ""
echo "=== Verification Results ==="
echo "Docker:          $(docker --version 2>/dev/null || echo 'NOT FOUND')"
echo "NVIDIA Toolkit:  $(dpkg -l 2>/dev/null | grep nvidia-container-toolkit | awk '{print $3}' || echo 'NOT FOUND')"
echo "PowerShell:      $(pwsh --version 2>/dev/null || echo 'NOT FOUND')"
echo "GitHub CLI:      $(gh --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "Git:             $(git --version 2>/dev/null || echo 'NOT FOUND')"
echo "Git LFS:         $(git lfs version 2>/dev/null || echo 'NOT FOUND')"
echo "CMake:           $(cmake --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "Ninja:           $(ninja --version 2>/dev/null || echo 'NOT FOUND')"
echo "GCC-11:          $(gcc-11 --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "G++-11:          $(g++-11 --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "autoconf:        $(autoconf --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "automake:        $(automake --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "libtoolize:      $(libtoolize --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "CUDNN:           $(dpkg -l 2>/dev/null | grep libcudnn8-dev | awk '{print $3}' || echo 'NOT FOUND')"
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
echo ""
echo "IMPORTANT: CUDA 11.8 requires GCC <= 11. The CI workflow uses GCC-11 via"
echo "environment variables (CC, CXX, CUDAHOSTCXX) set in build-test-lin.yml."
echo "GCC-11 is installed but NOT set as system default to avoid breaking other builds."
echo ""
echo "If you were added to the docker group, please log out and back in."
echo ""
echo "=== Next Steps ==="
echo "1. Install GitHub Actions runner (if not already done):"
echo "   - Go to: https://github.com/Apra-Labs/ApraPipes/settings/actions/runners/new"
echo "   - Download and configure with repo token"
echo "2. Add 'linux-cuda' label to the runner in GitHub settings"
echo "3. Start the runner as a service:"
echo "   cd ~/actions-runner && sudo ./svc.sh install && sudo ./svc.sh start"
