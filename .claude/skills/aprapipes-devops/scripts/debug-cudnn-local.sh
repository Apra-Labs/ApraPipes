#!/bin/bash
# Local Docker debugging for cuDNN vcpkg issue
# Smart DevOps: Test just cudnn package with same environment as CI

set -e

echo "=== Starting nvidia/cuda:11.8.0-devel-ubuntu22.04 container ==="
docker run --rm \
  -v "$(pwd):/workspace" \
  -w /workspace \
  nvidia/cuda:11.8.0-devel-ubuntu22.04 \
  bash -c '
    echo "=== Running same prep-cmd as CI ==="
    apt-get update -qq
    apt-get install -y python3-pip libcudnn8-dev
    pip3 install --upgrade cmake==3.30.1
    apt-get -y install ca-certificates curl zip unzip tar autoconf automake autoconf-archive autopoint build-essential gcc g++ make flex git-core git-lfs libass-dev libfreetype6-dev libgnutls28-dev libmp3lame-dev libsdl2-dev libtool libsoup-gnome2.4-dev libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev libncurses5-dev libncursesw5-dev ninja-build pkg-config texinfo wget yasm zlib1g-dev nasm gperf bison dos2unix libx11-dev libgles2-mesa-dev libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev python3-jinja2 libssl-dev
    pip3 install meson
    wget -q https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb
    dpkg -i packages-microsoft-prod.deb
    apt-get update -qq
    apt-get install -y powershell

    echo ""
    echo "=== Checking CUDA/cuDNN installation ==="
    nvcc --version || echo "CUDA not found"
    test -f /usr/include/cudnn.h && echo "cuDNN header found at /usr/include/cudnn.h" || echo "cuDNN header NOT found"
    test -f /usr/lib/x86_64-linux-gnu/libcudnn.so && echo "cuDNN lib found" || echo "cuDNN lib NOT found"
    ls -la /usr/lib/x86_64-linux-gnu/libcudnn* || true
    ls -la /usr/local/cuda/lib64/libcudnn* || true

    echo ""
    echo "=== Bootstrapping vcpkg ==="
    ./vcpkg/bootstrap-vcpkg.sh

    echo ""
    echo "=== Testing JUST cudnn package installation ==="
    echo "Installing cudnn:x64-linux..."
    ./vcpkg/vcpkg install cudnn:x64-linux --debug 2>&1 | tee /tmp/cudnn-install.log

    echo ""
    echo "=== Installation complete! Check /tmp/cudnn-install.log for details ==="
    tail -100 /tmp/cudnn-install.log
  '
