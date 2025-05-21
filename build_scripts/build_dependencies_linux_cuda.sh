#!/bin/bash

# List of required dependencies
dependencies=( "curl" "zip" "unzip" "tar" "autoconf" "automake" "autopoint" "build-essential" 
              "flex" "git-core" "git-lfs" "libass-dev" "libfreetype6-dev" "libgnutls28-dev" "libmp3lame-dev" 
              "libsdl2-dev" "libssl-dev" "libtool" "libsoup-gnome2.4-dev" "libncurses5-dev" "libva-dev" "libvdpau-dev" 
              "libvorbis-dev" "libxcb1-dev" "libxdamage-dev" "libxcursor-dev" "libxinerama-dev" "libx11-dev" "libgles2-mesa-dev" "libxcb-shm0-dev" "libxcb-xfixes0-dev" 
              "ninja-build" "pkg-config" "texinfo" "wget" "yasm" "zlib1g-dev" "nasm" "gperf" "bison" "python3" "python3-pip" "doxygen" "graphviz" "libxi-dev"
              "libgl1-mesa-dev" "libglu1-mesa-dev" "mesa-common-dev" "libxrandr-dev" "libxxf86vm-dev" "libxtst-dev" "libudev-dev" "libgl1-mesa-dev" "python3-jinja2")

missing_dependencies=()

# Check and collect missing dependencies
for dependency in "${dependencies[@]}"; do
  if ! dpkg -s "$dependency" >/dev/null 2>&1; then
    missing_dependencies+=("$dependency")
  fi
done

# If there are missing dependencies, install them
if [ "${#missing_dependencies[@]}" -gt 0 ]; then
  echo "Installing missing dependencies..."
  apt-get update -qq
  apt-get -y install "${missing_dependencies[@]}"
fi

# Install Meson if not present
if ! meson --version &>/dev/null; then
  echo "meson is not installed. Installing meson..."
  pip3 install meson
fi

# Install Cmake if not present
if ! cmake --version &>/dev/null; then
  echo "CMake is not installed. Installing CMake..."
  pip3 install cmake --upgrade
fi

# Install jq if not present
if ! jq --version &>/dev/null; then
  echo "jq is not installed. Installing jq..."
  apt install jq
fi

if [ ! -d "/usr/local/cuda/include" ] || [ ! -d "/usr/local/cuda/lib64" ]; then
    echo "ERROR: CUDA Toolkit is not properly installed. Please install CUDA Toolkit."
    exit 1
fi

echo "Dependencies verified and installed successfully."
