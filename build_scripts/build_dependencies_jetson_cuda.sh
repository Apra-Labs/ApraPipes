#!/bin/bash

# List of required dependencies
dependencies=("git-lfs" "libncurses5-dev" "ninja-build" "nasm" "curl" "libudev-dev" "libssl-dev")

missing_dependencies=()

# Check and collect missing dependencies
for dependency in "${dependencies[@]}"; do
  if ! dpkg -s "$dependency" 2>&1; then
    missing_dependencies+=("$dependency")
  fi
done

# If there are missing dependencies, install them
if [ "${#missing_dependencies[@]}" -gt 0 ]; then
  echo "Installing missing dependencies..."
  apt-get update -qq
  apt-get -y install "${missing_dependencies[@]}"
fi

# Install Cmake if not present
if ! cmake --version; then
  echo "CMake is not installed. Installing CMake..."
  snap install cmake --classic
fi

if [ ! -d "/usr/local/cuda/include" ] || [ ! -d "/usr/local/cuda/lib64" ]; then
  echo "ERROR: CUDA Toolkit is not properly installed. Please install CUDA Toolkit."
  exit 1
fi

if  nvcc --version; then
  TARGET_USER="$SUDO_USER"
  TARGET_HOME=$(eval echo ~$TARGET_USER)
  echo 'export VCPKG_FORCE_SYSTEM_BINARIES=1' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
  echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
  echo "Appended paths to ~/.bashrc and saved changes."
  source ~/.bashrc
  echo "Reloaded ~/.bashrc"
fi

echo "Dependencies verified and installed successfully."
