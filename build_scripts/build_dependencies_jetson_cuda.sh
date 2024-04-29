#!/bin/bash

# List of required dependencies
dependencies=("git-lfs" "libncurses5-dev" "ninja-build" "nasm" "curl" "libudev-dev" "libssl-dev" "doxygen" "graphviz")

missing_dependencies=()

# Check and collect missing dependencies
for dependency in "${dependencies[@]}"; do
  if ! sudo dpkg -s "$dependency" 2>&1; then
    missing_dependencies+=("$dependency")
  fi
done

# If there are missing dependencies, install them
if [ "${#missing_dependencies[@]}" -gt 0 ]; then
  echo "Installing missing dependencies..."
  sudo apt-get update -qq
  sudo apt-get -y install "${missing_dependencies[@]}"
fi

# Install Cmake if not present
if ! sudo cmake --version; then
  echo "CMake is not installed. Installing CMake..."
  sudo snap install cmake --classic
fi

if [ ! -d "/usr/local/cuda/include" ] || [ ! -d "/usr/local/cuda/lib64" ]; then
  echo "ERROR: CUDA Toolkit is not properly installed. Please install CUDA Toolkit."
  exit 1
fi

if nvcc --version; then
  USER_NAME=$(whoami)
  TARGET_USER="$USER_NAME"
  TARGET_HOME=$(eval echo ~$TARGET_USER)

  # Append lines to the target user's ~/.bashrc
  if ! grep -qxF 'export VCPKG_FORCE_SYSTEM_BINARIES=1' $TARGET_HOME/.bashrc; then
    echo 'export VCPKG_FORCE_SYSTEM_BINARIES=1' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
    echo "VCPKG_FORCE_SYSTEM_BINARIES flag added in .bashrc"
  else
    echo "VCPKG_FORCE_SYSTEM_BINARIES flag already exists in .bashrc"
  fi

  if ! grep -qxF 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' $TARGET_HOME/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
    echo "CUDA Binary Path added to .bashrc"
  else
    echo "CUDA Binary Path already exists in .bashrc"
  fi

  if ! grep -qxF 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' $TARGET_HOME/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
    echo "CUDA Library Path added to .bashrc"
  else
    echo "CUDA Library Path already exists in .bashrc"
  fi
  
  echo "Appended paths to ~/.bashrc and saved changes."
  source ~/.bashrc
  echo "Reloaded ~/.bashrc"
fi

echo "Dependencies verified and installed successfully."
