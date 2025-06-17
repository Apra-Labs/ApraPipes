#!/bin/bash

# List of required dependencies
dependencies=("git-lfs" "libncurses5-dev" "ninja-build" "nasm" "curl" "libudev-dev" "libssl-dev" "doxygen" "graphviz")

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

echo "Dependencies verified and installed successfully."
