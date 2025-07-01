#!/bin/bash

# List of required dependencies
dependencies=("curl" "zip" "unzip" "tar" "jq" "git-lfs" "bison" "libncurses5-dev" "ninja-build" "nasm" "curl" "libudev-dev" "libssl-dev" "doxygen" "graphviz" "autoconf" "automake" "libtool" "libxinerama-dev" "libxcursor-dev" "xorg-dev" "libglu1-mesa-dev" "pkg-config" "python3-jinja2" "nlohmann-json3-dev"
)

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
  chmod +x build_scripts/updateCmake3_29.sh
  ./build_scripts/updateCmake3_29.sh
fi

if [[ $(cmake --version | grep -oP 'cmake version \K[0-9]+\.[0-9]+') < "3.29" ]]; then
  echo "CMake version is less than 3.29. Updating CMake..."
  chmod +x build_scripts/updateCmake3_29.sh
  ./build_scripts/updateCmake3_29.sh
fi

echo "Dependencies verified and installed successfully."
