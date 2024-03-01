#!/bin/bash

# List of required dependencies
dependencies=( "curl" "zip" "unzip" "tar" "autoconf" "automake" "autopoint" "build-essential" 
              "flex" "git-core" "git-lfs" "libass-dev" "libfreetype6-dev" "libgnutls28-dev" "libmp3lame-dev" 
              "libsdl2-dev" "libssl-dev" "libtool" "libsoup-gnome2.4-dev" "libncurses5-dev" "libva-dev" "libvdpau-dev" 
              "libvorbis-dev" "libxcb1-dev" "libxdamage-dev" "libxcursor-dev" "libxinerama-dev" "libx11-dev" "libgles2-mesa-dev" "libxcb-shm0-dev" "libxcb-xfixes0-dev" 
               "ninja-build" "pkg-config" "texinfo" "wget" "yasm" "zlib1g-dev" "nasm" "gperf" "bison" "python3" "python3-pip" "doxygen")

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

if ! nvcc --version &>/dev/null; then
  userName=$(whoami)
  cudnn_archives="/home/$userName/Downloads/cudnn-*.tar.xz"

  for archive in $cudnn_archives; do
    if [ -e "$archive" ]; then
      echo "Extracting $archive..."
      tar xf "$archive" -C /home/$userName/Downloads/
    fi
  done

  echo "Copying files..."
  cp -r /home/$userName/Downloads/cudnn-*/include/* /usr/local/cuda/include/
  cp -r /home/$userName/Downloads/cudnn-*/lib/* /usr/local/cuda/lib64/

  TARGET_USER="$SUDO_USER"
  TARGET_HOME=$(eval echo ~$TARGET_USER)

  # Append lines to the target user's ~/.bashrc
  echo 'export PATH=/usr/local/cuda/bin:${PATH}' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc

  # Reload .bashrc
  source $TARGET_HOME/.bashrc

  echo "Appended line to ~/.bashrc and saved changes."
  echo "Reloaded ~/.bashrc"
fi

echo "Dependencies verified and installed successfully."