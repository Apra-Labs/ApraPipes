#!/bin/bash

# List of required dependencies
dependencies=( "curl" "zip" "unzip" "tar" "autoconf" "automake" "autopoint" "build-essential" 
              "flex" "git-core" "git-lfs" "libass-dev" "libfreetype6-dev" "libgnutls28-dev" "libmp3lame-dev" 
              "libsdl2-dev" "libssl-dev" "libtool" "libsoup-gnome2.4-dev" "libncurses5-dev" "libva-dev" "libvdpau-dev" 
              "libvorbis-dev" "libxcb1-dev" "libxdamage-dev" "libxcursor-dev" "libxinerama-dev" "libx11-dev" "libgles2-mesa-dev" "libxcb-shm0-dev" "libxcb-xfixes0-dev" 
               "ninja-build" "pkg-config" "texinfo" "wget" "yasm" "zlib1g-dev" "nasm" "gperf" "bison" "python3" "python3-pip" "doxygen" "graphviz")

missing_dependencies=()

# Check and collect missing dependencies
for dependency in "${dependencies[@]}"; do
  if ! sudo dpkg -s "$dependency" >/dev/null 2>&1; then
    missing_dependencies+=("$dependency")
  fi
done

# If there are missing dependencies, install them
if [ "${#missing_dependencies[@]}" -gt 0 ]; then
  echo "Installing missing dependencies..."
  sudo apt-get update -qq
  sudo apt-get -y install "${missing_dependencies[@]}"
fi

# Install Meson if not present
if ! sudo meson --version &>/dev/null; then
  echo "meson is not installed. Installing meson..."
  pip3 install meson
fi

# Install Cmake if not present
if ! sudo cmake --version &>/dev/null; then
  echo "CMake is not installed. Installing CMake..."
  pip3 install cmake --upgrade
fi

# Install jq if not present
if ! sudo jq --version &>/dev/null; then
  echo "jq is not installed. Installing jq..."
  sudo apt install jq
fi

if [ ! -d "/usr/local/cuda/include" ] || [ ! -d "/usr/local/cuda/lib64" ]; then
    echo "ERROR: CUDA Toolkit is not properly installed. Please install CUDA Toolkit."
    exit 1
fi

if ! sudo nvcc --version &>/dev/null; then
  userName=$(whoami)
  cudnn_archives="/home/$userName/Downloads/cudnn-*.tar.xz"

  for archive in $cudnn_archives; do
    if [ -e "$archive" ]; then
      extracted_folder="/home/$userName/Downloads/$(basename "$archive" .tar.xz)"
      if [ ! -d "$extracted_folder" ]; then
        echo "Extracting $archive..."
        tar xf "$archive" -C "/home/$userName/Downloads/"
      else
        echo "Archive already extracted: $extracted_folder"
      fi
    fi
  done

  echo "Copying files..."
  sudo cp -r /home/$userName/Downloads/cudnn-*/include/* /usr/local/cuda/include/
  sudo cp -r /home/$userName/Downloads/cudnn-*/lib/* /usr/local/cuda/lib64/

  TARGET_USER="$userName"
  TARGET_HOME=$(eval echo ~$TARGET_USER)

  # Append lines to the target user's ~/.bashrc
  if ! grep -qxF 'export PATH=/usr/local/cuda/bin:${PATH}' $TARGET_HOME/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin:${PATH}' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
    echo "CUDA Binary Path added to .bashrc"
  else
    echo "CUDA Binary Path already exists in .bashrc"
  fi

  if ! grep -qxF 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' $TARGET_HOME/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
    echo "CUDA Library Path added to .bashrc"
  else
    echo "CUDA Library Path already exists in .bashrc"
  fi

  # Reload .bashrc
  source ~/.bashrc

  echo "Appended line to ~/.bashrc and saved changes."
  echo "Reloaded ~/.bashrc"
fi

echo "Dependencies verified and installed successfully."