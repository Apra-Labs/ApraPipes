#!/bin/bash

# ============================================================================
# ApraPipes Linux CUDA Build Script with Component Selection
# ============================================================================

# Parse command line arguments
BUILD_DOC=0
COMPONENTS="ALL"
SHOW_HELP=0
PRESET=""

show_help() {
    cat << EOF

ApraPipes Linux CUDA Build Script with Component Selection
===========================================================

Usage: ./build_linux_cuda.sh [OPTIONS]

Options:
  --help, -h            Show this help message
  --build-doc           Build documentation after compilation
  --components "LIST"   Specify components to build (semicolon-separated)
  --preset NAME         Use a preset configuration

Available Presets:
  minimal              CORE only (~5-10 min build)
  video                CORE + VIDEO + IMAGE_PROCESSING (~15-25 min)
  cuda                 CORE + VIDEO + IMAGE_PROCESSING + CUDA_COMPONENT
  full                 All components (default, ~60-90 min)

Available Components:
  CORE                 Pipeline infrastructure (always required)
  VIDEO                Mp4, H264, RTSP
  IMAGE_PROCESSING     OpenCV CPU-based processing
  CUDA_COMPONENT       GPU acceleration
  WEBCAM               Webcam capture
  QR                   QR code reading
  AUDIO                Audio capture and transcription
  FACE_DETECTION       Face detection and landmarks
  GTK_RENDERING        Linux GUI rendering
  THUMBNAIL            Thumbnail generation
  IMAGE_VIEWER         Image viewing GUI

Examples:
  ./build_linux_cuda.sh
  ./build_linux_cuda.sh --preset minimal
  ./build_linux_cuda.sh --preset video
  ./build_linux_cuda.sh --components "CORE;VIDEO;IMAGE_PROCESSING"
  ./build_linux_cuda.sh --components "CORE;CUDA_COMPONENT" --build-doc

EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            ;;
        --build-doc)
            BUILD_DOC=1
            shift
            ;;
        --components)
            COMPONENTS="$2"
            shift 2
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Apply presets
if [ -n "$PRESET" ]; then
    case $PRESET in
        minimal)
            COMPONENTS="CORE"
            ;;
        video)
            COMPONENTS="CORE;VIDEO;IMAGE_PROCESSING"
            ;;
        cuda)
            COMPONENTS="CORE;VIDEO;IMAGE_PROCESSING;CUDA_COMPONENT"
            ;;
        full)
            COMPONENTS="ALL"
            ;;
        *)
            echo "ERROR: Unknown preset '$PRESET'"
            echo "Use --help to see available presets"
            exit 1
            ;;
    esac
fi

echo
echo "============================================================================"
echo "Building ApraPipes with Components: $COMPONENTS"
echo "============================================================================"
echo

sudo apt-get install clang-format
clang-format -style=llvm -dump-config > .clang-format
if ! command -v pip &> /dev/null; then
    # If pip is not available, download and install pip
    curl -O https://bootstrap.pypa.io/get-pip.py
    python3 get-pip.py
fi
pip install pre-commit
pre-commit install

chmod +x build_scripts/build_dependencies_linux_cuda.sh
sudo ./build_scripts/build_dependencies_linux_cuda.sh

if ! sudo nvcc --version &>/dev/null; then
  USER_NAME=$(whoami)
  cudnn_archives="/home/$USER_NAME/Downloads/cudnn-*.tar.xz"

  for archive in $cudnn_archives; do
    if [ -e "$archive" ]; then
      extracted_folder="/home/$USER_NAME/Downloads/$(basename "$archive" .tar.xz)"
      if [ ! -d "$extracted_folder" ]; then
        echo "Extracting $archive..."
        tar xf "$archive" -C "/home/$USER_NAME/Downloads/"
      else
        echo "Archive already extracted: $extracted_folder"
      fi
    fi
  done

  echo "Copying files..."
  sudo cp -r /home/$USER_NAME/Downloads/cudnn-*/include/* /usr/local/cuda/include/
  sudo cp -r /home/$USER_NAME/Downloads/cudnn-*/lib/* /usr/local/cuda/lib64/

  TARGET_USER="$USER_NAME"
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

if [[ $BUILD_DOC -eq 1 ]]; then
    chmod +x build_documentation.sh
    ./build_documentation.sh
fi

cd vcpkg
./bootstrap-vcpkg.sh
vcpkg integrate install
cd ..

CMAKE_THCOUNT=$(sh ./checkProc.sh)
mkdir -p _build
cd _build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_COMPONENTS="$COMPONENTS" ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$CMAKE_THCOUNT"

cd ..

mkdir -p _debugbuild
cd _debugbuild
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_COMPONENTS="$COMPONENTS" ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$CMAKE_THCOUNT"
