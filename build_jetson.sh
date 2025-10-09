#!/bin/bash

# ============================================================================
# ApraPipes Jetson/ARM64 Build Script with Component Selection
# ============================================================================

# Parse command line arguments
BUILD_DOC=0
COMPONENTS="ALL"
SHOW_HELP=0
PRESET=""

show_help() {
    cat << EOF

ApraPipes Jetson/ARM64 Build Script with Component Selection
=============================================================

Usage: ./build_jetson.sh [OPTIONS]

Options:
  --help, -h            Show this help message
  --build-doc           Build documentation after compilation
  --components "LIST"   Specify components to build (semicolon-separated)
  --preset NAME         Use a preset configuration

Available Presets:
  minimal              CORE only (~5-10 min build)
  video                CORE + VIDEO + IMAGE_PROCESSING (~15-25 min)
  jetson               CORE + VIDEO + IMAGE_PROCESSING + CUDA_COMPONENT + ARM64_COMPONENT
  full                 All components (default, ~60-90 min)

Available Components (Jetson/ARM64):
  CORE                 Pipeline infrastructure (always required)
  VIDEO                Mp4, H264, RTSP
  IMAGE_PROCESSING     OpenCV CPU-based processing
  CUDA_COMPONENT       GPU acceleration
  ARM64_COMPONENT      Jetson-specific modules (V4L2, NvArgus, L4TM)
  WEBCAM               Webcam capture
  QR                   QR code reading
  AUDIO                Audio capture and transcription
  FACE_DETECTION       Face detection and landmarks
  GTK_RENDERING        Linux GUI rendering
  THUMBNAIL            Thumbnail generation
  IMAGE_VIEWER         Image viewing GUI

Examples:
  ./build_jetson.sh
  ./build_jetson.sh --preset minimal
  ./build_jetson.sh --preset jetson
  ./build_jetson.sh --components "CORE;VIDEO;IMAGE_PROCESSING;ARM64_COMPONENT"

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
        jetson)
            COMPONENTS="CORE;VIDEO;IMAGE_PROCESSING;CUDA_COMPONENT;ARM64_COMPONENT"
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
echo "Building ApraPipes (Jetson/ARM64) with Components: $COMPONENTS"
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

chmod +x build_scripts/build_dependencies_jetson_cuda.sh
sudo ./build_scripts/build_dependencies_jetson_cuda.sh

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

if [[ $BUILD_DOC -eq 1 ]]; then
    chmod +x build_documentation.sh
    ./build_documentation.sh
fi

cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
cd ..

CMAKE_THCOUNT=$(sh ./checkProc.sh)
mkdir -p _build
cd _build
export VCPKG_FORCE_SYSTEM_BINARIES=1 && export VCPKG_OVERLAY_PORTS=../thirdparty/custom-overlay && cmake -B . -DENABLE_ARM64=ON -DENABLE_WINDOWS=OFF -DENABLE_COMPONENTS="$COMPONENTS" -DCMAKE_BUILD_TYPE=RelWithDebInfo ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$(($(nproc) - 1))"
