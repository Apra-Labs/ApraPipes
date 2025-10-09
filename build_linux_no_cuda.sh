#!/bin/bash

# ============================================================================
# ApraPipes Linux No-CUDA Build Script with Component Selection
# ============================================================================

# Parse command line arguments
BUILD_DOC=0
COMPONENTS="ALL"
SHOW_HELP=0
PRESET=""

show_help() {
    cat << EOF

ApraPipes Linux No-CUDA Build Script with Component Selection
==============================================================

Usage: ./build_linux_no_cuda.sh [OPTIONS]

Options:
  --help, -h            Show this help message
  --build-doc           Build documentation after compilation
  --components "LIST"   Specify components to build (semicolon-separated)
  --preset NAME         Use a preset configuration

Available Presets:
  minimal              CORE only (~5-10 min build)
  video                CORE + VIDEO + IMAGE_PROCESSING (~15-25 min)
  full                 All components (default, ~40-60 min)

Available Components (no CUDA/ARM64):
  CORE                 Pipeline infrastructure (always required)
  VIDEO                Mp4, H264, RTSP
  IMAGE_PROCESSING     OpenCV CPU-based processing
  WEBCAM               Webcam capture
  QR                   QR code reading
  AUDIO                Audio capture and transcription
  FACE_DETECTION       Face detection and landmarks
  GTK_RENDERING        Linux GUI rendering
  THUMBNAIL            Thumbnail generation
  IMAGE_VIEWER         Image viewing GUI

Examples:
  ./build_linux_no_cuda.sh
  ./build_linux_no_cuda.sh --preset minimal
  ./build_linux_no_cuda.sh --preset video
  ./build_linux_no_cuda.sh --components "CORE;VIDEO;IMAGE_PROCESSING"

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
echo "Building ApraPipes (No CUDA) with Components: $COMPONENTS"
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

chmod +x build_scripts/build_dependencies_linux_no_cuda.sh
sudo ./build_scripts/build_dependencies_linux_no_cuda.sh

chmod +x base/fix-vcpkg-json.sh
./base/fix-vcpkg-json.sh true false false

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
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_CUDA=OFF -DENABLE_COMPONENTS="$COMPONENTS" ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$CMAKE_THCOUNT"
cd ..

mkdir -p _debugbuild
cd _debugbuild
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=OFF -DENABLE_COMPONENTS="$COMPONENTS" ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$CMAKE_THCOUNT"
