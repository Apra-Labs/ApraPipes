#!/bin/bash
# Script to build ApraPipes inside a Docker container
# This script should be run inside the Docker container
# Usage: ./build-inside.sh [nocuda|cuda]

set -e

FLAVOR="${1:-nocuda}"

echo "Building ApraPipes with flavor: $FLAVOR"

# Navigate to workspace
cd /workspace

# Bootstrap vcpkg
echo "Bootstrapping vcpkg..."
./vcpkg/bootstrap-vcpkg.sh

# Remove CUDA dependencies if NoCUDA build
if [ "$FLAVOR" = "nocuda" ]; then
  echo "Removing CUDA dependencies from vcpkg.json..."
  cd base
  pwsh ./fix-vcpkg-json.ps1 -removeCUDA
  cd ..
fi

# Clean any previous build artifacts
echo "Cleaning previous build artifacts..."
rm -rf build
mkdir -p build
cd build

# Configure CMake (this will trigger vcpkg to install all dependencies)
echo "Configuring CMake (this will take 60-90 minutes to build all dependencies)..."
if [ "$FLAVOR" = "nocuda" ]; then
  cmake -G Ninja \
        -DENABLE_WINDOWS=OFF \
        -DENABLE_LINUX=ON \
        -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DENABLE_CUDA=OFF \
        ../base
else
  cmake -G Ninja \
        -DENABLE_WINDOWS=OFF \
        -DENABLE_LINUX=ON \
        -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DENABLE_CUDA=ON \
        ../base
fi

# Build
echo "Building ApraPipes..."
NPROC=$(nproc)
cmake --build . -j "$NPROC"

echo "Build complete!"
echo "Binary location: /workspace/build/aprapipesut"
