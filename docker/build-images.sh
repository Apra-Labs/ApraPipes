#!/usr/bin/env bash
# Convenience wrapper to build the ApraPipes CUDA images.
# Run from the repo root:  ./docker/build-images.sh [x64|jetson|all]
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root (build context)

TARGET="${1:-x64}"

build_x64() {
  echo ">> Building x86_64 + CUDA 11.8 image (aprapipes:x64-cuda11.8)"
  docker build \
    -f docker/Dockerfile.x64-cuda \
    -t aprapipes:x64-cuda11.8 \
    --build-arg CUDA_ARCH="${CUDA_ARCH:-52;60;70;75}" \
    .
}

build_jetson() {
  echo ">> Building Jetson (ARM64/L4T) CUDA image (aprapipes:jetson-cuda)"
  # ARM64 on an x86 host needs qemu: `docker run --privileged --rm tonistiigi/binfmt --install arm64`
  docker buildx build \
    --platform linux/arm64 \
    -f docker/Dockerfile.jetson-cuda \
    -t aprapipes:jetson-cuda \
    --build-arg CUDA_ARCH="${CUDA_ARCH:-72;87}" \
    --load \
    .
}

case "$TARGET" in
  x64)    build_x64 ;;
  jetson) build_jetson ;;
  all)    build_x64; build_jetson ;;
  *) echo "usage: $0 [x64|jetson|all]"; exit 1 ;;
esac

echo ">> Done."
