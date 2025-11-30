#!/bin/bash
# Script to run ApraPipes build inside Docker container
# Usage: ./run.sh [nocuda|cuda]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

FLAVOR="${1:-nocuda}"

case "$FLAVOR" in
  nocuda)
    echo "Starting NoCUDA Docker container..."
    docker run -it --rm \
      -v "$PROJECT_ROOT:/workspace" \
      --name aprapipes-build-nocuda \
      aprapipes-builder:nocuda bash
    ;;
  cuda)
    echo "Starting CUDA Docker container..."
    docker run -it --rm \
      -v "$PROJECT_ROOT:/workspace" \
      --gpus all \
      --name aprapipes-build-cuda \
      aprapipes-builder:cuda bash
    ;;
  *)
    echo "Error: Invalid flavor '$FLAVOR'"
    echo "Usage: $0 [nocuda|cuda]"
    exit 1
    ;;
esac
