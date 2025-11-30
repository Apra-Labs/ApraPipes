#!/bin/bash
# Script to build Docker images for ApraPipes
# Usage: ./build.sh [nocuda|cuda]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

FLAVOR="${1:-nocuda}"

case "$FLAVOR" in
  nocuda)
    echo "Building NoCUDA Docker image..."
    docker build -f "$SCRIPT_DIR/Dockerfile.nocuda" -t aprapipes-builder:nocuda "$PROJECT_ROOT"
    echo "Successfully built aprapipes-builder:nocuda"
    ;;
  cuda)
    echo "Building CUDA Docker image..."
    docker build -f "$SCRIPT_DIR/Dockerfile.cuda" -t aprapipes-builder:cuda "$PROJECT_ROOT"
    echo "Successfully built aprapipes-builder:cuda"
    ;;
  *)
    echo "Error: Invalid flavor '$FLAVOR'"
    echo "Usage: $0 [nocuda|cuda]"
    exit 1
    ;;
esac
