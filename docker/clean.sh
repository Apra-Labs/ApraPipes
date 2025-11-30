#!/bin/bash
# Script to clean up Docker containers and build artifacts
# Usage: ./clean.sh [all|containers|build]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="${1:-containers}"

case "$MODE" in
  containers)
    echo "Stopping and removing ApraPipes Docker containers..."
    docker stop aprapipes-build-nocuda 2>/dev/null || true
    docker stop aprapipes-build-cuda 2>/dev/null || true
    docker stop aprapipes-local-build 2>/dev/null || true
    docker rm aprapipes-build-nocuda 2>/dev/null || true
    docker rm aprapipes-build-cuda 2>/dev/null || true
    docker rm aprapipes-local-build 2>/dev/null || true
    echo "Containers cleaned up"
    ;;
  build)
    echo "Cleaning build artifacts..."
    rm -rf "$PROJECT_ROOT/build"
    rm -rf "$PROJECT_ROOT/vcpkg/buildtrees"
    rm -rf "$PROJECT_ROOT/vcpkg/packages"
    rm -rf "$PROJECT_ROOT/vcpkg_installed"
    echo "Build artifacts cleaned"
    ;;
  all)
    echo "Cleaning everything..."
    $0 containers
    $0 build
    echo "Removing Docker images..."
    docker rmi aprapipes-builder:nocuda 2>/dev/null || true
    docker rmi aprapipes-builder:cuda 2>/dev/null || true
    echo "Complete cleanup done"
    ;;
  *)
    echo "Usage: $0 [all|containers|build]"
    echo "  containers - Stop and remove Docker containers"
    echo "  build      - Clean build artifacts"
    echo "  all        - Clean everything (containers, build artifacts, images)"
    exit 1
    ;;
esac
