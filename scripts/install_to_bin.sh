#!/bin/bash
# ==============================================================================
# Install Build Outputs to bin/
# ==============================================================================
# Copies built executables and libraries to the bin/ directory for consistent
# access by test scripts and examples.
#
# Usage:
#   ./scripts/install_to_bin.sh [BUILD_DIR]
#
# Arguments:
#   BUILD_DIR    Build directory (default: _build)
#
# This script copies:
#   - aprapipes_cli       -> bin/aprapipes_cli
#   - aprapipes.node      -> bin/aprapipes.node (if exists)
#   - data/               -> bin/data/ (symlink for test assets)
# ==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${1:-_build}"
BUILD_PATH="$PROJECT_ROOT/$BUILD_DIR"
BIN_DIR="$PROJECT_ROOT/bin"

echo -e "${GREEN}=== Installing to bin/ ===${NC}"
echo "Build directory: $BUILD_PATH"
echo "Target directory: $BIN_DIR"
echo ""

# Check build directory exists
if [[ ! -d "$BUILD_PATH" ]]; then
    echo -e "${RED}Error: Build directory not found: $BUILD_PATH${NC}"
    echo "Please build first: cmake -B $BUILD_DIR && cmake --build $BUILD_DIR"
    exit 1
fi

# Create bin directory
mkdir -p "$BIN_DIR"

# Copy CLI
CLI_SRC="$BUILD_PATH/aprapipes_cli"
if [[ -f "$CLI_SRC" ]]; then
    cp "$CLI_SRC" "$BIN_DIR/"
    echo -e "${GREEN}[OK]${NC} aprapipes_cli"
else
    echo -e "${YELLOW}[SKIP]${NC} aprapipes_cli not found"
fi

# Copy Node.js addon (may not exist if not built)
NODE_SRC="$BUILD_PATH/aprapipes.node"
if [[ -f "$NODE_SRC" ]]; then
    cp "$NODE_SRC" "$BIN_DIR/"
    echo -e "${GREEN}[OK]${NC} aprapipes.node"
else
    # Also check for alternative names/locations
    NODE_SRC_ALT="$BUILD_PATH/aprapipes_node.node"
    if [[ -f "$NODE_SRC_ALT" ]]; then
        cp "$NODE_SRC_ALT" "$BIN_DIR/aprapipes.node"
        echo -e "${GREEN}[OK]${NC} aprapipes.node (from aprapipes_node.node)"
    else
        echo -e "${YELLOW}[SKIP]${NC} aprapipes.node not found (Node.js addon not built?)"
    fi
fi

# Create data directory structure
mkdir -p "$BIN_DIR/data/testOutput"

# Symlink assets if they exist
ASSETS_SRC="$PROJECT_ROOT/data/assets"
ASSETS_DST="$BIN_DIR/data/assets"
if [[ -d "$ASSETS_SRC" && ! -e "$ASSETS_DST" ]]; then
    ln -s "$ASSETS_SRC" "$ASSETS_DST"
    echo -e "${GREEN}[OK]${NC} data/assets (symlinked)"
elif [[ -e "$ASSETS_DST" ]]; then
    echo -e "${GREEN}[OK]${NC} data/assets (already exists)"
else
    echo -e "${YELLOW}[SKIP]${NC} data/assets not found"
fi

# Copy test input files if they exist
for file in "$PROJECT_ROOT/data/"*.jpg "$PROJECT_ROOT/data/"*.png "$PROJECT_ROOT/data/"*.mp4; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        if [[ ! -f "$BIN_DIR/data/$filename" ]]; then
            cp "$file" "$BIN_DIR/data/"
            echo -e "${GREEN}[OK]${NC} data/$filename"
        fi
    fi
done

echo ""
echo -e "${GREEN}=== Installation complete ===${NC}"
echo ""

# Show installed files
echo "Installed files:"
ls -la "$BIN_DIR" | grep -v "^d" | grep -v "^total" | awk '{print "  " $NF}'

# Show version info if possible
if [[ -x "$BIN_DIR/aprapipes_cli" ]]; then
    echo ""
    echo "CLI version:"
    "$BIN_DIR/aprapipes_cli" --version 2>/dev/null || echo "  (version command not available)"
fi
