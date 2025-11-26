#!/bin/bash
#
# ApraPipes Code Coverage Generation Script
# This script builds the project with coverage enabled and generates an HTML coverage report
#
# Usage:
#   ./generate_coverage.sh [clean]
#
# Options:
#   clean    - Perform a clean build (removes existing build directory)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build_coverage"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ApraPipes Code Coverage Generator${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if clean build is requested
if [ "$1" == "clean" ]; then
    echo -e "${YELLOW}Performing clean build...${NC}"
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        echo -e "${GREEN}Build directory cleaned.${NC}"
    fi
fi

# Check for required tools
echo -e "\n${BLUE}Checking for required tools...${NC}"

command -v cmake >/dev/null 2>&1 || { echo -e "${RED}Error: cmake is required but not installed.${NC}" >&2; exit 1; }
command -v lcov >/dev/null 2>&1 || { echo -e "${RED}Error: lcov is required but not installed. Install with: sudo apt-get install lcov${NC}" >&2; exit 1; }
command -v genhtml >/dev/null 2>&1 || { echo -e "${RED}Error: genhtml is required but not installed. Install with: sudo apt-get install lcov${NC}" >&2; exit 1; }
command -v gcov >/dev/null 2>&1 || { echo -e "${RED}Error: gcov is required but not installed.${NC}" >&2; exit 1; }

echo -e "${GREEN}All required tools found.${NC}"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake with coverage
echo -e "\n${BLUE}Configuring CMake with code coverage enabled...${NC}"
cmake -B . \
    -DENABLE_WINDOWS=OFF \
    -DENABLE_LINUX=ON \
    -DENABLE_CUDA=OFF \
    -DCODE_COVERAGE=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
    ../base

# Build the project
echo -e "\n${BLUE}Building project with coverage instrumentation...${NC}"
cmake --build . --config Debug -j $(nproc)

# Run coverage target
echo -e "\n${BLUE}Running tests and generating coverage report...${NC}"
make coverage

# Display summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Coverage Report Generated!${NC}"
echo -e "${GREEN}========================================${NC}"

if [ -f "coverage.info" ]; then
    echo -e "\n${BLUE}Coverage Summary:${NC}"
    lcov --summary coverage.info
fi

if [ -d "coverage" ]; then
    REPORT_PATH="${BUILD_DIR}/coverage/index.html"
    echo -e "\n${GREEN}HTML Report Location:${NC}"
    echo -e "  ${REPORT_PATH}"
    echo -e "\n${YELLOW}Open the report with:${NC}"
    echo -e "  xdg-open ${REPORT_PATH}"
    echo -e "  or"
    echo -e "  firefox ${REPORT_PATH}"
fi

echo -e "\n${GREEN}Done!${NC}\n"
