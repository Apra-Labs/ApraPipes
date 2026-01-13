#!/bin/bash
# ==============================================================================
# CUDA Pipeline Examples Test Script
# ==============================================================================
# Tests all CUDA declarative pipeline examples to verify GPU processing works.
#
# Prerequisites:
#   - CUDA-enabled GPU
#   - aprapipes_cli built with ENABLE_CUDA=ON
#
# Usage:
#   ./scripts/test_cuda_examples.sh [options]
#
# Options:
#   --verbose          Show detailed output
#   --keep-outputs     Don't cleanup output files after tests
#   --example <name>   Test only a specific example (e.g., "01_gaussian_blur")
#   --help             Show this help message
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#   2 - Script error (missing CLI, no CUDA, etc.)
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI_PATH="$PROJECT_ROOT/bin/aprapipes_cli"
EXAMPLES_DIR="$PROJECT_ROOT/examples/cuda"
OUTPUT_DIR="$PROJECT_ROOT/bin/data/testOutput"
RUN_TIMEOUT=30  # seconds timeout for each pipeline

# Options
VERBOSE=false
KEEP_OUTPUTS=false
SPECIFIC_EXAMPLE=""

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_test() {
    echo -e "\n${YELLOW}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_TESTS++))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_TESTS++))
}

print_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED_TESTS++))
}

print_info() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

show_help() {
    head -30 "$0" | tail -25
    exit 0
}

# ==============================================================================
# Argument Parsing
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --keep-outputs)
            KEEP_OUTPUTS=true
            shift
            ;;
        --example)
            SPECIFIC_EXAMPLE="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

# ==============================================================================
# Pre-flight Checks
# ==============================================================================

print_header "CUDA Pipeline Examples Test Suite"

# Check CLI exists
if [[ ! -f "$CLI_PATH" ]]; then
    echo -e "${RED}Error: CLI not found at $CLI_PATH${NC}"
    echo "Please build and install: ./scripts/install_to_bin.sh"
    exit 2
fi

# Check examples directory exists
if [[ ! -d "$EXAMPLES_DIR" ]]; then
    echo -e "${RED}Error: Examples directory not found: $EXAMPLES_DIR${NC}"
    exit 2
fi

# Check CUDA is available
if ! nvidia-smi &>/dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. CUDA may not be available.${NC}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}CLI:${NC} $CLI_PATH"
echo -e "${GREEN}Examples:${NC} $EXAMPLES_DIR"
echo -e "${GREEN}Output:${NC} $OUTPUT_DIR"

# ==============================================================================
# Test Functions
# ==============================================================================

# Run a single CUDA example pipeline
# Args: $1 = example name (e.g., "01_gaussian_blur_demo")
#       $2 = expected output prefix (e.g., "cuda_blur")
#       $3 = expected file count (e.g., 300)
run_example() {
    local example_name="$1"
    local output_prefix="$2"
    local expected_count="$3"
    local json_file="$EXAMPLES_DIR/${example_name}.json"

    ((TOTAL_TESTS++))
    print_test "$example_name"

    # Check if JSON exists
    if [[ ! -f "$json_file" ]]; then
        print_fail "JSON file not found: $json_file"
        return 1
    fi

    # Clean output files for this example
    rm -f "$OUTPUT_DIR/${output_prefix}_"*.jpg 2>/dev/null || true

    # Run the pipeline
    print_info "Running pipeline..."
    local output
    local exit_code=0

    cd "$PROJECT_ROOT/bin"
    output=$(timeout "$RUN_TIMEOUT" "$CLI_PATH" run "$json_file" 2>&1) || exit_code=$?

    # Check for errors
    if echo "$output" | grep -qi "error\|failed\|exception"; then
        if [ "$VERBOSE" = true ]; then
            echo "$output"
        fi
        print_fail "Pipeline reported errors"
        return 1
    fi

    # Count output files
    local file_count
    file_count=$(ls "$OUTPUT_DIR/${output_prefix}_"*.jpg 2>/dev/null | wc -l)

    print_info "Generated $file_count files (expected: $expected_count)"

    # Verify file count
    if [[ "$file_count" -lt "$expected_count" ]]; then
        print_fail "Expected $expected_count files, got $file_count"
        return 1
    fi

    # Check file sizes are reasonable (not empty/black frames)
    local sample_size
    sample_size=$(stat -c%s "$OUTPUT_DIR/${output_prefix}_0001.jpg" 2>/dev/null || echo "0")

    if [[ "$sample_size" -lt 1000 ]]; then
        print_fail "Output files seem too small (possible black frames): $sample_size bytes"
        return 1
    fi

    print_info "Sample file size: $sample_size bytes"
    print_pass "$example_name - $file_count files generated"
    return 0
}

# ==============================================================================
# Main Test Execution
# ==============================================================================

print_header "Running CUDA Examples"

# Define examples: name, output_prefix, expected_count
declare -A EXAMPLES=(
    ["gaussian_blur"]="cuda_blur:100"
    ["auto_bridge"]="cuda_auto:100"
    ["effects"]="cuda_effects:100"
    ["resize"]="cuda_resize:100"
    ["rotate"]="cuda_rotate:100"
    ["processing_chain"]="cuda_chain:100"
    ["nvjpeg_encoder"]="cuda_nvjpeg:100"
)

for example in "${!EXAMPLES[@]}"; do
    # Skip if specific example requested and this isn't it
    if [[ -n "$SPECIFIC_EXAMPLE" && "$example" != *"$SPECIFIC_EXAMPLE"* ]]; then
        continue
    fi

    IFS=':' read -r prefix count <<< "${EXAMPLES[$example]}"
    run_example "$example" "$prefix" "$count" || true
done

# ==============================================================================
# Cleanup and Summary
# ==============================================================================

if [ "$KEEP_OUTPUTS" = false ]; then
    print_info "Cleaning up output files..."
    rm -f "$OUTPUT_DIR/cuda_"*.jpg 2>/dev/null || true
fi

print_header "Test Summary"
echo -e "Total:   $TOTAL_TESTS"
echo -e "${GREEN}Passed:  $PASSED_TESTS${NC}"
echo -e "${RED}Failed:  $FAILED_TESTS${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED_TESTS${NC}"

if [[ $FAILED_TESTS -gt 0 ]]; then
    echo -e "\n${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
fi
