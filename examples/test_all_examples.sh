#!/bin/bash
# ==============================================================================
# Unified Examples Test Script
# ==============================================================================
# Tests all declarative pipeline examples (basic, cuda, advanced).
#
# Usage:
#   ./scripts/test_all_examples.sh [options]
#
# Options:
#   --basic            Test only basic (CPU) examples
#   --cuda             Test only CUDA (GPU) examples
#   --advanced         Test only advanced examples
#   --verbose          Show detailed output
#   --keep-outputs     Don't cleanup output files after tests
#   --help             Show this help message
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#   2 - Script error (missing CLI, etc.)
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
EXAMPLES_DIR="$PROJECT_ROOT/examples"
OUTPUT_DIR="$PROJECT_ROOT/bin/data/testOutput"
RUN_TIMEOUT=30  # seconds timeout for each pipeline

# Options
TEST_BASIC=true
TEST_CUDA=true
TEST_ADVANCED=true
VERBOSE=false
KEEP_OUTPUTS=false

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
    head -25 "$0" | tail -20
    exit 0
}

# ==============================================================================
# Argument Parsing
# ==============================================================================

# Reset test flags if specific category requested
SPECIFIC_REQUESTED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --basic)
            if [ "$SPECIFIC_REQUESTED" = false ]; then
                TEST_BASIC=false; TEST_CUDA=false; TEST_ADVANCED=false
                SPECIFIC_REQUESTED=true
            fi
            TEST_BASIC=true
            shift
            ;;
        --cuda)
            if [ "$SPECIFIC_REQUESTED" = false ]; then
                TEST_BASIC=false; TEST_CUDA=false; TEST_ADVANCED=false
                SPECIFIC_REQUESTED=true
            fi
            TEST_CUDA=true
            shift
            ;;
        --advanced)
            if [ "$SPECIFIC_REQUESTED" = false ]; then
                TEST_BASIC=false; TEST_CUDA=false; TEST_ADVANCED=false
                SPECIFIC_REQUESTED=true
            fi
            TEST_ADVANCED=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --keep-outputs)
            KEEP_OUTPUTS=true
            shift
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

print_header "ApraPipes Examples Test Suite"

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

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}CLI:${NC} $CLI_PATH"
echo -e "${GREEN}Examples:${NC} $EXAMPLES_DIR"
echo -e "${GREEN}Output:${NC} $OUTPUT_DIR"
echo ""
echo "Test categories: Basic=$TEST_BASIC, CUDA=$TEST_CUDA, Advanced=$TEST_ADVANCED"

# ==============================================================================
# Test Functions
# ==============================================================================

# Run a single JSON pipeline example
# Args: $1 = json file path
#       $2 = expected output prefix (optional)
#       $3 = expected file count (optional, default 100)
run_json_example() {
    local json_file="$1"
    local output_prefix="$2"
    local expected_count="${3:-100}"
    local example_name=$(basename "$json_file" .json)

    ((TOTAL_TESTS++))
    print_test "$example_name"

    # Check if JSON exists
    if [[ ! -f "$json_file" ]]; then
        print_fail "JSON file not found: $json_file"
        return 1
    fi

    # Clean output files for this example if prefix specified
    if [[ -n "$output_prefix" ]]; then
        rm -f "$OUTPUT_DIR/${output_prefix}_"*.jpg "$OUTPUT_DIR/${output_prefix}_"*.bmp "$OUTPUT_DIR/${output_prefix}_"*.raw 2>/dev/null || true
    fi

    # Run the pipeline
    print_info "Running pipeline..."
    local output
    local exit_code=0

    cd "$PROJECT_ROOT/bin"
    output=$(timeout "$RUN_TIMEOUT" "$CLI_PATH" run "$json_file" 2>&1) || exit_code=$?

    # Check for critical errors (ignore warnings)
    if echo "$output" | grep -qi "failed\|exception\|AIPException"; then
        if echo "$output" | grep -qi "not found\|Unknown module"; then
            print_skip "Module not available: $example_name"
            ((PASSED_TESTS--))  # Undo the increment from print_skip
            ((SKIPPED_TESTS++))
            return 0
        fi
        if [ "$VERBOSE" = true ]; then
            echo "$output"
        fi
        print_fail "Pipeline reported errors"
        return 1
    fi

    # If output prefix specified, verify files were created
    if [[ -n "$output_prefix" ]]; then
        local file_count
        file_count=$(ls "$OUTPUT_DIR/${output_prefix}_"*.jpg "$OUTPUT_DIR/${output_prefix}_"*.bmp "$OUTPUT_DIR/${output_prefix}_"*.raw 2>/dev/null | wc -l)

        print_info "Generated $file_count files (expected: $expected_count)"

        if [[ "$file_count" -lt "$expected_count" ]]; then
            print_fail "Expected $expected_count files, got $file_count"
            return 1
        fi
    fi

    print_pass "$example_name"
    return 0
}

# ==============================================================================
# Basic Examples Tests
# ==============================================================================

if [ "$TEST_BASIC" = true ]; then
    print_header "Testing Basic (CPU) Examples"

    # Examples with expected output
    run_json_example "$EXAMPLES_DIR/basic/simple_source_sink.json" "" 0 || true
    run_json_example "$EXAMPLES_DIR/basic/three_module_chain.json" "" 0 || true
    run_json_example "$EXAMPLES_DIR/basic/split_pipeline.json" "" 0 || true
    run_json_example "$EXAMPLES_DIR/basic/bmp_converter_pipeline.json" "bmp" 100 || true
    run_json_example "$EXAMPLES_DIR/basic/affine_transform_demo.json" "affine" 100 || true
    run_json_example "$EXAMPLES_DIR/basic/affine_transform_chain.json" "" 0 || true  # Uses StatSink, no file output
    run_json_example "$EXAMPLES_DIR/basic/ptz_with_conversion.json" "" 0 || true
    run_json_example "$EXAMPLES_DIR/basic/transform_ptz_with_conversion.json" "" 0 || true
fi

# ==============================================================================
# CUDA Examples Tests
# ==============================================================================

if [ "$TEST_CUDA" = true ]; then
    print_header "Testing CUDA (GPU) Examples"

    # Check CUDA availability
    if ! nvidia-smi &>/dev/null; then
        echo -e "${YELLOW}Warning: nvidia-smi not found. CUDA tests may fail.${NC}"
    fi

    run_json_example "$EXAMPLES_DIR/cuda/gaussian_blur.json" "cuda_blur" 100 || true
    run_json_example "$EXAMPLES_DIR/cuda/auto_bridge.json" "cuda_auto" 100 || true
    run_json_example "$EXAMPLES_DIR/cuda/effects.json" "cuda_effects" 100 || true
    run_json_example "$EXAMPLES_DIR/cuda/resize.json" "cuda_resize" 100 || true
    run_json_example "$EXAMPLES_DIR/cuda/rotate.json" "cuda_rotate" 100 || true
    run_json_example "$EXAMPLES_DIR/cuda/processing_chain.json" "cuda_chain" 100 || true
    run_json_example "$EXAMPLES_DIR/cuda/nvjpeg_encoder.json" "cuda_nvjpeg" 100 || true
fi

# ==============================================================================
# Advanced Examples Tests
# ==============================================================================

if [ "$TEST_ADVANCED" = true ]; then
    print_header "Testing Advanced Examples"

    # These are templates that may need additional setup
    run_json_example "$EXAMPLES_DIR/advanced/file_reader_writer.json" "" 0 || true
    run_json_example "$EXAMPLES_DIR/advanced/mp4_reader_writer.json" "" 0 || true
    run_json_example "$EXAMPLES_DIR/advanced/motion_vector_pipeline.json" "" 0 || true
    run_json_example "$EXAMPLES_DIR/advanced/multimedia_queue_pipeline.json" "" 0 || true
    run_json_example "$EXAMPLES_DIR/advanced/affine_transform_pipeline.json" "" 0 || true
fi

# ==============================================================================
# Cleanup and Summary
# ==============================================================================

if [ "$KEEP_OUTPUTS" = false ]; then
    print_info "Cleaning up output files..."
    rm -f "$OUTPUT_DIR"/*.jpg "$OUTPUT_DIR"/*.bmp "$OUTPUT_DIR"/*.raw 2>/dev/null || true
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
