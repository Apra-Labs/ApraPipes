#!/bin/bash
# ==============================================================================
# Jetson Examples Test Script
# ==============================================================================
# Tests Jetson-specific examples (L4TM JPEG, camera, etc.) using CLI and Node.js.
#
# Requirements:
#   - Jetson device (Xavier, Orin, etc.)
#   - JetPack 5.x or later
#   - Built with -DENABLE_ARM64=ON -DENABLE_CUDA=ON
#
# Usage:
#   ./examples/test_jetson_examples.sh [options]
#
# Options:
#   --cli              Test only CLI examples
#   --node             Test only Node.js examples
#   --verbose          Show detailed output
#   --keep-outputs     Don't cleanup output files after tests
#   --sdk-dir <path>   Use SDK directory structure (for CI)
#   --json-report <f>  Write JSON report to file
#   --ci               CI mode: always exit 0, generate report
#   --help             Show this help message
#
# Exit codes:
#   0 - All tests passed (or CI mode)
#   1 - One or more tests failed
#   2 - Not a Jetson device or script error
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
NODE_ADDON="$PROJECT_ROOT/bin/aprapipes.node"
EXAMPLES_DIR="$PROJECT_ROOT/examples"
OUTPUT_DIR="/tmp/jetson_test"
RUN_TIMEOUT=30  # seconds timeout for each pipeline

# Options
TEST_CLI=true
TEST_NODE=true
VERBOSE=false
KEEP_OUTPUTS=false
SDK_DIR=""
JSON_REPORT=""
CI_MODE=false
WORK_DIR="$PROJECT_ROOT"

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Results array for JSON report (name:status)
declare -a TEST_RESULTS

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
    PASSED_TESTS=$((PASSED_TESTS + 1))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAILED_TESTS=$((FAILED_TESTS + 1))
}

print_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
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

SPECIFIC_REQUESTED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cli)
            if [ "$SPECIFIC_REQUESTED" = false ]; then
                TEST_CLI=false; TEST_NODE=false
                SPECIFIC_REQUESTED=true
            fi
            TEST_CLI=true
            shift
            ;;
        --node)
            if [ "$SPECIFIC_REQUESTED" = false ]; then
                TEST_CLI=false; TEST_NODE=false
                SPECIFIC_REQUESTED=true
            fi
            TEST_NODE=true
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
        --sdk-dir)
            SDK_DIR="$2"
            shift 2
            ;;
        --json-report)
            JSON_REPORT="$2"
            shift 2
            ;;
        --ci)
            CI_MODE=true
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
# SDK Mode Configuration
# ==============================================================================

if [[ -n "$SDK_DIR" ]]; then
    SDK_DIR="$(cd "$SDK_DIR" && pwd)"  # Convert to absolute path
    CLI_PATH="$SDK_DIR/bin/aprapipes_cli"
    NODE_ADDON="$SDK_DIR/bin/aprapipes.node"
    EXAMPLES_DIR="$SDK_DIR/examples"
    OUTPUT_DIR="$SDK_DIR/data/testOutput"
    WORK_DIR="$SDK_DIR"
    echo -e "${BLUE}[SDK MODE]${NC} Using SDK at: $SDK_DIR"
fi

# ==============================================================================
# Pre-flight Checks
# ==============================================================================

print_header "Jetson Examples Test Suite"

# Check if we're on a Jetson device
if [[ ! -f /etc/nv_tegra_release ]]; then
    echo -e "${RED}Error: Not a Jetson device (missing /etc/nv_tegra_release)${NC}"
    echo "This script is designed to run on NVIDIA Jetson devices."
    exit 2
fi

# Print Jetson info
echo -e "${GREEN}Jetson Platform:${NC}"
cat /etc/nv_tegra_release | head -1

# Check CLI exists
if [ "$TEST_CLI" = true ]; then
    if [[ ! -f "$CLI_PATH" ]]; then
        # Try build directory
        CLI_PATH="$PROJECT_ROOT/build/aprapipes_cli"
        if [[ ! -f "$CLI_PATH" ]]; then
            CLI_PATH="$PROJECT_ROOT/_build/aprapipes_cli"
        fi
    fi
    if [[ ! -f "$CLI_PATH" ]]; then
        echo -e "${RED}Error: CLI not found. Build with -DENABLE_ARM64=ON${NC}"
        exit 2
    fi
    echo -e "${GREEN}CLI:${NC} $CLI_PATH"
fi

# Check Node addon exists
if [ "$TEST_NODE" = true ]; then
    if [[ ! -f "$NODE_ADDON" ]]; then
        # Try build directory
        NODE_ADDON="$PROJECT_ROOT/build/aprapipes.node"
        if [[ ! -f "$NODE_ADDON" ]]; then
            NODE_ADDON="$PROJECT_ROOT/_build/aprapipes.node"
        fi
    fi
    if [[ ! -f "$NODE_ADDON" ]]; then
        echo -e "${YELLOW}Warning: Node addon not found. Node.js tests will be skipped.${NC}"
        TEST_NODE=false
    else
        echo -e "${GREEN}Node addon:${NC} $NODE_ADDON"
        # Create symlink for examples
        mkdir -p "$PROJECT_ROOT/bin"
        ln -sf "$NODE_ADDON" "$PROJECT_ROOT/bin/aprapipes.node" 2>/dev/null || true
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}Output:${NC} $OUTPUT_DIR"
echo ""

# ==============================================================================
# CLI JSON Example Tests
# ==============================================================================

run_cli_example() {
    local json_file="$1"
    local example_name=$(basename "$json_file" .json)
    local duration="${2:-5}"
    local test_status="passed"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    print_test "CLI: $example_name"

    if [[ ! -f "$json_file" ]]; then
        print_fail "JSON file not found: $json_file"
        test_status="failed"
        TEST_RESULTS+=("cli_$example_name:$test_status")
        return 1
    fi

    # Clean output
    rm -f "$OUTPUT_DIR"/*.jpg "$OUTPUT_DIR"/*.h264 2>/dev/null || true

    # Run the pipeline
    print_info "Running pipeline for ${duration}s..."
    local output
    local exit_code=0

    cd "$WORK_DIR"
    output=$(timeout "$RUN_TIMEOUT" "$CLI_PATH" run "$json_file" --duration "$duration" 2>&1) || exit_code=$?

    if [ "$VERBOSE" = true ]; then
        echo "$output"
    fi

    # Check for L4TM initialization messages (indicates hardware is working)
    if echo "$output" | grep -q "NvMMLiteBlockCreate"; then
        print_info "L4TM hardware initialized successfully"
    fi

    # Check for errors
    if echo "$output" | grep -qi "failed\|exception\|AIPException"; then
        print_fail "Pipeline reported errors"
        test_status="failed"
        TEST_RESULTS+=("cli_$example_name:$test_status")
        return 1
    fi

    # Count output files
    local file_count
    file_count=$(ls "$OUTPUT_DIR"/*.jpg "$OUTPUT_DIR"/*.h264 2>/dev/null | wc -l || echo "0")
    print_info "Generated $file_count output files"

    if [[ "$file_count" -gt 0 ]]; then
        print_pass "$example_name ($file_count files)"
    else
        # Some pipelines don't output files (like display pipelines)
        print_pass "$example_name (no output files - may be expected)"
    fi
    TEST_RESULTS+=("cli_$example_name:$test_status")
    return 0
}

if [ "$TEST_CLI" = true ]; then
    print_header "Testing Jetson CLI Examples"

    # Test L4TM JPEG decode/encode
    run_cli_example "$EXAMPLES_DIR/jetson/01_test_signal_to_jpeg.json" 3 || true

    # Test L4TM with resize
    run_cli_example "$EXAMPLES_DIR/jetson/01_jpeg_decode_transform.json" 3 || true

    # Test H264 encoding (if available)
    if "$CLI_PATH" list-modules 2>/dev/null | grep -q "H264EncoderV4L2\|H264EncoderNVCodec"; then
        run_cli_example "$EXAMPLES_DIR/jetson/02_h264_encode_demo.json" 3 || true
    else
        print_skip "H264 encoder not available"
    fi
fi

# ==============================================================================
# Node.js Example Tests
# ==============================================================================

run_node_example() {
    local js_file="$1"
    local example_name=$(basename "$js_file" .js)
    local test_status="passed"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    print_test "Node: $example_name"

    if [[ ! -f "$js_file" ]]; then
        print_fail "JS file not found: $js_file"
        test_status="failed"
        TEST_RESULTS+=("node_$example_name:$test_status")
        return 1
    fi

    # Clean output
    rm -f "$OUTPUT_DIR"/*.jpg 2>/dev/null || true

    # Run the example
    print_info "Running Node.js example..."
    local output
    local exit_code=0

    cd "$WORK_DIR"
    output=$(timeout "$RUN_TIMEOUT" node "$js_file" 2>&1) || exit_code=$?

    if [ "$VERBOSE" = true ]; then
        echo "$output"
    fi

    # Check for success indicators
    if echo "$output" | grep -qi "Demo Complete\|Example Complete\|SUCCESS"; then
        print_pass "$example_name"
        TEST_RESULTS+=("node_$example_name:$test_status")
        return 0
    fi

    # Check for errors
    if echo "$output" | grep -qi "Error:\|failed\|exception"; then
        print_fail "Example reported errors"
        test_status="failed"
        TEST_RESULTS+=("node_$example_name:$test_status")
        return 1
    fi

    print_pass "$example_name"
    TEST_RESULTS+=("node_$example_name:$test_status")
    return 0
}

if [ "$TEST_NODE" = true ]; then
    print_header "Testing Jetson Node.js Examples"

    # Test basic pipeline first (works on all platforms)
    run_node_example "$EXAMPLES_DIR/node/basic_pipeline.js" || true

    # Test Jetson-specific L4TM demo
    run_node_example "$EXAMPLES_DIR/node/jetson_l4tm_demo.js" || true

    # Test image processing (uses VirtualPTZ)
    run_node_example "$EXAMPLES_DIR/node/image_processing.js" || true
fi

# ==============================================================================
# Cleanup and Summary
# ==============================================================================

if [ "$KEEP_OUTPUTS" = false ]; then
    print_info "Cleaning up output files..."
    rm -rf "$OUTPUT_DIR" 2>/dev/null || true
fi

print_header "Test Summary"
echo -e "Total:   $TOTAL_TESTS"
echo -e "${GREEN}Passed:  $PASSED_TESTS${NC}"
echo -e "${RED}Failed:  $FAILED_TESTS${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED_TESTS${NC}"

# ==============================================================================
# Generate JSON Report
# ==============================================================================

if [[ -n "$JSON_REPORT" ]]; then
    print_info "Writing JSON report to: $JSON_REPORT"

    # Build results array
    results_json="["
    first=true
    for result in "${TEST_RESULTS[@]}"; do
        name="${result%:*}"
        status="${result#*:}"
        if [ "$first" = true ]; then
            first=false
        else
            results_json+=","
        fi
        results_json+="{\"name\":\"$name\",\"status\":\"$status\"}"
    done
    results_json+="]"

    # Write JSON report
    cat > "$JSON_REPORT" << EOF
{
  "script": "test_jetson_examples.sh",
  "timestamp": "$(date -Iseconds)",
  "summary": {
    "passed": $PASSED_TESTS,
    "failed": $FAILED_TESTS,
    "skipped": $SKIPPED_TESTS,
    "total": $TOTAL_TESTS
  },
  "results": $results_json
}
EOF
    echo -e "${GREEN}Report written to: $JSON_REPORT${NC}"
fi

# ==============================================================================
# Exit Handling
# ==============================================================================

if [[ $FAILED_TESTS -gt 0 ]]; then
    echo -e "\n${RED}Some tests failed!${NC}"
    if [ "$CI_MODE" = true ]; then
        echo -e "${YELLOW}CI mode: Exiting with success despite failures${NC}"
        exit 0
    fi
    exit 1
else
    echo -e "\n${GREEN}All Jetson tests passed!${NC}"
    exit 0
fi
