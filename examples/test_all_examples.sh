#!/bin/bash
# ==============================================================================
# Unified Examples Test Script
# ==============================================================================
# Tests all declarative pipeline examples (basic, cuda, advanced, node).
#
# Usage:
#   ./examples/test_all_examples.sh [options]
#
# Options:
#   --basic            Test only basic (CPU) examples
#   --cuda             Test only CUDA (GPU) examples
#   --advanced         Test only advanced examples
#   --node             Test only Node.js addon examples
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
#   2 - Script error (missing CLI, missing Node.js, etc.)
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
OUTPUT_DIR="$PROJECT_ROOT/data/testOutput"
WORK_DIR="$PROJECT_ROOT"  # Directory to run CLI from (for relative paths in JSON)
RUN_TIMEOUT=30  # seconds timeout for each pipeline

# Options
TEST_BASIC=true
TEST_CUDA=true
TEST_ADVANCED=true
TEST_NODE=true
VERBOSE=false
KEEP_OUTPUTS=false
SDK_DIR=""
JSON_REPORT=""
CI_MODE=false

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

# Portable timeout function (works on Linux, macOS, and Windows Git Bash)
run_with_timeout() {
    local timeout_sec=$1
    shift
    local cmd=("$@")

    # Try GNU timeout (Linux) - check it's actually GNU timeout, not Windows timeout
    # GNU timeout supports --version, Windows timeout does not
    if command -v timeout &>/dev/null && timeout --version &>/dev/null 2>&1; then
        timeout "$timeout_sec" "${cmd[@]}"
        return $?
    fi

    # Try gtimeout (macOS with coreutils)
    if command -v gtimeout &>/dev/null; then
        gtimeout "$timeout_sec" "${cmd[@]}"
        return $?
    fi

    # Fallback: Just run without timeout
    # (Background process timeout doesn't capture output properly)
    "${cmd[@]}"
    return $?
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
                TEST_BASIC=false; TEST_CUDA=false; TEST_ADVANCED=false; TEST_NODE=false
                SPECIFIC_REQUESTED=true
            fi
            TEST_BASIC=true
            shift
            ;;
        --cuda)
            if [ "$SPECIFIC_REQUESTED" = false ]; then
                TEST_BASIC=false; TEST_CUDA=false; TEST_ADVANCED=false; TEST_NODE=false
                SPECIFIC_REQUESTED=true
            fi
            TEST_CUDA=true
            shift
            ;;
        --advanced)
            if [ "$SPECIFIC_REQUESTED" = false ]; then
                TEST_BASIC=false; TEST_CUDA=false; TEST_ADVANCED=false; TEST_NODE=false
                SPECIFIC_REQUESTED=true
            fi
            TEST_ADVANCED=true
            shift
            ;;
        --node)
            if [ "$SPECIFIC_REQUESTED" = false ]; then
                TEST_BASIC=false; TEST_CUDA=false; TEST_ADVANCED=false; TEST_NODE=false
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
# In SDK mode, paths are relative to the SDK directory:
#   sdk/bin/aprapipes_cli
#   sdk/examples/basic/*.json
#   sdk/data/frame.jpg (referenced as ./data/frame.jpg in JSON)
#
# We run CLI from SDK root so relative paths in JSON resolve correctly.

if [[ -n "$SDK_DIR" ]]; then
    SDK_DIR="$(cd "$SDK_DIR" && pwd)"  # Convert to absolute path
    CLI_PATH="$SDK_DIR/bin/aprapipes_cli"
    EXAMPLES_DIR="$SDK_DIR/examples"
    OUTPUT_DIR="$SDK_DIR/data/testOutput"
    WORK_DIR="$SDK_DIR"  # Run CLI from SDK root
    echo -e "${BLUE}[SDK MODE]${NC} Using SDK at: $SDK_DIR"

    # Add SDK bin to PATH for Windows (DLL loading requires this)
    export PATH="$SDK_DIR/bin:$PATH"
fi

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
echo "Test categories: Basic=$TEST_BASIC, CUDA=$TEST_CUDA, Advanced=$TEST_ADVANCED, Node=$TEST_NODE"

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
    local test_status="passed"

    cd "$WORK_DIR"
    print_info "CLI: $CLI_PATH"
    print_info "JSON: $json_file"
    print_info "PWD: $(pwd)"
    output=$(run_with_timeout "$RUN_TIMEOUT" "$CLI_PATH" run "$json_file" 2>&1) || exit_code=$?
    print_info "Exit code: $exit_code"

    # Check for critical errors (ignore warnings)
    if echo "$output" | grep -qi "failed\|exception\|AIPException"; then
        if echo "$output" | grep -qi "not found\|Unknown module"; then
            print_skip "Module not available: $example_name"
            test_status="skipped"
            TEST_RESULTS+=("$example_name:$test_status")
            return 0
        fi
        # Always show error output (last few lines for context)
        echo -e "${RED}Error output:${NC}"
        echo "$output" | tail -10
        print_fail "Pipeline reported errors"
        test_status="failed"
        TEST_RESULTS+=("$example_name:$test_status")
        return 1
    fi

    # If output prefix specified, verify files were created
    if [[ -n "$output_prefix" ]]; then
        local file_count
        file_count=$(ls "$OUTPUT_DIR/${output_prefix}_"*.jpg "$OUTPUT_DIR/${output_prefix}_"*.bmp "$OUTPUT_DIR/${output_prefix}_"*.raw 2>/dev/null | wc -l)

        print_info "Generated $file_count files (expected: $expected_count)"

        if [[ "$file_count" -lt "$expected_count" ]]; then
            # Show detailed diagnostics for debugging
            echo -e "${RED}=== DIAGNOSTICS ===${NC}"
            echo "Working directory: $(pwd)"
            echo "Output directory: $OUTPUT_DIR"
            echo "Looking for pattern: ${output_prefix}_*.{jpg,bmp,raw}"
            echo "CLI exit code: $exit_code"
            echo "Output dir exists: $(test -d "$OUTPUT_DIR" && echo 'YES' || echo 'NO')"
            if [[ -d "$OUTPUT_DIR" ]]; then
                echo "Files in output dir:"
                ls -la "$OUTPUT_DIR" 2>/dev/null | head -20 || echo "  (empty or error)"
            fi
            echo -e "${RED}CLI output:${NC}"
            echo "$output" | tail -20
            echo -e "${RED}===================${NC}"
            print_fail "Expected $expected_count files, got $file_count"
            test_status="failed"
            TEST_RESULTS+=("$example_name:$test_status")
            return 1
        fi
    fi

    print_pass "$example_name"
    TEST_RESULTS+=("$example_name:$test_status")
    return 0
}

# Run a single Node.js example
# Args: $1 = js file path
#       $2 = output prefix (optional, for file count validation)
#       $3 = expected file count (optional, default 0 = no check)
run_node_example() {
    local js_file="$1"
    local output_prefix="$2"
    local expected_count="${3:-0}"
    local example_name=$(basename "$js_file" .js)

    ((TOTAL_TESTS++))
    print_test "$example_name (Node.js)"

    # Check if JS file exists
    if [[ ! -f "$js_file" ]]; then
        print_fail "JS file not found: $js_file"
        TEST_RESULTS+=("$example_name:failed")
        return 1
    fi

    # Check if Node.js is available
    if ! command -v node &>/dev/null; then
        print_skip "Node.js not available"
        TEST_RESULTS+=("$example_name:skipped")
        return 0
    fi

    # Determine the node output directory (examples write to examples/node/output/)
    local node_output_dir="$EXAMPLES_DIR/node/output"

    # Clean output files for this example if prefix specified
    if [[ -n "$output_prefix" ]]; then
        rm -f "$node_output_dir/${output_prefix}_"*.jpg "$node_output_dir/${output_prefix}_"*.bmp 2>/dev/null || true
    fi

    # Run the Node.js example
    print_info "Running Node.js example..."
    local output
    local exit_code=0
    local test_status="passed"

    cd "$WORK_DIR"
    output=$(run_with_timeout "$RUN_TIMEOUT" node "$js_file" 2>&1) || exit_code=$?

    # Check for critical errors
    if [[ $exit_code -ne 0 ]]; then
        # Check if it's a module availability issue
        if echo "$output" | grep -qi "Unknown module\\|Module not found\\|not available"; then
            print_skip "Module not available: $example_name"
            test_status="skipped"
            TEST_RESULTS+=("$example_name:$test_status")
            return 0
        fi

        # Check if addon failed to load (which is expected if not built)
        if echo "$output" | grep -qi "Failed to load addon"; then
            print_skip "Node.js addon not available"
            test_status="skipped"
            TEST_RESULTS+=("$example_name:$test_status")
            return 0
        fi

        echo -e "${RED}Error output:${NC}"
        echo "$output" | tail -15
        print_fail "Node.js example failed with exit code $exit_code"
        test_status="failed"
        TEST_RESULTS+=("$example_name:$test_status")
        return 1
    fi

    # Check for errors in output even if exit code is 0
    if echo "$output" | grep -qi "Error:\\|exception\\|AIPException"; then
        if echo "$output" | grep -qi "not found\\|Unknown module"; then
            print_skip "Module not available: $example_name"
            test_status="skipped"
            TEST_RESULTS+=("$example_name:$test_status")
            return 0
        fi
        echo -e "${RED}Error output:${NC}"
        echo "$output" | tail -15
        print_fail "Example reported errors"
        test_status="failed"
        TEST_RESULTS+=("$example_name:$test_status")
        return 1
    fi

    # If output prefix specified, verify files were created
    if [[ -n "$output_prefix" ]] && [[ "$expected_count" -gt 0 ]]; then
        local file_count
        file_count=$(ls "$node_output_dir/${output_prefix}_"*.jpg "$node_output_dir/${output_prefix}_"*.bmp 2>/dev/null | wc -l)

        print_info "Generated $file_count files (expected: $expected_count)"

        if [[ "$file_count" -lt "$expected_count" ]]; then
            echo -e "${RED}Node.js output:${NC}"
            echo "$output" | tail -20
            print_fail "Expected $expected_count files, got $file_count"
            test_status="failed"
            TEST_RESULTS+=("$example_name:$test_status")
            return 1
        fi
    fi

    print_pass "$example_name"
    TEST_RESULTS+=("$example_name:$test_status")
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
# Node.js Examples Tests
# ==============================================================================

if [ "$TEST_NODE" = true ]; then
    print_header "Testing Node.js Addon Examples"

    # Check if Node.js is available
    if ! command -v node &>/dev/null; then
        echo -e "${YELLOW}Warning: Node.js not found. Skipping Node.js tests.${NC}"
    else
        echo -e "${GREEN}Node.js:${NC} $(node --version)"

        # Check if addon exists (expected at bin/aprapipes.node)
        if [[ -f "$WORK_DIR/bin/aprapipes.node" ]]; then
            echo -e "${GREEN}Addon:${NC} $WORK_DIR/bin/aprapipes.node"
        else
            echo -e "${YELLOW}Warning: Node.js addon not found at $WORK_DIR/bin/aprapipes.node${NC}"
        fi

        # Create node output directory if needed
        mkdir -p "$EXAMPLES_DIR/node/output"

        # Basic examples that work without external dependencies
        # These use TestSignalGenerator + FileWriterModule
        # Output file patterns: frame_????.jpg, processed_????.jpg, etc.
        run_node_example "$EXAMPLES_DIR/node/basic_pipeline.js" "frame" 10 || true
        run_node_example "$EXAMPLES_DIR/node/event_handling.js" "event" 10 || true
        run_node_example "$EXAMPLES_DIR/node/image_processing.js" "processed" 10 || true
        run_node_example "$EXAMPLES_DIR/node/ptz_control.js" "ptz" 10 || true

        # archive_space_demo.js is pure JS (doesn't use addon modules) - still run it
        run_node_example "$EXAMPLES_DIR/node/archive_space_demo.js" "" 0 || true

        # Skip these - they need external resources:
        # - rtsp_pusher_demo.js: needs RTSP server
        # - face_detection_demo.js: needs model files
        # - jetson_l4tm_demo.js: ARM64/Jetson only (tested separately)
    fi
fi

# ==============================================================================
# Cleanup and Summary
# ==============================================================================

if [ "$KEEP_OUTPUTS" = false ]; then
    print_info "Cleaning up output files..."
    rm -f "$OUTPUT_DIR"/*.jpg "$OUTPUT_DIR"/*.bmp "$OUTPUT_DIR"/*.raw 2>/dev/null || true
    # Also clean Node.js output directory
    rm -rf "$EXAMPLES_DIR/node/output" 2>/dev/null || true
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
  "script": "test_all_examples.sh",
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
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
fi
