#!/bin/bash
# ==============================================================================
# Declarative Pipeline Integration Test Script
# ==============================================================================
# This script tests all working declarative pipelines to ensure no regressions.
#
# Runtime: Uses Node.js addon when available, falls back to CLI
#
# Usage:
#   ./scripts/test_declarative_pipelines.sh [options]
#
# Options:
#   --validate-only    Only validate pipelines, don't run them
#   --verbose          Show detailed output
#   --keep-outputs     Don't cleanup output files after tests
#   --pipeline <name>  Test only a specific pipeline (e.g., "01_simple")
#   --help             Show this help message
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#   2 - Script error (missing addon/CLI, etc.)
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
CLI_PATH="$PROJECT_ROOT/build/aprapipes_cli"
NODE_ADDON_PATH="$PROJECT_ROOT/build/aprapipes.node"
NODE_RUNNER="$SCRIPT_DIR/pipeline_test_runner.js"
WORKING_DIR="$PROJECT_ROOT/docs/declarative-pipeline/examples/working"
OUTPUT_DIR="$PROJECT_ROOT/data/testOutput"
RUN_DURATION=2  # seconds to run each pipeline

# Runtime mode: 'node' or 'cli'
RUNTIME_MODE="cli"

# On Linux, the Node.js addon requires GTK3 to be preloaded for OpenCV/GUI symbols
if [[ "$(uname -s)" == "Linux" ]]; then
    GTK3_LIB=$(ldconfig -p 2>/dev/null | grep 'libgtk-3.so.0' | awk '{print $NF}' | head -1)
    if [[ -n "$GTK3_LIB" && -f "$GTK3_LIB" ]]; then
        export LD_PRELOAD="$GTK3_LIB"
    fi
fi

# Options
VALIDATE_ONLY=false
VERBOSE=false
KEEP_OUTPUTS=false
SPECIFIC_PIPELINE=""

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

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --validate-only)
                VALIDATE_ONLY=true
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
            --pipeline)
                SPECIFIC_PIPELINE="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check for Node.js and addon first (preferred method)
    if command -v node &> /dev/null && [ -f "$NODE_ADDON_PATH" ] && [ -f "$NODE_RUNNER" ]; then
        RUNTIME_MODE="node"
        print_info "Node.js found: $(node --version)"
        print_info "Node addon found: $NODE_ADDON_PATH"
        echo -e "${GREEN}Using Node.js runtime${NC}"
    else
        # Fallback to CLI
        if [ ! -f "$CLI_PATH" ]; then
            echo -e "${RED}ERROR: Neither Node.js addon nor CLI found${NC}"
            echo "Please build the project first: cmake --build build -DBUILD_NODE_ADDON=ON"
            exit 2
        fi
        RUNTIME_MODE="cli"
        print_info "CLI found: $CLI_PATH"
        echo -e "${YELLOW}Using CLI runtime (Node.js addon not available)${NC}"
    fi

    # Check working directory exists
    if [ ! -d "$WORKING_DIR" ]; then
        echo -e "${RED}ERROR: Working directory not found: $WORKING_DIR${NC}"
        exit 2
    fi
    print_info "Working directory: $WORKING_DIR"

    # Create output directory if needed
    mkdir -p "$OUTPUT_DIR"
    print_info "Output directory: $OUTPUT_DIR"

    echo -e "${GREEN}Prerequisites OK${NC}"
}

# Cleanup function
cleanup_outputs() {
    if [ "$KEEP_OUTPUTS" = false ]; then
        print_info "Cleaning up test outputs..."
        rm -f "$OUTPUT_DIR"/test_pipeline_*.jpg 2>/dev/null || true
        rm -f "$OUTPUT_DIR"/test_pipeline_*.bmp 2>/dev/null || true
        rm -f "$OUTPUT_DIR"/affine_*.jpg 2>/dev/null || true
        rm -f "$OUTPUT_DIR"/bmp_*.bmp 2>/dev/null || true
        rm -rf /tmp/declarative_test 2>/dev/null || true
    fi
}

# ==============================================================================
# Test Functions
# ==============================================================================

# Validate a pipeline file
validate_pipeline() {
    local pipeline_file="$1"
    local pipeline_name=$(basename "$pipeline_file" .json)

    print_info "Validating $pipeline_name..."

    if [ "$RUNTIME_MODE" = "node" ]; then
        if node "$NODE_RUNNER" validate "$pipeline_file" > /dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    else
        if "$CLI_PATH" validate "$pipeline_file" > /dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    fi
}

# Run a pipeline for a short duration
run_pipeline() {
    local pipeline_file="$1"
    local duration="$2"
    local pipeline_name=$(basename "$pipeline_file" .json)

    print_info "Running $pipeline_name for ${duration}s..."

    if [ "$RUNTIME_MODE" = "node" ]; then
        # Use Node.js runner (handles start/stop/terminate internally)
        if node "$NODE_RUNNER" run "$pipeline_file" "$duration" > /tmp/pipeline_$$.log 2>&1; then
            if [ "$VERBOSE" = true ]; then
                echo "Pipeline log:"
                cat /tmp/pipeline_$$.log
            fi
            rm -f /tmp/pipeline_$$.log
            return 0
        else
            if [ "$VERBOSE" = true ]; then
                echo "Pipeline log:"
                cat /tmp/pipeline_$$.log
            fi
            rm -f /tmp/pipeline_$$.log
            return 1
        fi
    else
        # Use CLI (original implementation)
        # Start pipeline in background
        "$CLI_PATH" run "$pipeline_file" > /tmp/pipeline_$$.log 2>&1 &
        local pid=$!

        # Wait for specified duration
        sleep "$duration"

        # Stop the pipeline gracefully
        kill -SIGINT $pid 2>/dev/null || true

        # Wait for it to finish (with timeout)
        local wait_count=0
        while kill -0 $pid 2>/dev/null && [ $wait_count -lt 10 ]; do
            sleep 0.5
            ((wait_count++))
        done

        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid 2>/dev/null || true
        fi

        # Check for errors in log
        if grep -q "error\|FAILED\|Assertion failed" /tmp/pipeline_$$.log 2>/dev/null; then
            if [ "$VERBOSE" = true ]; then
                echo "Pipeline log:"
                cat /tmp/pipeline_$$.log
            fi
            rm -f /tmp/pipeline_$$.log
            return 1
        fi

        rm -f /tmp/pipeline_$$.log
        return 0
    fi
}

# Check if output files were created
check_output_files() {
    local pattern="$1"
    local min_count="$2"

    local count=$(ls -1 $pattern 2>/dev/null | wc -l | tr -d ' ')

    if [ "$count" -ge "$min_count" ]; then
        print_info "Found $count output files (expected >= $min_count)"
        return 0
    else
        print_info "Found only $count output files (expected >= $min_count)"
        return 1
    fi
}

# ==============================================================================
# Pipeline-Specific Tests
# ==============================================================================

# Get pipeline configuration
# Returns: can_run|output_pattern|min_outputs|notes
get_pipeline_config() {
    local name="$1"
    case "$name" in
        "01_simple_source_sink")
            echo "yes|||Simple source to sink"
            ;;
        "02_three_module_chain")
            echo "yes|||Three module chain"
            ;;
        "03_split_pipeline")
            echo "yes|||Split to multiple sinks"
            ;;
        "04_ptz_with_conversion")
            echo "yes|||PTZ with color conversion"
            ;;
        "05_transform_ptz_with_conversion")
            echo "yes|||Transform + PTZ chain"
            ;;
        "06_face_detector_with_conversion")
            echo "skip|||Requires face detection model"
            ;;
        "09_face_detection_demo")
            echo "skip|||Requires face detection model"
            ;;
        "10_bmp_converter_pipeline")
            echo "yes|/tmp/declarative_test/bmp_output/frame_*.bmp|3|BMP converter output"
            ;;
        "14_affine_transform_chain")
            echo "yes|||Affine transform chain"
            ;;
        "14_affine_transform_demo")
            echo "yes|$OUTPUT_DIR/affine_*.jpg|5|Affine transform with JPEG output"
            ;;
        *)
            echo "yes|||Unknown pipeline"
            ;;
    esac
}

# Run test for a single pipeline
test_pipeline() {
    local pipeline_file="$1"
    local pipeline_name=$(basename "$pipeline_file" .json)

    ((TOTAL_TESTS++))
    print_test "$pipeline_name"

    # Get configuration
    local config=$(get_pipeline_config "$pipeline_name")
    IFS='|' read -r can_run output_pattern min_outputs notes <<< "$config"

    print_info "Config: can_run=$can_run, output=$output_pattern, min=$min_outputs"
    print_info "Notes: $notes"

    # Step 1: Validate
    if ! validate_pipeline "$pipeline_file"; then
        print_fail "$pipeline_name - Validation failed"
        return 1
    fi
    print_info "Validation passed"

    # If validate-only mode, we're done
    if [ "$VALIDATE_ONLY" = true ]; then
        print_pass "$pipeline_name - Validation OK"
        return 0
    fi

    # Step 2: Check if we should run this pipeline
    if [ "$can_run" = "skip" ]; then
        print_skip "$pipeline_name - $notes"
        ((TOTAL_TESTS--))  # Don't count skipped tests
        return 0
    fi

    # Step 3: Clean up any existing outputs and create directories
    if [ -n "$output_pattern" ]; then
        rm -f $output_pattern 2>/dev/null || true
        # Create output directory if needed
        local output_dir=$(dirname "$output_pattern")
        mkdir -p "$output_dir" 2>/dev/null || true
    fi

    # Step 4: Run the pipeline
    if ! run_pipeline "$pipeline_file" "$RUN_DURATION"; then
        print_fail "$pipeline_name - Runtime error"
        return 1
    fi
    print_info "Pipeline ran successfully"

    # Step 5: Check outputs if expected
    if [ -n "$output_pattern" ] && [ -n "$min_outputs" ]; then
        if ! check_output_files "$output_pattern" "$min_outputs"; then
            print_fail "$pipeline_name - Expected output files not found"
            return 1
        fi
    fi

    print_pass "$pipeline_name"
    return 0
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    parse_args "$@"

    print_header "Declarative Pipeline Integration Tests"
    echo "Project root: $PROJECT_ROOT"
    echo "Mode: $([ "$VALIDATE_ONLY" = true ] && echo "Validate only" || echo "Full test")"

    check_prerequisites

    # Cleanup before tests
    cleanup_outputs

    print_header "Running Tests"

    local failed_pipelines=()

    # Find all working pipelines
    for pipeline_file in "$WORKING_DIR"/*.json; do
        if [ ! -f "$pipeline_file" ]; then
            continue
        fi

        local pipeline_name=$(basename "$pipeline_file" .json)

        # Filter by specific pipeline if requested
        if [ -n "$SPECIFIC_PIPELINE" ]; then
            if [[ ! "$pipeline_name" == *"$SPECIFIC_PIPELINE"* ]]; then
                continue
            fi
        fi

        if ! test_pipeline "$pipeline_file"; then
            failed_pipelines+=("$pipeline_name")
        fi
    done

    # Cleanup after tests
    cleanup_outputs

    # Print summary
    print_header "Test Summary"
    echo "Total:   $TOTAL_TESTS"
    echo -e "Passed:  ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed:  ${RED}$FAILED_TESTS${NC}"
    echo -e "Skipped: ${YELLOW}$SKIPPED_TESTS${NC}"

    if [ ${#failed_pipelines[@]} -gt 0 ]; then
        echo ""
        echo -e "${RED}Failed pipelines:${NC}"
        for name in "${failed_pipelines[@]}"; do
            echo "  - $name"
        done
    fi

    echo ""
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed!${NC}"
        exit 1
    fi
}

# Trap to ensure cleanup on exit
trap cleanup_outputs EXIT

main "$@"
