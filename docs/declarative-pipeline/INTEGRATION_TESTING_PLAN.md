# SDK Integration Testing Plan

> Created: 2026-01-17

## Goal

Add integration testing phase to all CI workflows that:
1. Runs examples from the SDK after build using existing test scripts
2. Reports which examples pass/fail per platform (JSON report)
3. Does NOT fail CI builds (informational only, initially)
4. Ensures examples continue working over time

## Existing Test Scripts

We already have well-structured test scripts in `examples/`:

| Script | Purpose | Platforms |
|--------|---------|-----------|
| `test_all_examples.sh` | Basic + CUDA + Advanced | All (cloud + GPU) |
| `test_cuda_examples.sh` | CUDA-specific tests | Windows GPU, Linux GPU |
| `test_jetson_examples.sh` | Jetson L4TM + camera | Jetson only |
| `test_declarative_pipelines.sh` | Full declarative test | All |

## Test Matrix

| Workflow | Cloud Runner | GPU Runner | Scripts |
|----------|--------------|------------|---------|
| CI-Windows | `test_all_examples.sh --basic` | `test_all_examples.sh --cuda` | Both |
| CI-Linux | `test_all_examples.sh --basic` | `test_all_examples.sh --cuda` | Both |
| CI-MacOSX | `test_all_examples.sh --basic` | N/A | Cloud only |
| CI-Linux-ARM64 | `test_jetson_examples.sh` | N/A | Single runner |

## Implementation Plan

### Phase 1: Update Test Scripts for CI

Modify existing scripts to:
1. Accept `--json-report <file>` option for JSON output
2. Accept `--ci` mode to avoid interactive prompts
3. Use SDK paths instead of build paths when in SDK mode
4. Always exit 0 in CI mode (report failures, don't fail build)

### Phase 2: Files to Modify

| File | Changes |
|------|---------|
| `examples/test_all_examples.sh` | Add `--json-report`, `--ci`, `--sdk-dir` options |
| `examples/test_jetson_examples.sh` | Add `--json-report`, `--ci`, `--sdk-dir` options |
| `.github/workflows/build-test.yml` | Add integration test steps (cloud + GPU) |
| `.github/workflows/build-test-macosx.yml` | Add integration test step |
| `.github/workflows/build-test-lin.yml` | Add integration test step |
| `.github/workflows/CI-CUDA-Tests.yml` | Add CUDA integration test step |

### Phase 3: Script Enhancements

#### Add to test_all_examples.sh:

```bash
# New options
JSON_REPORT=""
CI_MODE=false
SDK_DIR=""

# In argument parsing, add:
--json-report)
    JSON_REPORT="$2"
    shift 2
    ;;
--ci)
    CI_MODE=true
    shift
    ;;
--sdk-dir)
    SDK_DIR="$2"
    shift 2
    ;;

# Use SDK paths if specified
if [ -n "$SDK_DIR" ]; then
    CLI_PATH="$SDK_DIR/bin/aprapipes_cli"
    EXAMPLES_DIR="$SDK_DIR/examples"
fi

# At end, generate JSON report if requested
if [ -n "$JSON_REPORT" ]; then
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
  "results": [
$(for key in "${!TEST_RESULTS[@]}"; do
    echo "    {\"name\": \"$key\", \"status\": \"${TEST_RESULTS[$key]}\"},"
done | sed '$ s/,$//')
  ]
}
EOF
fi

# In CI mode, always exit 0
if [ "$CI_MODE" = true ]; then
    exit 0
fi
```

### Phase 4: Workflow Integration

#### build-test.yml (Windows/Linux x64)

```yaml
# After SDK packaging, add in build job:
- name: Run integration tests (cloud)
  if: success()
  continue-on-error: true
  shell: bash
  run: |
    chmod +x examples/test_all_examples.sh
    ./examples/test_all_examples.sh \
      --basic \
      --sdk-dir "${{ github.workspace }}/sdk" \
      --json-report integration_report_cloud.json \
      --ci

- name: Upload integration report (cloud)
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: IntegrationReport_${{ inputs.flav }}_cloud
    path: integration_report_cloud.json
  continue-on-error: true
```

#### CI-CUDA-Tests.yml (GPU runners)

```yaml
# After GPU tests, add:
- name: Run CUDA integration tests
  if: success()
  continue-on-error: true
  shell: bash
  run: |
    chmod +x examples/test_all_examples.sh
    ./examples/test_all_examples.sh \
      --cuda \
      --sdk-dir ./sdk \
      --json-report integration_report_cuda.json \
      --ci

- name: Upload CUDA integration report
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: IntegrationReport_${{ inputs.flav }}_cuda
    path: integration_report_cuda.json
  continue-on-error: true
```

#### build-test-macosx.yml

```yaml
# After SDK packaging, add:
- name: Run integration tests
  if: ${{ success() && !inputs.is-prep-phase }}
  continue-on-error: true
  run: |
    chmod +x examples/test_all_examples.sh
    ./examples/test_all_examples.sh \
      --basic \
      --sdk-dir "${{ github.workspace }}/sdk" \
      --json-report integration_report.json \
      --ci

- name: Upload integration report
  if: ${{ always() && !inputs.is-prep-phase }}
  uses: actions/upload-artifact@v4
  with:
    name: IntegrationReport_${{ inputs.flav }}
    path: integration_report.json
  continue-on-error: true
```

#### build-test-lin.yml (ARM64/Jetson)

```yaml
# After SDK packaging, add:
- name: Run Jetson integration tests
  if: ${{ success() && !inputs.is-prep-phase }}
  continue-on-error: true
  run: |
    chmod +x examples/test_jetson_examples.sh examples/test_all_examples.sh

    # Run Jetson-specific tests
    ./examples/test_jetson_examples.sh \
      --cli \
      --sdk-dir "${{ github.workspace }}/sdk" \
      --json-report integration_report_jetson.json \
      --ci

    # Also run basic tests
    ./examples/test_all_examples.sh \
      --basic \
      --sdk-dir "${{ github.workspace }}/sdk" \
      --json-report integration_report_basic.json \
      --ci

- name: Upload integration reports
  if: ${{ always() && !inputs.is-prep-phase }}
  uses: actions/upload-artifact@v4
  with:
    name: IntegrationReport_${{ inputs.flav }}
    path: |
      integration_report_jetson.json
      integration_report_basic.json
  continue-on-error: true
```

## JSON Report Format

```json
{
  "script": "test_all_examples.sh",
  "timestamp": "2026-01-17T12:00:00Z",
  "platform": "linux-x64",
  "mode": "cloud",
  "summary": {
    "passed": 8,
    "failed": 2,
    "skipped": 3,
    "total": 13
  },
  "results": [
    {"name": "basic/simple_source_sink.json", "status": "passed", "duration_ms": 1200},
    {"name": "basic/face_detection_demo.json", "status": "failed", "error": "Model not found"},
    {"name": "cuda/gaussian_blur.json", "status": "skipped", "reason": "No GPU"}
  ]
}
```

## Artifact Summary

| Workflow | Artifact Name | Contents |
|----------|---------------|----------|
| CI-Windows build | `IntegrationReport_Windows_cloud` | Basic tests |
| CI-Windows cuda | `IntegrationReport_Windows-CUDA_cuda` | CUDA tests |
| CI-Linux build | `IntegrationReport_Linux_cloud` | Basic tests |
| CI-Linux cuda | `IntegrationReport_Linux-CUDA_cuda` | CUDA tests |
| CI-MacOSX | `IntegrationReport_MacOSX` | Basic tests |
| CI-Linux-ARM64 | `IntegrationReport_Linux_ARM64` | Basic + Jetson tests |

## Implementation Tasks

### Phase 1: Script Updates

- [x] Update `test_all_examples.sh` with `--json-report`, `--ci`, `--sdk-dir`
- [x] Update `test_jetson_examples.sh` with `--json-report`, `--ci`, `--sdk-dir`
- [ ] Test scripts locally with new options

### Phase 2: Workflow Integration

- [x] Add integration steps to `build-test.yml`
- [x] Add integration steps to `build-test-macosx.yml`
- [x] Add integration steps to `build-test-lin.yml`
- [x] Add CUDA integration steps to `CI-CUDA-Tests.yml`

### Phase 3: Verification

- [ ] Verify all workflows produce reports
- [ ] Verify CI doesn't fail on test failures
- [ ] Review reports and fix obviously broken examples

### Phase 4: Future Enhancements (Deferred)

- [ ] Create summary dashboard in PR comments
- [ ] Add GitHub check annotations for failures
- [ ] Track pass/fail trends over time
- [ ] Option to fail builds when critical examples break

## Success Criteria

Phase 1 complete when:
- [ ] Scripts accept new CLI options
- [ ] JSON reports generated correctly
- [ ] Scripts work with SDK directory structure

Phase 2 complete when:
- [ ] All 4 workflows produce integration reports
- [ ] Reports uploaded as artifacts
- [ ] CI does not fail on integration test failures
- [ ] At least basic examples pass on each platform

## Next Steps

1. Update `examples/test_all_examples.sh` with new options
2. Update `examples/test_jetson_examples.sh` with new options
3. Test locally
4. Add workflow integration steps
5. Push and verify reports generated
