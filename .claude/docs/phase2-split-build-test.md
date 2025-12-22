# Phase 2: Split Build and Test Workflows

## Goal

Create a clean two-workflow architecture that:
1. Builds and tests on cloud (CUDA-agnostic)
2. Publishes SDK artifact for distribution
3. Runs GPU tests on self-hosted runners (triggered separately)

## Design Principles

1. **CI-Build-Test workflow is CUDA-agnostic** - It has no knowledge of whether it's running on a CUDA or non-CUDA device. CUDA-dependent tests skip silently via runtime detection (`if_compute_cap_supported()`).

2. **SDK is a distributable artifact** - Contains everything needed for external users (include/, lib/, bin/, doc/).

3. **GPU testing is decoupled** - Runs as a separate workflow triggered by successful build. If GPU runners are offline, builds still succeed.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Workflow 1: CI-<OS>-Build-Test                              │
│  Trigger: push, pull_request                                 │
│  Runner: GitHub-hosted (cloud)                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1.1: Build                                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ - Checkout code                                        │  │
│  │ - Setup vcpkg, restore cache                           │  │
│  │ - CMake configure + build                              │  │
│  │ - Compile with CUDA headers (APRA_HAS_CUDA_HEADERS)    │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  Step 1.2: Test (CUDA-agnostic)                       ←GATE  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ - Run ALL tests                                        │  │
│  │ - CUDA tests skip silently (no GPU = precondition fail)│  │
│  │ - Non-CUDA tests must pass                             │  │
│  │ - If ANY test fails → workflow fails, no SDK published │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  Step 1.3: Publish Test Results                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ - Upload test results XML                              │  │
│  │ - Publish to PR checks                                 │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  Step 1.4: Publish SDK Artifact                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ - Package SDK (include/, lib/, bin/, doc/)             │  │
│  │ - Upload as GitHub artifact                            │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                           │
                           │ workflow_run trigger
                           │ (only if Workflow 1 succeeds)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Workflow 2: CI-<OS>-CUDA-Tests                              │
│  Trigger: workflow_run (on CI-<OS>-Build-Test success)       │
│  Runner: Self-hosted GPU (windows-cuda, linux-cuda)          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 2.1: Setup                                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ - git checkout (gets test_data, source)                │  │
│  │ - Download SDK artifact from Workflow 1                │  │
│  │ - Extract bin/ folder (exe + DLLs)                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  Step 2.2: Run CUDA Tests                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ - Run ALL tests (same exe from Workflow 1)             │  │
│  │ - CUDA tests NOW RUN (GPU available!)                  │  │
│  │ - Full test coverage with GPU acceleration             │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  Step 2.3: Publish CUDA Test Results                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ - Upload test results XML                              │  │
│  │ - Publish to PR checks (separate check from Workflow 1)│  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Workflows Per Platform

| Platform | Workflow 1 (Build+Test) | Workflow 2 (CUDA Tests) | GPU Runner |
|----------|-------------------------|-------------------------|------------|
| Windows x64 | CI-Windows-Build-Test | CI-Windows-CUDA-Tests | windows-cuda |
| Linux x64 | CI-Linux-Build-Test | CI-Linux-CUDA-Tests | linux-cuda |
| Linux ARM64 | CI-ARM64-Build-Test | (native GPU, no split needed) | AGX |
| MacOSX | CI-MacOSX-Build-Test | N/A (no CUDA on Mac) | - |

## SDK Artifact Structure

```
aprapipes-sdk-<os>-<arch>/
├── include/
│   └── *.h                    # Public headers
├── lib/
│   ├── aprapipes.lib          # Static/import library
│   └── *.lib                  # Dependencies
├── bin/
│   ├── aprapipesut.exe        # Test executable
│   └── *.dll                  # Runtime DLLs
└── doc/
    └── *.md                   # Documentation
```

## Workflow YAML Examples

### Workflow 1: CI-Windows-Build-Test

```yaml
name: CI-Windows-Build-Test

on:
  push:
    paths-ignore: ['**.md', '.claude/**']
  pull_request:
    branches: [main]

jobs:
  build-test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      # ... vcpkg setup, cmake configure, build ...

      - name: Run Tests
        run: |
          ./build/aprapipesut.exe --log_level=test_suite \
            --report_format=XML --report_sink=test_results.xml
        # CUDA tests skip silently - workflow doesn't know or care

      - name: Publish Test Results
        uses: dorny/test-reporter@v1
        with:
          name: 'Windows Build Tests'
          path: test_results.xml
          reporter: junit

      - name: Package SDK
        run: |
          mkdir -p sdk/{include,lib,bin,doc}
          cp -r base/include/* sdk/include/
          cp build/*.lib sdk/lib/
          cp build/*.exe build/*.dll sdk/bin/
          # ... copy docs ...

      - name: Upload SDK Artifact
        uses: actions/upload-artifact@v4
        with:
          name: aprapipes-sdk-windows-x64
          path: sdk/
          retention-days: 7
```

### Workflow 2: CI-Windows-CUDA-Tests

```yaml
name: CI-Windows-CUDA-Tests

on:
  workflow_run:
    workflows: ["CI-Windows-Build-Test"]
    types: [completed]

jobs:
  cuda-tests:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: [self-hosted, windows-cuda]
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
          # Gets test_data and source

      - name: Download SDK Artifact
        uses: actions/download-artifact@v4
        with:
          name: aprapipes-sdk-windows-x64
          path: sdk/
          run-id: ${{ github.event.workflow_run.id }}
          github-token: ${{ secrets.GITHUB_TOKEN }}

      - name: Run CUDA Tests
        run: |
          cd sdk/bin
          ./aprapipesut.exe --log_level=test_suite \
            --report_format=XML --report_sink=cuda_test_results.xml
        # CUDA tests NOW RUN because GPU is available!

      - name: Publish CUDA Test Results
        uses: dorny/test-reporter@v1
        with:
          name: 'Windows CUDA Tests'
          path: sdk/bin/cuda_test_results.xml
          reporter: junit
```

## Workflows to Delete (After Migration)

| Delete | Replaced By |
|--------|-------------|
| CI-Windows-Unified.yml | CI-Windows-Build-Test.yml |
| CI-Win-CUDA.yml | CI-Windows-CUDA-Tests.yml |
| CI-Linux-CUDA-Unified.yml | CI-Linux-Build-Test.yml |
| CI-Linux-CUDA.yml | CI-Linux-CUDA-Tests.yml |

## Benefits

1. **Clean separation** - Build workflow is CUDA-agnostic
2. **Fast feedback** - Cloud tests complete quickly
3. **No blocking** - GPU runner offline doesn't block builds
4. **SDK distribution** - Every successful build produces distributable SDK
5. **Full coverage** - CUDA tests run on real GPU hardware
6. **Simpler maintenance** - Clear workflow responsibilities

## Implementation Steps

| Step | Task | Effort |
|------|------|--------|
| 1 | Create SDK packaging in CMake/script | Medium |
| 2 | Create CI-Windows-Build-Test workflow | Low |
| 3 | Create CI-Windows-CUDA-Tests workflow | Low |
| 4 | Test and validate Windows workflows | Medium |
| 5 | Repeat for Linux | Low |
| 6 | Delete old workflows | Low |

## Open Items

1. **Artifact retention**: 7 days default, adjust based on storage costs
2. **SDK versioning**: Include version/commit in artifact name?
3. **ARM64/Jetson**: Keep as single workflow (native GPU) or split?

---
*Created: 2025-12-22*
*Status: PLANNING*
