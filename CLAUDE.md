# CLAUDE.md — ApraPipes DevOps Agent

## Mission Statement

### Phase 1: Unified Builds (COMPLETED)
Create unified builds that compile with CUDA+cuDNN support but detect GPU availability at runtime, eliminating the need for separate CUDA and NoCUDA build variants.

### Phase 2: Split Build/Test Architecture (IN PROGRESS)
Create a two-workflow architecture per platform:
1. **CI-\<OS\>-Build-Test** - Builds on cloud, runs tests (CUDA tests skip silently), publishes SDK artifact
2. **CI-\<OS\>-CUDA-Tests** - Triggered by successful Build-Test, runs on self-hosted GPU, publishes CUDA test results

**Key Design Principles:**
- Build-Test workflow is **CUDA-agnostic** - no knowledge of GPU availability
- SDK artifact contains: include/, lib/, bin/, doc/
- GPU testing is decoupled - offline GPU runners don't block builds
- Self-hosted runners only run tests (no builds)

### Progress

| Platform | Build-Test | CUDA-Tests | Status |
|----------|------------|------------|--------|
| Windows x64 | CI-Windows-Build-Test | CI-Windows-CUDA-Tests | TODO |
| Linux x64 | CI-Linux-Build-Test | CI-Linux-CUDA-Tests | TODO |
| Linux ARM64 (Jetson) | CI-ARM64-Build-Test | (native GPU) | TODO |
| MacOSX | CI-MacOSX-Build-Test | N/A | TODO |

### Key Insight
Runtime CUDA detection via Driver API (`libcuda.so.1` on Linux, `nvcuda.dll` on Windows) allows binaries compiled with CUDA to run on systems without GPU/driver. Tests skip gracefully when GPU unavailable.

---

## Operating Instructions

### At Session Start
1. Read `.claude/LEARNINGS.md` — don't repeat past mistakes
2. Read `.claude/CURRENT_STATE.md` — know where we left off
3. Check build status: `gh run list -L3`

### During Work
- After triggering workflows, ALWAYS watch them: `gh run watch <id> --exit-status`
- Update `CURRENT_STATE.md` when significant progress is made
- Add new learnings to `LEARNINGS.md` immediately after discovering them

### Before Stopping
1. Update `CURRENT_STATE.md` with current status
2. Document any new learnings in `LEARNINGS.md`
3. Check for pending workflows: `gh run list -L3 --json status -q '.[] | select(.status=="in_progress")'`

### Slash Commands
- `/project:status` — Quick status check, continue working
- `/project:resume` — Resume from saved state
- `/project:checkpoint` — Force save current state
- `/project:last-run` — Analyze most recent GH Actions run
- `/project:learned <desc>` — Quick-add a learning

---

## Key Files

### State & Memory
- `.claude/CURRENT_STATE.md` — Current task, build status, blockers
- `.claude/LEARNINGS.md` — Institutional memory, past mistakes & fixes

### Documentation
- `.claude/docs/phase2-split-build-test.md` — Phase 2 detailed design
- `.claude/docs/` — Other implementation docs, experiment results
- `.claude/skills/aprapipes-devops/` — Troubleshooting guides by platform

### CI Workflows (Phase 2 Target)
- `.github/workflows/CI-<OS>-Build-Test.yml` — Build + test on cloud
- `.github/workflows/CI-<OS>-CUDA-Tests.yml` — GPU tests on self-hosted
- `.github/workflows/build-test-<os>.yml` — Reusable build workflows
- `.github/workflows/publish-test.yml` — Reusable test publishing

### Key Config
- `vcpkg/triplets/community/x64-windows-cuda.cmake` — Custom triplet for Windows CUDA (v142 toolset)
- `base/vcpkg.json` — Dependencies with CUDA features
- `vcpkg/ports/opencv4/vcpkg.json` — OpenCV with pthreads for CUDA

---

## Windows Build Issues (All Resolved)

1. **pthreads missing** — Added to opencv4 cuda feature dependency
2. **CUDA compiler not found** — Copy MSBuildExtensions to VS BuildCustomizations
3. **VS version mismatch** — CUDA 11.8 requires v142 toolset, not v143
4. **DLL_NOT_FOUND on non-GPU systems** — Use DELAYLOAD for all CUDA DLLs (toolkit + driver)

### Windows DELAYLOAD Solution
All CUDA DLLs must be delay-loaded to allow the exe to start on systems without NVIDIA drivers:
- **Toolkit DLLs**: nvjpeg64_11, nppig64_11, nppicc64_11, nppidei64_11, nppial64_11, nppc64_11, cublas64_11, cublasLt64_11, cudart64_110
- **Driver DLLs**: nvcuvid, nvEncodeAPI64

---

## Runtime Detection Approach

### Linux
```cpp
void* handle = dlopen("libcuda.so.1", RTLD_LAZY);
auto cuInit = (cuInit_t)dlsym(handle, "cuInit");
// Call cuInit, cuDeviceGetCount via function pointers
```

### Windows
```cpp
HMODULE handle = LoadLibraryA("nvcuda.dll");
auto cuInit = (cuInit_t)GetProcAddress(handle, "cuInit");
// Same Driver API, different OS primitives
```

Tests detect GPU availability and skip CUDA tests gracefully when unavailable.
