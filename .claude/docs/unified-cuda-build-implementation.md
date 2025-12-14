# Unified CUDA Build Implementation Guide

**Branch:** `feature/get-rid-of-nocuda-builds`
**Status:** In Progress
**Target Platform (Phase 1):** Linux x64 only
**DO NOT MERGE TO MAIN** until explicitly approved

---

## 🚨 CRITICAL RULE FOR AGENTS 🚨

**WHEN YOU TRIGGER A WORKFLOW, YOU MUST ACTIVELY MONITOR IT!**

- **DO NOT** sit idle waiting for user to ask for status
- **DO** use `gh run watch <run-id>` to monitor progress
- **DO** check results immediately when workflow completes
- **DO** update the Progress Tracking section with results
- **DO** automatically proceed to next experiment if current passes
- **DO** document failures and blockers immediately
- **DO** push work forward autonomously

**Example workflow:**
```bash
# Trigger workflow
RUN_ID=$(gh workflow run experiment-01-cuda-install.yml --ref feature/get-rid-of-nocuda-builds --json 2>&1 | grep -oP 'Run ID: \K[0-9]+')

# Watch it actively
gh run watch $RUN_ID

# Check result
STATUS=$(gh run view $RUN_ID --json conclusion --jq '.conclusion')

# Document immediately
if [ "$STATUS" = "success" ]; then
    # Update Progress Tracking section
    # Proceed to next experiment
    # Commit results
else
    # Document failure
    # Analyze logs
    # Propose fixes or alternative approaches
fi
```

---

## Mission Statement

Eliminate the complexity of maintaining separate CUDA and NoCUDA builds by creating a single CUDA-enabled build that detects CUDA availability at runtime and gracefully degrades when GPU is unavailable.

**Current Problem:**
- 7 workflows (3 NoCUDA + 4 CUDA)
- Duplicate vcpkg caches
- Duplicate build configurations
- CUDA tests only on self-hosted runners (slow feedback)

**Target Solution:**
- 3 workflows (Windows + Linux + ARM64)
- Single vcpkg cache per platform
- Build on GitHub runners (fast), test CUDA on self-hosted
- Runtime CUDA detection instead of build-time conditionals

---

## Critical Constraints

1. **Branch Isolation:** All work on `feature/get-rid-of-nocuda-builds` branch
2. **Linux First:** Prove end-to-end on Linux before touching Windows/Mac
3. **No Main Branch Changes:** Can commit/push to feature branch, NO MERGE without approval
4. **Workflow Isolation:** Disable all workflows except experimental Linux workflow
5. **Backward Compatibility:** Keep existing code working during transition

---

## Current State (Completed Research)

### What We Found ✅

1. **Runtime CUDA Detection EXISTS:**
   - `CudaUtils::isCudaSupported()` in `base/src/CudaCommon.cpp:4-15`
   - Uses `cudaGetDeviceCount()` API
   - Already used in tests

2. **Test Framework Ready:**
   - `if_compute_cap_supported()` in `base/test/nv_test_utils.h:27-54`
   - `if_h264_encoder_supported()` for codec checks
   - Boost.Test preconditions skip tests when CUDA unavailable
   - ~30+ test files already use this pattern

3. **Build System:**
   - CMake option: `ENABLE_CUDA` (ON/OFF)
   - Compile flag: `APRA_CUDA_ENABLED`
   - Used in ~50+ files with `#ifdef APRA_CUDA_ENABLED`

4. **vcpkg Dependencies:**
   - `base/vcpkg.json` has opencv4[cuda,cudnn], whisper[cuda]
   - OpenCV already uses runtime detection model
   - Static CUDA runtime: `cudart_static.lib`

### Key Insight

**OpenCV already does exactly what we want!**
- Builds with `WITH_CUDA=ON`
- Runs on systems without GPU (just skips GPU features)
- Runtime detection, not build-time conditional

---

## Implementation Phases

### Phase 0: Experiments (Current Phase)

**Goal:** Validate core assumptions before major changes

**Experiments Required:**

#### Experiment 1: CUDA Toolkit on GitHub Runner ⚠️ CRITICAL
**File:** `.github/workflows/experiment-01-cuda-install.yml`

```yaml
name: Experiment-01-CUDA-Install
on: workflow_dispatch

jobs:
  test-cuda-install:
    runs-on: ubuntu-latest
    steps:
      - name: Install CUDA Toolkit
        run: |
          # Install CUDA 11.8 (matches current builds)
          wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
          sudo dpkg -i cuda-keyring_1.0-1_all.deb
          sudo apt-get update
          sudo apt-get -y install cuda-toolkit-11-8

      - name: Test CUDA Compiler
        run: |
          export PATH=/usr/local/cuda-11.8/bin:$PATH
          nvcc --version

      - name: Compile Simple CUDA Code
        run: |
          cat > test.cu << 'EOF'
          #include <cuda_runtime.h>
          #include <stdio.h>
          int main() {
              int deviceCount = 0;
              cudaError_t error = cudaGetDeviceCount(&deviceCount);
              printf("CUDA Device Count: %d (error: %d)\n", deviceCount, error);
              return 0;
          }
          EOF
          /usr/local/cuda-11.8/bin/nvcc test.cu -o test
          ./test
```

**Success Criteria:**
- [ ] CUDA toolkit installs successfully
- [ ] nvcc compiles CUDA code
- [ ] Executable runs without crash
- [ ] `cudaGetDeviceCount()` returns 0 (no GPU) without error

**If Fails:** Entire approach invalid, need alternative strategy

---

#### Experiment 2: vcpkg OpenCV with CUDA
**File:** `.github/workflows/experiment-02-vcpkg-opencv-cuda.yml`

```yaml
name: Experiment-02-vcpkg-opencv-cuda
on: workflow_dispatch

jobs:
  test-opencv-cuda-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Install CUDA Toolkit
        run: |
          wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
          sudo dpkg -i cuda-keyring_1.0-1_all.deb
          sudo apt-get update
          sudo apt-get -y install cuda-toolkit-11-8
          echo "/usr/local/cuda-11.8/bin" >> $GITHUB_PATH

      - name: Install vcpkg Dependencies
        run: |
          export CUDA_PATH=/usr/local/cuda-11.8
          ./vcpkg/bootstrap-vcpkg.sh
          ./vcpkg/vcpkg install opencv4[core,cuda,cudnn]:x64-linux
```

**Success Criteria:**
- [ ] vcpkg builds opencv4 with CUDA features
- [ ] No GPU-related build errors
- [ ] Libraries created successfully

**If Fails:** May need to pin opencv4 version or adjust CUDA version

---

#### Experiment 3: CudaCapabilities Singleton
**Goal:** Implement new capability detection without breaking existing code

**Files to Create:**
- `base/include/CudaCapabilities.h`
- `base/src/CudaCapabilities.cpp`

**Implementation:** See "Detailed Component Design" section below

**Success Criteria:**
- [ ] Compiles with existing code
- [ ] Singleton initializes correctly
- [ ] Returns false for all capabilities on non-GPU system
- [ ] Existing tests still pass

---

#### Experiment 4: Unified Workflow (Linux Only)
**File:** `.github/workflows/experiment-04-unified-linux.yml`

**Jobs:**
1. **build-and-basic-tests** (ubuntu-latest)
   - Install CUDA toolkit
   - Build aprapipesut with ENABLE_CUDA=ON
   - Run all tests (CUDA tests skip via preconditions)
   - Upload aprapipesut artifact

2. **cuda-tests** (self-hosted linux-cuda)
   - Download aprapipesut artifact
   - Run all tests (CUDA tests execute)

**Success Criteria:**
- [ ] Build succeeds on GitHub runner
- [ ] Non-CUDA tests pass on GitHub runner
- [ ] CUDA tests skip gracefully on GitHub runner
- [ ] All tests pass on self-hosted CUDA runner

---

### Phase 1: CudaCapabilities Implementation

**Goal:** Create robust runtime capability detection

**Files:**
```
base/include/CudaCapabilities.h    (NEW)
base/src/CudaCapabilities.cpp      (NEW)
base/CMakeLists.txt                (MODIFY - add new files)
```

**Design:** See "Detailed Component Design" section

**Testing:**
- Build with CUDA device → all capabilities true
- Build without CUDA device → all capabilities false
- No crashes, no undefined behavior

---

### Phase 2: Module Integration

**Goal:** Add capability checks to all CUDA modules

**Strategy:** One module at a time, non-breaking

**Example Module: ResizeNPPI**

```cpp
// base/src/ResizeNPPI.cpp - Constructor
ResizeNPPI::ResizeNPPI(ResizeNPPIProps props) : Module(RESIZE_NPPI, props) {
    auto& cuda = CudaCapabilities::getInstance();

    if (!cuda.isAvailable()) {
        throw AIP_Exception(AIP_FATAL,
            "ResizeNPPI requires CUDA device. No CUDA device available. "
            "Use ImageResizeCV for CPU-based resizing.");
    }

    if (!cuda.hasNPP()) {
        throw AIP_Exception(AIP_FATAL,
            "NPP library not available on CUDA device.");
    }

    // Existing initialization code...
}
```

**Modules to Update (Priority Order):**
1. ResizeNPPI
2. CCNPPI
3. RotateNPPI
4. JPEGEncoderNVJPEG
5. JPEGDecoderNVJPEG
6. H264EncoderNVCodec
7. H264Decoder
8. OverlayNPPI
9. EffectsNPPI
10. CudaMemCopy
11. MemTypeConversion

---

### Phase 3: Workflow Unification (Linux)

**Goal:** Single workflow for Linux builds

**File:** `.github/workflows/CI-Linux-Unified.yml`

**Jobs:**
```yaml
jobs:
  linux-build-prep:
    # Phase 1: Install dependencies + cache
    runs-on: ubuntu-latest

  linux-build-and-test:
    # Phase 2: Build + Non-CUDA tests
    needs: linux-build-prep
    runs-on: ubuntu-latest

  linux-cuda-tests:
    # Phase 3: CUDA tests on self-hosted
    needs: linux-build-and-test
    runs-on: [self-hosted, linux, cuda]
```

**Testing:**
- Run in parallel with existing CI-Linux-NoCUDA and CI-Linux-CUDA
- Compare results
- Validate artifact compatibility

---

### Phase 4: Cleanup (After Validation)

**Goal:** Remove build-time conditionals

**Tasks:**
1. Remove `#ifdef APRA_CUDA_ENABLED` from source files
2. Remove `ENABLE_CUDA` CMake option
3. Disable old workflows (CI-Linux-NoCUDA.yml, CI-Linux-CUDA.yml)
4. Update documentation

**NOT DONE IN THIS BRANCH** - requires approval before main merge

---

## Detailed Component Design

### CudaCapabilities Class

**File:** `base/include/CudaCapabilities.h`

```cpp
#pragma once

#include <vector>
#include <string>
#include <mutex>

/**
 * @brief Runtime CUDA capability detection singleton
 *
 * Detects CUDA device availability and specific capabilities at runtime.
 * Thread-safe singleton initialized once at program startup.
 *
 * Usage:
 *   auto& cuda = CudaCapabilities::getInstance();
 *   if (!cuda.isAvailable()) {
 *       // Handle no CUDA device
 *   }
 */
class CudaCapabilities {
public:
    struct DeviceInfo {
        int deviceId;
        std::string name;
        int computeCapabilityMajor;
        int computeCapabilityMinor;
        size_t totalMemory;
        bool supportsNVENC;
        bool supportsNVDEC;
    };

    /**
     * @brief Get singleton instance
     * Thread-safe lazy initialization
     */
    static CudaCapabilities& getInstance();

    // Core detection
    bool isAvailable() const { return mCudaAvailable; }
    int getDeviceCount() const { return mDeviceCount; }

    // Capability queries
    bool hasMinComputeCapability(int major, int minor) const;
    bool hasNPP() const;
    bool hasNVJPEG() const;
    bool hasNVENC() const;
    bool hasNVDEC() const;

    // Device info
    const std::vector<DeviceInfo>& getDevices() const { return mDevices; }
    const DeviceInfo* getDevice(int deviceId) const;

    // Logging
    void logCapabilities() const;

private:
    CudaCapabilities();  // Private constructor
    ~CudaCapabilities() = default;

    // Prevent copying
    CudaCapabilities(const CudaCapabilities&) = delete;
    CudaCapabilities& operator=(const CudaCapabilities&) = delete;

    // Detection methods
    void detectDevices();
    void detectNVENC();
    void detectNVDEC();

    // State
    bool mCudaAvailable;
    int mDeviceCount;
    std::vector<DeviceInfo> mDevices;
    bool mNVENCAvailable;
    bool mNVDECAvailable;

    static std::mutex sMutex;
};
```

**File:** `base/src/CudaCapabilities.cpp`

```cpp
#include "CudaCapabilities.h"
#include "Logger.h"
#include <cuda_runtime.h>

// For NVENC/NVDEC detection
#ifdef ENABLE_LINUX
#include <dlfcn.h>
#endif

std::mutex CudaCapabilities::sMutex;

CudaCapabilities& CudaCapabilities::getInstance() {
    static CudaCapabilities instance;
    return instance;
}

CudaCapabilities::CudaCapabilities()
    : mCudaAvailable(false)
    , mDeviceCount(0)
    , mNVENCAvailable(false)
    , mNVDECAvailable(false)
{
    detectDevices();
    if (mCudaAvailable) {
        detectNVENC();
        detectNVDEC();
    }
    logCapabilities();
}

void CudaCapabilities::detectDevices() {
    cudaError_t error = cudaGetDeviceCount(&mDeviceCount);

    if (error != cudaSuccess) {
        LOG_INFO << "CUDA not available: " << cudaGetErrorString(error);
        mCudaAvailable = false;
        mDeviceCount = 0;
        return;
    }

    if (mDeviceCount == 0) {
        LOG_INFO << "No CUDA devices found";
        mCudaAvailable = false;
        return;
    }

    mCudaAvailable = true;

    // Query each device
    for (int i = 0; i < mDeviceCount; i++) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, i);

        if (err != cudaSuccess) {
            LOG_WARNING << "Failed to get properties for device " << i;
            continue;
        }

        DeviceInfo info;
        info.deviceId = i;
        info.name = prop.name;
        info.computeCapabilityMajor = prop.major;
        info.computeCapabilityMinor = prop.minor;
        info.totalMemory = prop.totalGlobalMem;

        mDevices.push_back(info);
    }
}

void CudaCapabilities::detectNVENC() {
    // Try to load NVENC library
    // On Linux: libnvidia-encode.so
    // On Windows: nvEncodeAPI.dll

#ifdef ENABLE_LINUX
    void* handle = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
    if (handle) {
        mNVENCAvailable = true;
        dlclose(handle);
        LOG_INFO << "NVENC library available";
    } else {
        LOG_INFO << "NVENC library not available: " << dlerror();
    }
#elif defined(ENABLE_WINDOWS)
    HMODULE handle = LoadLibrary(TEXT("nvEncodeAPI64.dll"));
    if (handle) {
        mNVENCAvailable = true;
        FreeLibrary(handle);
        LOG_INFO << "NVENC library available";
    } else {
        LOG_INFO << "NVENC library not available";
    }
#endif
}

void CudaCapabilities::detectNVDEC() {
    // Try to load NVDEC library
#ifdef ENABLE_LINUX
    void* handle = dlopen("libnvcuvid.so.1", RTLD_LAZY);
    if (handle) {
        mNVDECAvailable = true;
        dlclose(handle);
        LOG_INFO << "NVDEC library available";
    } else {
        LOG_INFO << "NVDEC library not available: " << dlerror();
    }
#elif defined(ENABLE_WINDOWS)
    HMODULE handle = LoadLibrary(TEXT("nvcuvid.dll"));
    if (handle) {
        mNVDECAvailable = true;
        FreeLibrary(handle);
        LOG_INFO << "NVDEC library available";
    } else {
        LOG_INFO << "NVDEC library not available";
    }
#endif
}

bool CudaCapabilities::hasMinComputeCapability(int major, int minor) const {
    if (!mCudaAvailable) return false;

    for (const auto& device : mDevices) {
        if (device.computeCapabilityMajor > major) return true;
        if (device.computeCapabilityMajor == major &&
            device.computeCapabilityMinor >= minor) return true;
    }
    return false;
}

bool CudaCapabilities::hasNPP() const {
    // NPP is part of CUDA toolkit, available if CUDA is available
    // and compute capability >= 3.0
    return mCudaAvailable && hasMinComputeCapability(3, 0);
}

bool CudaCapabilities::hasNVJPEG() const {
    // NVJPEG available with compute capability >= 3.0
    return mCudaAvailable && hasMinComputeCapability(3, 0);
}

bool CudaCapabilities::hasNVENC() const {
    return mNVENCAvailable;
}

bool CudaCapabilities::hasNVDEC() const {
    return mNVDECAvailable;
}

const CudaCapabilities::DeviceInfo* CudaCapabilities::getDevice(int deviceId) const {
    for (const auto& device : mDevices) {
        if (device.deviceId == deviceId) {
            return &device;
        }
    }
    return nullptr;
}

void CudaCapabilities::logCapabilities() const {
    LOG_INFO << "=== CUDA Capabilities ===";
    LOG_INFO << "CUDA Available: " << (mCudaAvailable ? "YES" : "NO");
    LOG_INFO << "Device Count: " << mDeviceCount;

    for (const auto& device : mDevices) {
        LOG_INFO << "Device " << device.deviceId << ": " << device.name;
        LOG_INFO << "  Compute Capability: " << device.computeCapabilityMajor
                 << "." << device.computeCapabilityMinor;
        LOG_INFO << "  Total Memory: " << (device.totalMemory / 1024 / 1024) << " MB";
    }

    LOG_INFO << "NPP Available: " << (hasNPP() ? "YES" : "NO");
    LOG_INFO << "NVJPEG Available: " << (hasNVJPEG() ? "YES" : "NO");
    LOG_INFO << "NVENC Available: " << (hasNVENC() ? "YES" : "NO");
    LOG_INFO << "NVDEC Available: " << (hasNVDEC() ? "YES" : "NO");
    LOG_INFO << "========================";
}
```

---

## Test Framework Integration

### Enhanced Preconditions

**File:** `base/test/nv_test_utils.h`

Update existing preconditions to use CudaCapabilities:

```cpp
#pragma once
#include <boost/test/unit_test.hpp>
#include "Logger.h"
#include "CudaCapabilities.h"

namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

struct if_cuda_available {
    tt::assertion_result operator()(utf::test_unit_id) {
        auto& cuda = CudaCapabilities::getInstance();
        if (!cuda.isAvailable()) {
            LOG_INFO << "Skipping test - no CUDA device available";
            return false;
        }
        return true;
    }
};

struct if_compute_cap_supported {
    tt::assertion_result operator()(utf::test_unit_id) {
        auto& cuda = CudaCapabilities::getInstance();
        if (!cuda.hasMinComputeCapability(5, 2)) {
            LOG_INFO << "Skipping test - requires compute capability >= 5.2";
            return false;
        }
        return true;
    }
};

struct if_h264_encoder_supported {
    tt::assertion_result operator()(utf::test_unit_id) {
        auto& cuda = CudaCapabilities::getInstance();
        if (!cuda.hasNVENC()) {
            LOG_INFO << "Skipping test - NVENC not available";
            return false;
        }
        return true;
    }
};

struct if_nvjpeg_supported {
    tt::assertion_result operator()(utf::test_unit_id) {
        auto& cuda = CudaCapabilities::getInstance();
        if (!cuda.hasNVJPEG()) {
            LOG_INFO << "Skipping test - NVJPEG not available";
            return false;
        }
        return true;
    }
};
```

---

## Workflow Strategy

### Disabling Existing Workflows

**DO THIS FIRST** before creating experimental workflows

**Method 1: Rename to disable (recommended for this branch)**
```bash
cd .github/workflows
mv CI-Linux-NoCUDA.yml CI-Linux-NoCUDA.yml.disabled
mv CI-Linux-CUDA.yml CI-Linux-CUDA.yml.disabled
mv CI-Linux-CUDA-Docker.yml CI-Linux-CUDA-Docker.yml.disabled
mv CI-Linux-CUDA-wsl.yml CI-Linux-CUDA-wsl.yml.disabled
mv CI-Win-NoCUDA.yml CI-Win-NoCUDA.yml.disabled
mv CI-Win-CUDA.yml CI-Win-CUDA.yml.disabled
mv CI-Linux-ARM64.yml CI-Linux-ARM64.yml.disabled
```

**Method 2: Conditional execution (if renaming not desired)**
Add to top of each workflow:
```yaml
on:
  workflow_dispatch:  # Only manual trigger
  # Comment out push/pull_request triggers
```

### Experimental Workflow Naming

All experimental workflows use prefix `experiment-XX-`:
- `experiment-01-cuda-install.yml`
- `experiment-02-vcpkg-opencv-cuda.yml`
- `experiment-03-cuda-capabilities.yml`
- `experiment-04-unified-linux.yml`

**Benefits:**
- Clear distinction from production workflows
- Easy to find and manage
- Won't trigger on normal push/PR

---

## Progress Tracking

### Experiment Results

Track results here as experiments complete:

#### Experiment 1: CUDA Install on GitHub Runner
- [ ] Status: Not Started
- [ ] Date Run:
- [ ] Result: PASS / FAIL
- [ ] Notes:
- [ ] Run ID:

#### Experiment 2: vcpkg OpenCV CUDA Build
- [ ] Status: Not Started
- [ ] Date Run:
- [ ] Result: PASS / FAIL
- [ ] Notes:
- [ ] Run ID:

#### Experiment 3: CudaCapabilities Implementation
- [ ] Status: Not Started
- [ ] Date Implemented:
- [ ] Compiles: YES / NO
- [ ] Tests Pass: YES / NO
- [ ] Notes:

#### Experiment 4: Unified Linux Workflow
- [ ] Status: Not Started
- [ ] Date Run:
- [ ] Build Job: PASS / FAIL
- [ ] Test Job (GH): PASS / FAIL
- [ ] Test Job (Self-hosted): PASS / FAIL
- [ ] Run ID:
- [ ] Notes:

---

## Decision Log

Track major decisions and their rationale:

### Decision 1: Linux First
**Date:** 2024-12-13
**Rationale:** Linux has simpler CUDA install, self-hosted runner available, fewer platform quirks
**Alternatives Considered:** Windows first (rejected - more complex)

### Decision 2: Feature Branch Isolation
**Date:** 2024-12-13
**Rationale:** High-risk architectural change, need validation before main merge
**Protection:** No PR merge without explicit approval

### Decision 3: Disable All Workflows Initially
**Date:** 2024-12-13
**Rationale:** Prevent accidental CI runs during experimentation, focus on controlled tests
**Method:** Rename `.yml` to `.yml.disabled`

---

## Rollback Plan

If experiments fail or issues discovered:

### Rollback Step 1: Restore Workflows
```bash
cd .github/workflows
mv CI-Linux-NoCUDA.yml.disabled CI-Linux-NoCUDA.yml
mv CI-Linux-CUDA.yml.disabled CI-Linux-CUDA.yml
# ... restore all workflows
git add .
git commit -m "Rollback: Restore original workflows"
git push
```

### Rollback Step 2: Abandon Branch
- Feature branch remains in repo for future reference
- Main branch unaffected
- No production impact

### Rollback Step 3: Document Lessons
- Update this document with failure reasons
- Document blockers for future attempts
- Preserve experiment workflows for reference

---

## Communication Protocol

### For Future Agents

When picking up this task:

1. **Read this document completely** before starting
2. **Check Progress Tracking section** for current state
3. **Check Decision Log** for context on past choices
4. **Update Progress Tracking** as you complete steps
5. **Document any blockers** in Decision Log
6. **DO NOT MERGE TO MAIN** without explicit approval

### Status Reporting Format

When reporting status to user:

```
## Status Update: [Date]

**Current Phase:** [Phase name]
**Last Completed:** [Task name]
**Currently Working On:** [Task name]
**Blockers:** [Any issues]

**Experiment Results:**
- Experiment 1: [PASS/FAIL/PENDING]
- Experiment 2: [PASS/FAIL/PENDING]
...

**Next Steps:**
1. [Next task]
2. [Following task]

**Estimated Completion:** [Phase name] by [date estimate if applicable]
```

---

## Success Criteria (Linux Phase)

Before moving to Windows/Mac, must achieve:

- [ ] Experiment 1: CUDA toolkit installs on ubuntu-latest
- [ ] Experiment 2: vcpkg builds opencv4[cuda] on ubuntu-latest
- [ ] Experiment 3: CudaCapabilities singleton works correctly
- [ ] Experiment 4: Unified workflow completes successfully
- [ ] All non-CUDA tests pass on GitHub runner
- [ ] All CUDA tests skip gracefully on GitHub runner (no crashes)
- [ ] All tests (CUDA + non-CUDA) pass on self-hosted runner
- [ ] Build time on GitHub runner < 2 hours
- [ ] No regressions compared to existing CI-Linux-CUDA workflow

**Only after ALL criteria met:** Consider Windows/Mac expansion

---

## File Manifest

Track all files modified/created in this branch:

### Documentation
- [ ] `.claude/docs/unified-cuda-build-implementation.md` (this file)

### Workflows (Experiments)
- [ ] `.github/workflows/experiment-01-cuda-install.yml`
- [ ] `.github/workflows/experiment-02-vcpkg-opencv-cuda.yml`
- [ ] `.github/workflows/experiment-03-cuda-capabilities.yml`
- [ ] `.github/workflows/experiment-04-unified-linux.yml`

### Workflows (Disabled)
- [ ] `.github/workflows/CI-Linux-NoCUDA.yml.disabled`
- [ ] `.github/workflows/CI-Linux-CUDA.yml.disabled`
- [ ] `.github/workflows/CI-Linux-CUDA-Docker.yml.disabled`
- [ ] `.github/workflows/CI-Linux-CUDA-wsl.yml.disabled`
- [ ] `.github/workflows/CI-Win-NoCUDA.yml.disabled`
- [ ] `.github/workflows/CI-Win-CUDA.yml.disabled`
- [ ] `.github/workflows/CI-Linux-ARM64.yml.disabled`

### Source Code
- [ ] `base/include/CudaCapabilities.h` (NEW)
- [ ] `base/src/CudaCapabilities.cpp` (NEW)
- [ ] `base/CMakeLists.txt` (MODIFY - add CudaCapabilities)
- [ ] `base/test/nv_test_utils.h` (MODIFY - use CudaCapabilities)

### Module Updates (Phase 2)
- [ ] `base/src/ResizeNPPI.cpp` (add capability check)
- [ ] `base/src/CCNPPI.cpp` (add capability check)
- [ ] ... (other modules)

---

## Known Issues / FAQ

### Q: Why not just use Docker for CUDA builds?
**A:** Docker adds complexity, doesn't solve the fundamental problem of duplicate builds. We want native builds on GitHub runners.

### Q: What if CUDA toolkit install fails on GitHub runner?
**A:** That's Experiment 1 - if it fails, we document the blocker and potentially pursue alternative strategies (e.g., pre-built containers with CUDA toolkit).

### Q: Can we use this for Windows too?
**A:** Yes, but AFTER proving it works on Linux. Windows has different CUDA install mechanisms (chocolatey vs apt).

### Q: What about vcpkg cache size with CUDA?
**A:** We'll monitor in experiments. CUDA-enabled packages are larger but we're eliminating duplicate caches (NoCUDA + CUDA), so net impact may be neutral.

### Q: How do we handle cuDNN?
**A:** cuDNN is required by opencv4[cudnn]. We'll need to install it alongside CUDA toolkit in experiments. This is part of Experiment 2 validation.

---

## References

- Original analysis document: `<in this conversation>`
- CMake CUDA documentation: https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cuda
- vcpkg CUDA ports: `vcpkg/ports/cuda/`
- Boost.Test preconditions: https://www.boost.org/doc/libs/1_84_0/libs/test/doc/html/boost_test/testing_tools/test_org.html

---

## Appendix A: CUDA Version Matrix

Current builds use different CUDA versions per platform:

| Platform | CUDA Version | Reason |
|----------|--------------|--------|
| Linux x64 (self-hosted) | 11.8 | Stable, good NPP support |
| Windows (self-hosted) | 11.8 | Match Linux |
| Jetson ARM64 | 10.2 (JetPack 4.x) | Fixed by Jetson OS |

For unified builds, we'll standardize on **CUDA 11.8** for Windows/Linux.

---

## Appendix B: Quick Start for New Agent

```bash
# 1. Checkout branch
git checkout feature/get-rid-of-nocuda-builds

# 2. Read this document completely
cat .claude/docs/unified-cuda-build-implementation.md

# 3. Check what's already done
# Look at Progress Tracking section

# 4. If starting fresh:
# - Disable workflows (rename to .disabled)
# - Create Experiment 1 workflow
# - Trigger manually via gh CLI
# - Document results in Progress Tracking

# 5. Update this document as you go

# 6. Commit frequently to feature branch
git add .
git commit -m "Progress: [describe what you did]"
git push origin feature/get-rid-of-nocuda-builds

# 7. NEVER merge to main without approval!
```

---

**Last Updated:** 2024-12-13
**Document Version:** 1.0
**Author:** Claude Code (Sonnet 4.5)
**Status:** Initial Draft - Awaiting Experiment Results
