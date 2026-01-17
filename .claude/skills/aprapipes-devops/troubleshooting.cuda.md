# CUDA Build Troubleshooting

Platform-agnostic troubleshooting for GPU test jobs on self-hosted runners.

**Scope**: GPU test jobs (CI-Windows cuda job, CI-Linux cuda job) running on self-hosted runners with NVIDIA GPUs.

**Note**: Cloud build jobs (CI-Windows build, CI-Linux build) also have CUDA toolkit installed for compilation, but don't run GPU tests. This guide covers only the GPU test execution on self-hosted runners.

**Platform-Specific Notes**:
- **Jetson ARM64 builds**: See `troubleshooting.jetson.md` for CI-Linux-ARM64 (different architecture - builds AND tests on same self-hosted runner)

---

## Self-Hosted vs Hosted Runners

### Key Differences

| Aspect | Cloud Runners (build jobs) | Self-Hosted (cuda jobs) |
|--------|----------------------------|-------------------------|
| Purpose | Build with CUDA toolkit | Run GPU tests only |
| Caching | Yes (vcpkg cache) | NOT needed (persistent disk) |
| Time Limit | 6 hours | None |
| Disk Space | Limited (~14 GB) | Depends on hardware |
| CUDA Toolkit | Installed (compilation) | Pre-installed (GPU required) |
| State Between Builds | Clean | Persistent (needs cleanup) |
| GPU Access | No | Yes (required) |

### Workflow Configuration

**New Architecture (Dec 2025)**:
- Cloud build jobs compile code with CUDA toolkit installed
- GPU test jobs (cuda) run via CI-CUDA-Tests.yml on self-hosted runners
- Self-hosted runners receive pre-built binaries from cloud build job
- Test results uploaded as TestResults_{flav}-CUDA artifacts

---

## Issue C1: CUDA_HOME / CUDA_PATH Not Set

**Symptom**:
```
CUDA_HOME is not set
CMake Error: Could not find CUDA toolkit
nvcc: command not found
```

**Root Cause**:
- Environment variables not set on self-hosted runner
- Required for CMake to find CUDA toolkit
- Required for nvcc compiler

**Expected Values**:
```bash
# Windows
CUDA_HOME=c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
CUDA_PATH=c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin

# Linux
CUDA_HOME=/usr/local/cuda-11.8
CUDA_PATH=/usr/local/cuda-11.8/bin
```

**Fix**:

Workflow should set these (check `.github/workflows/build-test-*.yml`):
```yaml
- name: Check builder for CUDA
  run: |
    echo "CUDA_HOME=c:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8" >> $GITHUB_ENV
    echo "CUDA_PATH=c:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin" >> $GITHUB_ENV
```

**Verification**:
```bash
# Windows
echo $env:CUDA_HOME
nvcc --version  # Should show v11.8

# Linux
echo $CUDA_HOME
nvcc --version  # Should show v11.8
```

---

## Issue C2: nvcc Not Found

**Symptom**:
```
nvcc: command not found
CUDA compiler not found
```

**Root Cause**:
- CUDA toolkit not installed on self-hosted runner
- nvcc not in PATH
- Wrong CUDA version installed

**Diagnostic Steps**:

1. Check nvcc exists:
   ```bash
   # Windows
   Test-Path "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe"

   # Linux
   ls /usr/local/cuda-11.8/bin/nvcc
   ```

2. Check version:
   ```bash
   nvcc --version
   # Expected: release 11.8
   ```

**Fix**:

If not installed, install CUDA 11.8 on self-hosted runner:
- **Windows**: Download from NVIDIA website, run installer
- **Linux**: Follow NVIDIA installation guide for your distro
- **Jetson**: Pre-installed with JetPack SDK

**If wrong version**: Update to CUDA 11.8 (required version)

---

## Issue C3: cuDNN Header Missing

**Symptom**:
```
cudnn.h: No such file or directory
Could not find cuDNN
```

**Root Cause**:
- cuDNN not installed
- cuDNN installed in wrong location
- Wrong cuDNN version for CUDA 11.8

**Diagnostic Steps**:

```bash
# Windows
Get-Item "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\cudnn.h"

# Linux
ls /usr/local/cuda-11.8/include/cudnn.h
```

**Fix**:

Install cuDNN for CUDA 11.8:
1. Download from NVIDIA (requires account)
2. Extract to CUDA installation directory
3. Verify cudnn.h exists in include/

**Verification**:
```bash
# Header exists
ls {CUDA_HOME}/include/cudnn.h

# Library exists
# Windows: cudnn.lib in lib/x64/
# Linux: libcudnn.so in lib64/
```

---

## Issue C4: OpenCV CUDA Features Build Failure

**Symptom**:
```
error: opencv4[cuda] build failed
CUDA architecture not supported
```

**Root Cause**:
- OpenCV trying to build for unsupported CUDA architecture
- GPU compute capability mismatch
- cuDNN version incompatible

**vcpkg.json Configuration**:
```json
{
  "name": "opencv4",
  "features": [
    "contrib",
    "cuda",
    "cudnn",
    "dnn",
    "jpeg",
    "nonfree",
    "png",
    "tiff",
    "webp"
  ]
}
```

**Fix**:

Check GPU compute capability and ensure OpenCV builds for it.

**To Be Expanded**: Add specific OpenCV CUDA build flags and troubleshooting steps.

---

## Issue C5: Whisper CUDA Compilation Issues

**Symptom**:
```
error: whisper[cuda] build failed
```

**vcpkg.json Configuration**:
```json
{
  "name": "whisper",
  "features": ["cuda"]
}
```

**To Be Expanded**: Document common whisper CUDA build issues.

---

## Issue C6: Self-Hosted Runner Cleanup

**Symptom**:
- Build fails with "file already exists"
- Previous build artifacts interfere
- Disk space issues on self-hosted runner

**Root Cause**:
- Self-hosted runners have persistent state
- vcpkg downloads accumulate
- Build directories not cleaned between runs

**Cleanup in Workflow** (check `.github/workflows/build-test-*.yml`):

```yaml
- name: Cleanup workspace on self hosted runners
  if: inputs.is-selfhosted
  run: 'Remove-Item -Recurse -Force *'
  shell: pwsh
  continue-on-error: true
```

**Manual Cleanup** (if needed):
```bash
# Windows
Remove-Item -Recurse -Force D:\a\ApraPipes\ApraPipes\*

# Linux
rm -rf /home/runner/work/ApraPipes/ApraPipes/*
```

---

## Issue C7: GPU Not Accessible

**Symptom**:
```
CUDA error: no CUDA-capable device is detected
GPU not found
```

**Diagnostic Steps**:

```bash
# Windows
nvidia-smi

# Linux
nvidia-smi

# Jetson
jetson_stats
```

**Fix**:

Ensure GPU is accessible:
- Check GPU driver installed
- Check GPU not in use by another process
- Restart runner if needed

---

## CUDA-Specific Quick Fixes Checklist

### Pre-Test Checklist (Self-Hosted Runner Setup)
- [ ] CUDA 11.8 toolkit installed
- [ ] cuDNN for CUDA 11.8 installed
- [ ] nvcc --version shows release 11.8
- [ ] cudnn.h exists in CUDA include directory
- [ ] GPU accessible (nvidia-smi works)
- [ ] CUDA_HOME and CUDA_PATH set correctly

### GPU Test Job Checklist
- [ ] Workflow calls CI-CUDA-Tests.yml reusable workflow
- [ ] Cleanup step runs at start
- [ ] Pre-built binaries received from cloud build job
- [ ] CUDA environment variables set in workflow
- [ ] GPU tests execute on self-hosted runner
- [ ] TestResults_{flav}-CUDA artifact uploaded

### Post-Test Checklist
- [ ] Workspace cleaned for next test run
- [ ] GPU memory released
- [ ] No lingering processes

---

## Platform-Specific CUDA Notes

### Windows CUDA
- Refer to `troubleshooting.windows.md` for Windows-specific issues
- CUDA paths use backslashes
- PowerShell-based scripts

### Linux CUDA
- Refer to `troubleshooting.linux.md` for Linux-specific issues
- CUDA paths use forward slashes
- Bash-based scripts

### Jetson CUDA
- Refer to `troubleshooting.jetson.md` for ARM64-specific issues
- JetPack SDK provides CUDA + cuDNN
- Memory constraints more severe

---

## Issue C8: ck() Macro Return Value Ignored

**Symptom**:
```
Memory access violation at address 0x3f8
Crash accessing NvDecoder methods after cuCtxCreate
```

**Root Cause**:
The `ck()` macro logs errors but does NOT throw exceptions - it returns `false`. If you ignore the return value, execution continues with invalid CUDA state.

**Bad Code**:
```cpp
// Continues with invalid cuContext if cuCtxCreate fails
ck(loader.cuCtxCreate(&cuContext, 0, cuDevice));
helper.reset(new NvDecoder(cuContext, ...));  // Crash later!
```

**Good Code**:
```cpp
// Throw on failure to prevent invalid state
if (!ck(loader.cuCtxCreate(&cuContext, 0, cuDevice))) {
    throw std::runtime_error("cuCtxCreate failed (possibly out of GPU memory)");
}
```

**Fix**: Always check `ck()` return value and throw exception on failure.

---

## Issue C9: CUDA Context Memory Leak

**Symptom**:
```
CUDA_ERROR_OUT_OF_MEMORY after creating/destroying multiple decoders
GPU OOM in tests that run late in the test suite
```

**Root Cause**:
CUDA contexts consume significant GPU memory. If `cuCtxDestroy()` is not called in destructors, memory accumulates until exhausted.

**Bad Code**:
```cpp
NvDecoder::~NvDecoder() {
    cuvidDestroyVideoParser(m_hParser);
    cuvidDestroyDecoder(m_hDecoder);
    // Missing: cuCtxDestroy(m_cuContext)!
}
```

**Good Code**:
```cpp
NvDecoder::~NvDecoder() {
    cuvidDestroyVideoParser(m_hParser);
    cuvidDestroyDecoder(m_hDecoder);
    if (m_cuContext && loader.cuCtxDestroy) {
        loader.cuCtxDestroy(m_cuContext);
        m_cuContext = nullptr;
    }
}
```

**Fix**: Always destroy CUDA contexts in destructors.

---

## Issue C10: GPU OOM from Multiple Context Creation

**Symptom**:
```
CUDA_ERROR_OUT_OF_MEMORY when creating contexts
Tests fail late in test suite (e.g., h264decoder_tests runs last)
```

**Root Cause**:
Each `cuCtxCreate` allocates GPU memory. When running many tests sequentially, memory accumulates even with proper destruction due to overlapping lifetimes.

**Bad Code**:
```cpp
CUcontext cuContext;
cuCtxCreate(&cuContext, 0, cuDevice);  // Creates new context each time
// ... use context ...
cuCtxDestroy(cuContext);  // Too late if many instances created
```

**Good Code**:
```cpp
CUcontext cuContext;
cuDevicePrimaryCtxRetain(&cuContext, cuDevice);  // Reference-counted, shared
m_ownedDevice = cuDevice;  // Store device for release
// ... use context ...
cuDevicePrimaryCtxRelease(m_ownedDevice);  // Just decrements refcount
```

**Fix**: Use primary context API (`cuDevicePrimaryCtxRetain/Release`) instead of `cuCtxCreate/Destroy`. This matches the pattern in `ApraCUcontext` in `CudaCommon.h`.

**Fixed File**: `H264DecoderNvCodecHelper.cpp`

---

## To Be Expanded

This guide will be expanded as CUDA-specific issues are encountered:
- OpenCV CUDA build flags and architecture configuration
- Whisper CUDA compilation issues
- CUDA compute capability troubleshooting
- Multi-GPU configurations
- CUDA version upgrade procedures

**Cross-Platform Issues**: See `troubleshooting.windows.md` and `troubleshooting.linux.md` for non-CUDA issues that may also affect CUDA builds.

---

**Applies to**: GPU test jobs (CI-Windows cuda, CI-Linux cuda) on self-hosted runners
**Related Guides**: reference.md, troubleshooting.windows.md (cloud builds), troubleshooting.linux.md (cloud builds), troubleshooting.jetson.md (ARM64 self-hosted)
