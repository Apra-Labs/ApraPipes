# CUDA Build Troubleshooting

Platform-agnostic troubleshooting for all CUDA builds on self-hosted runners.

**Scope**: ALL CUDA builds (Windows CUDA, Linux CUDA, Jetson) running on self-hosted runners.

**Platform-Specific Notes**:
- **Jetson ARM64 builds**: Also see `troubleshooting.jetson.md` for JetPack 5.x requirements (gcc-9.4, CUDA 11.4, multimedia API changes)

**Critical**: Self-hosted runners behave completely differently from GitHub-hosted runners.

---

## Self-Hosted vs Hosted Runners

### Key Differences

| Aspect | GitHub-Hosted | Self-Hosted (CUDA) |
|--------|---------------|---------------------|
| Caching | Required (two-phase) | NOT needed (persistent disk) |
| Time Limit | 1 hour/phase | None |
| Disk Space | Limited (~14 GB) | Depends on hardware |
| CUDA Toolkit | Not installed | Pre-installed (required) |
| State Between Builds | Clean | Persistent (needs cleanup) |
| vcpkg downloads | Cleared each run | Accumulates |

### Workflow Configuration

**is-selfhosted: true** skips:
- Cache save/restore steps
- Disk cleanup steps
- Two-phase build (runs single-phase)

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

### Pre-Build Checklist (Self-Hosted Runner Setup)
- [ ] CUDA 11.8 toolkit installed
- [ ] cuDNN for CUDA 11.8 installed
- [ ] nvcc --version shows release 11.8
- [ ] cudnn.h exists in CUDA include directory
- [ ] GPU accessible (nvidia-smi works)
- [ ] CUDA_HOME and CUDA_PATH set correctly

### Build Checklist
- [ ] workflow has `is-selfhosted: true`
- [ ] Cleanup step runs at start
- [ ] CUDA environment variables set in workflow
- [ ] opencv4[cuda] and whisper[cuda] features enabled
- [ ] No caching steps (not needed for self-hosted)

### Post-Build Checklist
- [ ] Workspace cleaned for next build
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

## To Be Expanded

This guide will be expanded as CUDA-specific issues are encountered:
- OpenCV CUDA build flags and architecture configuration
- Whisper CUDA compilation issues
- CUDA compute capability troubleshooting
- GPU memory issues and profiling
- Multi-GPU configurations
- CUDA version upgrade procedures

**Cross-Platform Issues**: See `troubleshooting.windows.md` and `troubleshooting.linux.md` for non-CUDA issues that may also affect CUDA builds.

---

**Applies to**: All CUDA builds (Windows, Linux, Jetson) on self-hosted runners
**Related Guides**: reference.md, troubleshooting.windows.md, troubleshooting.linux.md, troubleshooting.jetson.md
