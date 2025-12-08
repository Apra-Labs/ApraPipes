# NVIDIA CUDA & Video Codec SDK Compatibility Guide

## Purpose
This document provides compatibility information between NVIDIA CUDA Toolkit, Video Codec SDK, and driver requirements for ApraPipes project. Use this for making decisions about version upgrades and compatibility issues.

---

## Table of Contents
- [Overview](#overview)
- [Current Project Setup](#current-project-setup)
- [Understanding the Components](#understanding-the-components)
- [Compatibility Matrix](#compatibility-matrix)
- [Build vs Runtime Considerations](#build-vs-runtime-considerations)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Upgrade Recommendations](#upgrade-recommendations)
- [References](#references)

---

## Overview

ApraPipes uses NVIDIA CUDA for GPU acceleration and NVIDIA Video Codec SDK for hardware-accelerated video encoding/decoding. Understanding the compatibility between these components and the NVIDIA driver is critical for both build success and runtime stability.

### Key Components
1. **CUDA Toolkit** - Provides GPU compute capabilities
2. **Video Codec SDK** - Provides hardware video encode (NVENC) and decode (NVDEC) APIs
3. **NVIDIA Driver** - Provides runtime implementation of both CUDA and Video Codec features

---

## Current Project Setup

### CUDA Toolkit
- **Version**: 11.8.0
- **Release Date**: September 2022
- **Min Driver Required**: 450.80.02 (Linux) / 452.39 (Windows)
- **Docker Image**: `nvidia/cuda:11.8.0-devel-ubuntu22.04`

### Video Codec SDK
- **Version**: 10.0.26
- **Release Date**: July 2020
- **Designed For**: CUDA 10.0
- **Min Driver Required**: 435.21 (Linux) / 436.15 (Windows)
- **Location**: `thirdparty/Video_Codec_SDK_10.0.26/`

### OpenCV
- **Version**: 4.10.0 (upgraded from 4.8.0 for CMake 3.29 compatibility)
- **CUDA Support**: Enabled
- **Features**: contrib, cuda, cudnn, dnn, jpeg, nonfree, png, tiff, webp

### CMake
- **Min Version**: 3.29
- **Reason**: Required for project features
- **Note**: CMake 3.27+ removed deprecated FindCUDA module (uses FindCUDAToolkit instead)

---

## Understanding the Components

### CUDA Toolkit vs Video Codec SDK

**These are SEPARATE products from NVIDIA:**

| Feature | CUDA Toolkit | Video Codec SDK |
|---------|--------------|-----------------|
| **Purpose** | General GPU computing | Video encode/decode acceleration |
| **APIs** | CUDA Runtime, cuBLAS, cuDNN, NPP, etc. | NVENC (encode), NVDEC (decode) |
| **Stub Libraries Location** | `/usr/local/cuda/lib64/stubs/` | **NOT in CUDA toolkit** |
| **Versioning** | Independent | Independent |
| **Required For** | GPU compute tasks | Video encoding/decoding |

**Critical:** Video Codec SDK libraries (`libnvcuvid.so`, `libnvidia-encode.so`) are **NOT** part of CUDA Toolkit!

### Stub Libraries vs Runtime Libraries

#### Stub Libraries (Compile/Link Time)
- **Purpose**: Allow compilation without requiring NVIDIA driver
- **Location**:
  - CUDA: `/usr/local/cuda/lib64/stubs/`
  - Video Codec SDK: `thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64/`
- **Size**: Small placeholder files
- **Contains**: Function signatures only, no implementation
- **Used By**: Linker during build
- **Used When**: Docker builds, development systems without GPUs

#### Runtime Libraries (Execution Time)
- **Purpose**: Provide actual GPU acceleration functionality
- **Location**: `/usr/lib/x86_64-linux-gnu/` (on Linux with NVIDIA driver)
- **Source**: NVIDIA Driver installation
- **Size**: Full implementation (several MB)
- **Contains**: Complete GPU driver code
- **Used By**: Running application
- **Used When**: Production systems with GPUs

**Key Insight:** Stub libraries are **NEVER** used at runtime! They are purely for linking.

---

## Compatibility Matrix

### Video Codec SDK Version History

| SDK Version | Release Date | CUDA Version | Min Driver (Linux) | Min Driver (Windows) | Notes |
|-------------|--------------|--------------|-------------------|---------------------|-------|
| 10.0.26 | July 2020 | CUDA 10.0 | 435.21 | 436.15 | Current in ApraPipes |
| 11.0.x | Oct 2020 | CUDA 10.2+ | 450.80 | 456.71 | Added GA100 support |
| 11.1.x | Apr 2021 | CUDA 11.0+ | 470.57 | 471.41 | New NVENC presets |
| 12.0.x | Sept 2022 | CUDA 11.0+ | 522.25 | 522.06 | CUDA 11/12 compatible |
| 12.1.x | May 2023 | CUDA 11.4+ | 530.30 | 531.14 | AV1 improvements |
| 12.2.x | Oct 2023 | CUDA 11.4+ | 535.54 | 536.23 | Latest (as of 2024) |

### CUDA Toolkit Version History

| CUDA Version | Release Date | Min Driver (Linux) | Min Driver (Windows) |
|--------------|--------------|-------------------|---------------------|
| 10.0 | Sept 2018 | 410.48 | 411.31 |
| 10.2 | Nov 2019 | 440.33 | 441.22 |
| 11.0 | May 2020 | 450.36 | 451.22 |
| 11.8 | Sept 2022 | 450.80.02 | 452.39 |
| 12.0 | Dec 2022 | 525.60.13 | 527.41 |
| 12.2 | Aug 2023 | 535.54.03 | 536.67 |

### Driver Backward Compatibility

**Important:** NVIDIA drivers are generally backward compatible within a major version family.

- Driver 450.80+ supports CUDA 11.0 through 11.8
- Driver 525.60+ supports CUDA 11.x AND 12.x
- Driver 535.54+ supports CUDA 11.4+ and 12.x with additional features

---

## Build vs Runtime Considerations

### Docker Build Environment (GitHub Actions)

**Environment:**
- Container: `nvidia/cuda:11.8.0-devel-ubuntu22.04`
- No NVIDIA driver present
- No GPU hardware

**What's Available:**
- ✅ CUDA Toolkit headers and libraries
- ✅ CUDA stub libraries in `/usr/local/cuda/lib64/stubs/`
- ❌ Video Codec SDK libraries (NOT in CUDA toolkit)
- ❌ Runtime driver libraries

**Build Requirements:**
- Must use Video Codec SDK stub libraries from `thirdparty/Video_Codec_SDK_10.0.26/`
- CMake finds stubs for linking
- Build succeeds even without GPU/driver

**CMake Configuration (Linux x86_64):**
```cmake
find_library(LIBNVCUVID libnvcuvid.so
    PATHS
        /usr/lib/x86_64-linux-gnu              # Runtime driver libs (bare metal)
        ../thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64  # Stubs (Docker)
    NO_DEFAULT_PATH
)
```

### Bare Metal Environment (Self-Hosted Runners)

**Environment:**
- Physical hardware with NVIDIA GPU
- NVIDIA driver installed
- CUDA Toolkit installed

**What's Available:**
- ✅ CUDA Toolkit (same as Docker)
- ✅ NVIDIA Driver runtime libraries in `/usr/lib/x86_64-linux-gnu/`
- ✅ Actual GPU hardware

**Build Requirements:**
- CMake finds runtime driver libraries first
- No need for stubs (but harmless if present)
- Build uses actual driver libs

**Runtime Requirements:**
- Driver version must satisfy BOTH:
  - CUDA 11.8 requirement: ≥ 450.80.02
  - Video Codec SDK 10.0.26 requirement: ≥ 435.21
- **Effective minimum**: 450.80.02

### ARM64 Jetson Environment

**Environment:**
- Jetson AGX Xavier / Orin
- Jetpack with CUDA and drivers pre-installed
- ARM64 architecture

**What's Different:**
- Uses V4L2 hardware encoding (not NVENC/NVDEC)
- `NVCODEC_LIB` is set to **EMPTY**
- Video Codec SDK libraries are **NOT USED**
- Different code path in CMakeLists.txt (lines 81-127)

**CMake Note:**
```cmake
IF(ENABLE_ARM64)
    SET(NVCODEC_LIB)  # Empty! ARM64 doesn't use Video Codec SDK
    # Uses Jetson multimedia API instead
```

---

## Build vs Runtime Compatibility

### Scenario 1: SDK 10.0.26 + CUDA 11.8 (Current Setup)

#### Build Time: ✅ WORKS
- Stub libraries allow linking
- No driver required
- Build succeeds in Docker

#### Runtime Compatibility: ⚠️ DEPENDS ON DRIVER

**If Driver is 450.80 - 522.24:**
- ✅ CUDA 11.8 supported
- ⚠️ Video Codec SDK 10.0.26 uses old APIs
- ⚠️ Untested combination (SDK designed for CUDA 10.0)
- ⚠️ **Risk**: API version mismatch, deprecated calls
- ✅ **Likely works** due to backward compatibility

**If Driver is 522.25+:**
- ✅ CUDA 11.8 supported
- ✅ Video Codec SDK 12.0+ officially supported
- ⚠️ SDK 10.0.26 is 2 years older than driver
- ⚠️ **Risk**: Using old SDK with very new driver
- ✅ **Should work** but not optimal

**If Driver is < 450.80:**
- ❌ CUDA 11.8 **NOT** supported
- ⚠️ Video Codec SDK 10.0.26 might work (needs 435.21+)
- ❌ **Incompatible** - must upgrade driver

### Scenario 2: SDK 12.0+ + CUDA 11.8 (Recommended Future)

#### Build Time: ✅ WORKS
- Newer SDK stub libraries
- Same build process

#### Runtime Compatibility: ✅ BETTER

**If Driver is 522.25+:**
- ✅ Perfect match
- ✅ Officially supported combination
- ✅ Modern APIs
- ✅ Best performance and compatibility

**If Driver is 450.80 - 522.24:**
- ⚠️ Driver too old for SDK 12.0
- ❌ Must upgrade driver OR keep SDK 10.0.26

---

## Troubleshooting Guide

### Build Issues

#### Problem: `LIBNVCUVID` or `LIBNVENCODE` set to NOTFOUND

**Cause:** CMake cannot find Video Codec SDK libraries

**Solution:** Ensure CMakeLists.txt searches stub libraries:
```cmake
find_library(LIBNVCUVID libnvcuvid.so
    PATHS
        /usr/lib/x86_64-linux-gnu  # For bare metal
        ../thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64  # For Docker
    NO_DEFAULT_PATH
)
```

**Verify stub files exist:**
```bash
ls -la thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64/
# Should show: libnvcuvid.so, libnvcuvid.so.1, libnvidia-encode.so, libnvidia-encode.so.1
```

**Note:** Files may be Git LFS pointers (129 bytes). Run `git lfs pull` to fetch actual files.

#### Problem: CMake Error about FindCUDA.cmake not found

**Cause:** CMake 3.27+ removed deprecated FindCUDA module

**Solution:**
- OpenCV 4.8.0 still uses old FindCUDA → upgrade to OpenCV 4.10.0+
- Or downgrade CMake to 3.26 (not recommended)

**Background:** OpenCV 4.10.0+ migrated to FindCUDAToolkit (modern approach)

### Runtime Issues

#### Problem: Application fails with "cannot open shared object file"

**Error:**
```
error while loading shared libraries: libnvcuvid.so.1: cannot open shared object file
```

**Cause:** NVIDIA driver not installed or wrong driver version

**Solution:**
```bash
# Check if driver is installed
nvidia-smi

# Check driver version
cat /proc/driver/nvidia/version

# Verify libraries exist
ls -la /usr/lib/x86_64-linux-gnu/libnvcuvid*
ls -la /usr/lib/x86_64-linux-gnu/libnvidia-encode*
```

**If missing:** Install/upgrade NVIDIA driver

#### Problem: NVENC/NVDEC initialization fails at runtime

**Symptoms:**
- Build succeeds
- Application starts
- NVENC/NVDEC operations fail with errors

**Debug Steps:**
1. Check driver version meets minimum requirements
2. Verify GPU supports NVENC/NVDEC (not all GPUs do)
3. Check for API version mismatches in logs
4. Consider upgrading Video Codec SDK

**Driver Version Check:**
```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

**GPU Capability Check:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

---

## Upgrade Recommendations

### Current State Assessment

**Strengths:**
- ✅ CUDA 11.8 is modern and well-supported
- ✅ OpenCV 4.10.0 compatible with CMake 3.29
- ✅ Build process works in Docker

**Weaknesses:**
- ⚠️ Video Codec SDK 10.0.26 is 4+ years old
- ⚠️ SDK designed for CUDA 10.0, not CUDA 11.8
- ⚠️ Untested compatibility combination
- ⚠️ Missing modern Video Codec features

### Recommended Upgrade Path

#### Phase 1: Immediate (Current)
**Status:** ✅ Complete
- Fixed Docker build to use SDK stub libraries
- Build succeeds with SDK 10.0.26 + CUDA 11.8

**Action:** Test runtime on self-hosted runners with actual GPU

#### Phase 2: Short-term Validation (Before Production)
**Priority:** HIGH
**Timeline:** Before next production deployment

**Tasks:**
1. **Check driver version on production systems:**
   ```bash
   nvidia-smi --query-gpu=driver_version --format=csv,noheader
   ```

2. **Run comprehensive Video Codec tests:**
   - H.264 encoding/decoding
   - Different resolutions and bitrates
   - Verify no errors/warnings in logs

3. **Document findings:**
   - Driver version that works
   - Any issues encountered
   - Performance baseline

**Decision Point:**
- If all tests pass → Can proceed with SDK 10.0.26 for now
- If issues found → Expedite SDK upgrade (Phase 3)

#### Phase 3: Medium-term Upgrade (Recommended)
**Priority:** MEDIUM
**Timeline:** Next development sprint

**Goal:** Upgrade to Video Codec SDK 12.2

**Benefits:**
- ✅ Official CUDA 11.8 support
- ✅ Modern APIs and features
- ✅ Better driver compatibility
- ✅ Future-proof for years
- ✅ Bug fixes and improvements

**Challenges:**
- ⚠️ API changes may require code updates
- ⚠️ May require driver upgrade (522.25+)
- ⚠️ Testing effort required

**Steps:**
1. Download Video Codec SDK 12.2 from NVIDIA
2. Place in `thirdparty/Video_Codec_SDK_12.2/`
3. Update CMakeLists.txt paths
4. Review API changes (check migration guide)
5. Update code if needed
6. Test thoroughly
7. Update this document with new version info

### Alternative: Keep SDK 10.0.26

**When to Choose:**
- Video Codec features are not critical
- Runtime testing shows no issues
- Cannot upgrade drivers in production
- Development resources limited

**Mitigation:**
- Document minimum driver version (450.80.02)
- Add runtime version checks in code
- Monitor for issues in production
- Plan upgrade when resources available

---

## CMake Configuration Details

### Current Implementation (base/CMakeLists.txt)

#### For Linux x86_64 (ENABLE_LINUX):
```cmake
ELSEIF(ENABLE_LINUX)
    # Try to find NVIDIA Video Codec libraries in multiple locations:
    # 1. System driver libraries (on bare metal with NVIDIA drivers)
    # 2. Video Codec SDK stub libraries (in thirdparty - for Docker/development)
    # Note: These are NOT part of CUDA toolkit, they're from Video Codec SDK
    find_library(LIBNVCUVID libnvcuvid.so
        PATHS
            /usr/lib/x86_64-linux-gnu
            ../thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64
        NO_DEFAULT_PATH
    )
    find_library(LIBNVENCODE libnvidia-encode.so
        PATHS
            /usr/lib/x86_64-linux-gnu
            ../thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64
        NO_DEFAULT_PATH
    )
    SET(NVCODEC_LIB ${LIBNVCUVID} ${LIBNVENCODE})
```

**Search Order:**
1. `/usr/lib/x86_64-linux-gnu` - Checked first (bare metal)
2. `../thirdparty/Video_Codec_SDK_10.0.26/...` - Fallback (Docker)

**Behavior:**
- Bare metal: Finds driver libs first, uses those (optimal)
- Docker: Fallback to stub libs (allows build)

#### For ARM64 Jetson (ENABLE_ARM64):
```cmake
IF(ENABLE_ARM64)
    SET(NVCODEC_LIB)  # Set to EMPTY
    # Jetson uses V4L2 hardware encoding, not NVENC/NVDEC
    # Different libraries: nvv4l2, nvbuf_utils, etc.
```

**Note:** ARM64 code path is completely separate, not affected by Video Codec SDK changes.

---

## Testing Checklist

### Before Merging CUDA/SDK Changes

#### Build Tests:
- [ ] Docker build succeeds (GitHub Actions)
- [ ] Bare metal build succeeds (self-hosted runner)
- [ ] ARM64 build succeeds (Jetson)
- [ ] All platforms with CUDA=ON
- [ ] All platforms with CUDA=OFF (for non-CUDA builds)

#### Runtime Tests (on systems with GPU):
- [ ] Check driver version: `nvidia-smi`
- [ ] Run unit tests with NVENC
- [ ] Run unit tests with NVDEC
- [ ] Test H.264 encoding at various resolutions
- [ ] Test H.264 decoding
- [ ] Check for memory leaks
- [ ] Monitor GPU utilization
- [ ] Verify no error messages in logs

#### Compatibility Tests:
- [ ] Test on minimum supported driver (450.80.02)
- [ ] Test on current production driver
- [ ] Test on latest driver (if different)
- [ ] Document any driver-specific issues

---

## References

### Official Documentation
- [NVIDIA Video Codec SDK Download](https://developer.nvidia.com/nvidia-video-codec-sdk/download)
- [Video Codec SDK Archive](https://developer.nvidia.com/video-codec-sdk-archive)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)

### Version-Specific Docs
- [Video Codec SDK 10.0 Docs](https://docs.nvidia.com/video-technologies/video-codec-sdk/10.0/)
- [Video Codec SDK 12.2 Docs](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.2/)
- [CUDA 11.8 Release Notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html)

### Related ApraPipes Documentation
- `knowledge_docs/Mp4Writer_module.md` - Uses NVENC for encoding
- `knowledge_docs/H.264_decoder_module.md` - Uses NVDEC for decoding
- `.github/workflows/CI-Linux-CUDA-Docker.yml` - Docker build configuration
- `.github/workflows/CI-Linux-CUDA.yml` - Bare metal build configuration

---

## Document Maintenance

**Last Updated:** December 2025

**Review Schedule:**
- When upgrading CUDA Toolkit version
- When upgrading Video Codec SDK version
- When upgrading OpenCV version
- When minimum driver version changes
- Annually for general review

**Maintainer Notes:**
- Keep compatibility matrix updated with tested combinations
- Document any runtime issues discovered in production
- Update recommendations based on real-world experience
- Add new troubleshooting scenarios as they arise
