# vcpkg Triplet Configuration Guide

This document explains the vcpkg triplet configuration used across different platforms in ApraPipes CI/CD.

## Overview

All platforms use release-only triplets (`VCPKG_BUILD_TYPE release`) to avoid building Debug packages unnecessarily, saving significant build time.

## Platform-Specific Triplets

| Platform | Triplet | Library Linkage | Notes |
|----------|---------|-----------------|-------|
| Linux x64 | `x64-linux-release` | static | Standard vcpkg triplet |
| ARM64 | `arm64-linux-release` | static | Standard vcpkg triplet |
| macOS | `x64-osx-release` | static | Standard vcpkg triplet |
| Docker | `x64-linux-release` | static | Same as Linux x64 |
| Windows | `x64-windows-cuda` | **dynamic** | Custom triplet (see below) |

## Windows Dynamic Linkage Rationale

Windows uses dynamic library linkage (`VCPKG_LIBRARY_LINKAGE dynamic`) instead of static for a specific reason:

### DELAYLOAD for CUDA DLLs

The unified build creates executables compiled with CUDA support that can run on systems without NVIDIA drivers. This is achieved using Windows DELAYLOAD:

```cmake
# From base/CMakeLists.txt
set(CUDA_DELAYLOAD_DLLS
    nvjpeg64_11.dll
    nppig64_11.dll
    nppicc64_11.dll
    nppidei64_11.dll
    nppial64_11.dll
    nppc64_11.dll
    cublas64_11.dll
    cublasLt64_11.dll
    cudart64_110.dll
    nvcuvid.dll
    nvEncodeAPI64.dll
)
```

DELAYLOAD only works with DLLs (dynamic libraries). Static linking would embed CUDA symbols directly into the executable, causing immediate load failures on systems without NVIDIA drivers.

### Custom Triplet Location

The custom triplet is located at:
```
vcpkg/triplets/community/x64-windows-cuda.cmake
```

Key settings:
- `VCPKG_LIBRARY_LINKAGE dynamic` - Required for DELAYLOAD
- `VCPKG_PLATFORM_TOOLSET "v142"` - VS 2019 toolset (CUDA 11.8 compatibility)
- `VCPKG_BUILD_TYPE release` - No Debug builds
- `CUDA_ARCH_BIN=7.5;8.6` - Limit CUDA architectures to reduce build time

## CUDA Architecture Optimization

OpenCV CUDA builds are time-consuming because they compile CUDA kernels for many GPU architectures. We limit this to common cloud GPU architectures:

- **7.5**: Turing (T4, commonly used in cloud)
- **8.6**: Ampere (A10, A30)

This is configured via:
- Linux: `-DCUDA_ARCH_BIN="7.5;8.6"` in cmake configure
- Windows: Set in `x64-windows-cuda.cmake` triplet

## Adding a New Platform

To add a new platform with release-only builds:

1. Use an existing `*-release` triplet if available (check `vcpkg/triplets/community/`)
2. Pass triplet to cmake:
   ```yaml
   cmake -DVCPKG_TARGET_TRIPLET=<triplet> -DVCPKG_HOST_TRIPLET=<triplet> ...
   ```
3. For reusable workflows, add `vcpkg-triplet` input parameter

## Troubleshooting

### vcpkg building Debug packages
- Ensure triplet has `set(VCPKG_BUILD_TYPE release)`
- Verify triplet is passed to cmake with both `VCPKG_TARGET_TRIPLET` and `VCPKG_HOST_TRIPLET`

### Windows exe crashes on non-GPU system
- All CUDA DLLs must be in DELAYLOAD list
- Check for missing DLLs with Dependency Walker or dumpbin
