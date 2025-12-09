# Jetson/ARM64 Build Troubleshooting

Platform-specific troubleshooting for Jetson ARM64 builds with CUDA.

**Scope**: Jetson devices (ARM64 + CUDA), most complex platform combination.

**Related**: See `troubleshooting.cuda.md` for CUDA-specific issues that apply here.

---

## Jetson-Specific Architecture

### Build Configuration
- **Platform**: ARM64 (aarch64)
- **CUDA**: Yes (provided by JetPack SDK)
- **Runner**: Self-hosted (Jetson device)
- **Constraints**: Memory limited, package availability limited

### Key Characteristics
- ARM64 + CUDA combination (unique constraints)
- JetPack SDK provides CUDA + cuDNN optimized for Jetson
- Many vcpkg packages not available for ARM64
- Cross-compilation vs native build considerations

### JetPack 4.x Toolchain Requirements

**CRITICAL**: Jetson uses older toolchain due to JetPack 4.x compatibility. These constraints are ONLY for Jetson - all other platforms use modern tooling.

| Component | Jetson (JetPack 4.x) | Other Platforms | Reason |
|-----------|----------------------|-----------------|--------|
| **Compiler** | gcc-8 / g++-8 | gcc-11 | JetPack 4.x requirement |
| **curl** | 7.58.0 + HTTP/1.1 | 8.x + HTTP/2 | System curl has HTTP/2 framing bugs |
| **OpenCV** | 4.8.0 | 4.10.0 | JetPack 4.x compatibility |

**Configuration in CMakeLists.txt:**
```cmake
IF(ENABLE_ARM64)
  set(CMAKE_C_COMPILER /usr/bin/gcc-8)
  set(CMAKE_CXX_COMPILER /usr/bin/g++-8)
ENDIF()
```

**HTTP/1.1 Workaround:**
```bash
# System curl 7.58.0 has HTTP/2 framing layer bugs
# Force HTTP/1.1 via .curlrc
echo "http1.1" > ~/.curlrc
```

---

## Issue J1: ARM64 Packages Not Available in vcpkg

**Symptom**:
```
error: package '<name>' is not available for arm64-linux
error: no portfile for <package> on ARM64
```

**Root Cause**:
- Many vcpkg packages don't support ARM64
- Package may not have ARM64 triplet
- Package may require x64-specific dependencies

**Known Exclusions**:
```json
{
  "name": "hiredis",
  "platform": "!arm64"
},
{
  "name": "redis-plus-plus",
  "platform": "!arm64"
}
```

**Fix**:

1. Check if package has ARM64 support in vcpkg
2. If not, exclude with platform filter: `"platform": "!arm64"`
3. Consider alternative packages or system libraries

---

## Issue J2: VCPKG_FORCE_SYSTEM_BINARIES Required

**Symptom**:
```
error: vcpkg requires system binaries on ARM64
CMAKE_CROSSCOMPILING is true but tools not found
```

**Root Cause**:
- vcpkg needs to use system binaries on ARM64
- Can't use pre-built vcpkg binaries (x64-only)

**Fix**:

Set environment variable (check `build-test-linux.yml`):
```yaml
- name: Configure for ARM64
  if: ${{ contains(inputs.flav, 'arm64') }}
  run: |
    echo "VCPKG_FORCE_SYSTEM_BINARIES=1" >> $GITHUB_ENV
```

Or in CMakeLists.txt:
```cmake
IF(ENABLE_ARM64)
  set(ENV{VCPKG_FORCE_SYSTEM_BINARIES} 1)
ENDIF()
```

---

## Issue J3: JetPack SDK Version Mismatch

**Symptom**:
```
CUDA version mismatch
cuDNN version incompatible
```

**Root Cause**:
- JetPack SDK provides specific CUDA/cuDNN versions
- vcpkg packages may expect different versions

**Diagnostic Steps**:

```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Check CUDA version
nvcc --version

# Check cuDNN version
cat /usr/include/aarch64-linux-gnu/cudnn_version.h | grep CUDNN_MAJOR
```

**Fix**:

Ensure vcpkg packages compatible with JetPack CUDA/cuDNN versions.

**To Be Expanded**: Document JetPack version compatibility matrix.

---

## Issue J4: Memory Constraints on Jetson

**Symptom**:
```
c++: fatal error: Killed signal terminated program
make: *** [build] Error 1
```

**Root Cause**:
- Jetson devices have limited RAM (4-8 GB typical)
- Large packages (OpenCV, Boost) can exhaust memory during compilation
- Parallel builds may use too many threads

**Fix**:

1. Reduce parallel build threads in workflow:
   ```yaml
   nProc: 2  # Instead of 6 for hosted runners
   ```

2. Increase swap space on Jetson:
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. Build large packages sequentially (not in parallel)

---

## Issue J5: glib Platform Filter for ARM64

**Symptom**:
```
error: glib requires different features on ARM64
```

**Platform Filters**:
```json
{
  "name": "glib",
  "default-features": false,
  "features": ["libmount"],
  "platform": "(linux & x64)",
  "$reason": "skip linux:arm64 and windows"
}
```

**Note**: glib on ARM64 may need different configuration.

**To Be Expanded**: Document correct glib configuration for Jetson/ARM64.

---

## Jetson-Specific Quick Fixes Checklist

### Pre-Build Checklist (Jetson Setup)
- [ ] JetPack SDK installed
- [ ] nvcc --version shows correct CUDA version
- [ ] cuDNN version matches JetPack
- [ ] Swap space configured (8 GB recommended)
- [ ] VCPKG_FORCE_SYSTEM_BINARIES set

### vcpkg.json Checklist
- [ ] ARM64-incompatible packages excluded (`"platform": "!arm64"`)
- [ ] glib platform filter correct for ARM64
- [ ] OpenCV ARM64-compatible features selected
- [ ] CUDA features compatible with JetPack CUDA version

### Build Checklist
- [ ] Parallel build threads reduced (nProc: 2)
- [ ] Memory usage monitored during build
- [ ] Large packages build without OOM

---

## Cross-Compilation vs Native Build

**Native Build** (Recommended for Jetson):
- Build directly on Jetson device
- Simpler configuration
- Slower but more reliable

**Cross-Compilation** (Advanced):
- Build on x64 host for ARM64 target
- Faster but complex
- Requires ARM64 cross-compiler toolchain

**To Be Expanded**: Document cross-compilation setup if needed.

---

## To Be Expanded

This guide will be expanded as Jetson-specific issues are encountered:
- JetPack version compatibility matrix
- ARM64 package availability in vcpkg
- Cross-compilation procedures
- Memory optimization strategies
- Jetson-specific OpenCV CUDA configurations
- Performance profiling on Jetson

**CUDA Issues**: See `troubleshooting.cuda.md` for CUDA-related issues (nvcc, cuDNN, GPU access) that apply to Jetson.

---

**Applies to**: Jetson ARM64 builds with CUDA
**Related Guides**: reference.md, troubleshooting.cuda.md, troubleshooting.linux.md, methodology.md
