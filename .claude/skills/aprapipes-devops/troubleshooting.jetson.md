# Jetson/ARM64 Build Troubleshooting

Platform-specific troubleshooting for Jetson ARM64 builds with CUDA.

**Scope**: CI-Linux-ARM64 workflow running on self-hosted Jetson devices (ARM64 + CUDA).

**Related**: See `troubleshooting.cuda.md` for general CUDA concepts (note: Jetson architecture differs - builds AND tests on same runner).

---

## Jetson-Specific Architecture

### Build Configuration
- **Workflow**: CI-Linux-ARM64.yml
- **Job**: ci (single job for build and test)
- **Platform**: ARM64 (aarch64)
- **CUDA**: Yes (provided by JetPack SDK)
- **Runner**: Self-hosted (Jetson device)
- **Constraints**: Memory limited, disk space limited, package availability limited

### Workflow Files
- **Top-level**: `.github/workflows/CI-Linux-ARM64.yml`
- **Reusable**: `.github/workflows/build-test-lin.yml`

### Key Characteristics
- ARM64 + CUDA combination (unique constraints)
- JetPack SDK provides CUDA + cuDNN optimized for Jetson
- Many vcpkg packages not available for ARM64
- Embedded device with split filesystems (small `/` root, large `/data`)
- Builds AND tests on same self-hosted runner (unlike x64 CUDA which splits cloud build + GPU test)

---

## JetPack Version Compatibility Matrix

| JetPack | L4T | Ubuntu | CUDA | cuDNN | GCC | GLIBC | Notes |
|---------|-----|--------|------|-------|-----|-------|-------|
| 5.0 | 35.1 | 20.04 | 11.4 | 8.6 | 9.4 | 2.31 | **Recommended minimum** |
| 5.1+ | 35.x | 20.04 | 11.4+ | 8.6+ | 9.4 | 2.31 | Current |
| 6.0+ | 36.x | 22.04 | 12.x | 8.9+ | 11+ | 2.35 | Latest |

**Note**: JetPack 4.x (Ubuntu 18.04) is **not supported** as of December 2025. GitHub Actions requires GLIBC 2.28+ for Node 20.

---

## Issue J1: Disk Space Exhaustion on Embedded Devices

**Symptom**:
```
fatal: write error: No space left on device
```

**Root Cause**:
- Jetson often has split filesystems: small root `/` (14GB), large data `/data` (100GB+)
- vcpkg binary cache may be on `/data`, but registry cache defaults to `~/.cache/vcpkg/registries`
- Build artifacts, apt cache, pip cache accumulate on root

**Fix - Redirect All Caches**:
```bash
# Set XDG_CACHE_HOME to redirect ALL caches to large partition
export XDG_CACHE_HOME=/data/.cache
```

In GitHub Actions workflow:
```yaml
env:
  XDG_CACHE_HOME: ${{ startsWith(inputs.cache-path, '/data') && '/data/.cache' || '' }}
```

**Cleanup Commands**:
```bash
rm -rf ~/.cache/vcpkg/registries  # Old registry cache
sudo apt-get clean                 # APT cache
rm -rf ~/.cache/pip               # Pip cache
```

**Invariant**: On embedded devices, always verify which filesystem has free space and redirect caches accordingly.

---

## Issue J2: GCC Version Mismatch Across Platforms

**Symptom**:
```
vcpkg was unable to detect the active compiler's information
CMAKE_C_COMPILER not set
```

**Root Cause**:
- x64 CUDA 11.8 builds require GCC 11 (Ubuntu 24.04 default is GCC 13)
- Jetson CUDA 11.4 works fine with GCC 9.4 (default)
- Hardcoded `/usr/bin/gcc-11` breaks on systems without it

**Fix - Conditional Compiler Selection**:
```yaml
- name: Set GCC-11 for CUDA compatibility (x64 only)
  if: ${{ contains(inputs.cuda,'ON') }}
  run: |
    if [ -x /usr/bin/gcc-11 ]; then
      echo "CC=/usr/bin/gcc-11" >> $GITHUB_ENV
      echo "CXX=/usr/bin/g++-11" >> $GITHUB_ENV
    else
      echo "Using default compiler"
    fi
```

**Invariant**: When setting compiler overrides, ALWAYS check if the compiler exists first.

---

## Issue J3: Static vs Shared Library Linking

**Symptom**:
```
undefined reference to symbol 'dlsym@@GLIBC_2.17'
undefined reference to `pthread_create'
```

**Root Cause**:
- Release-only vcpkg triplets (e.g., `arm64-linux-release`) build static libraries (`.a`)
- Static libraries don't auto-resolve dependencies like shared libraries (`.so`)
- Libraries using `dlopen`/`dlsym` need explicit `-ldl`
- Multi-threaded code needs explicit `-lpthread`

**Fix - Explicit Link Libraries**:
```cmake
# In vcpkg patches or CMakeLists.txt
target_link_libraries(mylib PRIVATE
    ${CMAKE_DL_LIBS}      # -ldl
    ${CMAKE_THREAD_LIBS}  # -lpthread
    m                     # -lm (math)
)
```

**Fix - Update find_library to Accept Both**:
```cmake
# Accept both shared and static variants
find_library(MYLIB NAMES libfoo.so libfoo.a foo REQUIRED)
```

**Invariant**: When switching from shared to static libraries, explicitly add `-ldl`, `-lpthread`, `-lm`.

---

## Issue J4: vcpkg libpng vs OpenCV Header Path Mismatch

**Symptom**:
```
png.h: No such file or directory
looking for include/libpng/png.h
```

**Root Cause**:
- vcpkg installs libpng headers to `include/libpng16/`
- OpenCV's `CHECK_INCLUDE_FILE` expects `include/libpng/png.h`
- System `libpng-dev` creates symlink, vcpkg does not

**Fix - Create libpng Overlay Port**:
```cmake
# In portfile.cmake, after vcpkg_cmake_install():
if(NOT VCPKG_BUILD_TYPE OR VCPKG_BUILD_TYPE STREQUAL "release")
    if(EXISTS "${CURRENT_PACKAGES_DIR}/include/libpng16" AND NOT EXISTS "${CURRENT_PACKAGES_DIR}/include/libpng")
        file(CREATE_LINK "libpng16" "${CURRENT_PACKAGES_DIR}/include/libpng" SYMBOLIC)
    endif()
endif()
```

**Invariant**: vcpkg libpng uses `libpng16/` but many packages expect `libpng/` - create symlink for compatibility.

---

## Issue J5: Missing CUDA Libraries on Minimal JetPack Install

**Symptom**:
```
CUDA_npps_LIBRARY-NOTFOUND
CUDA_cublas_LIBRARY-NOTFOUND
CUDA_cufft_LIBRARY-NOTFOUND
```

**Root Cause**:
- JetPack minimal install only includes `cuda-cudart` and `cuda-nvcc`
- OpenCV CUDA requires NPP, cuBLAS, cuFFT for image processing acceleration
- These are NOT installed by default

**Fix - Install CUDA Development Packages**:
```bash
# For CUDA 11.4 (JetPack 5.x)
sudo apt-get install -y \
    libnpp-dev-11-4 \
    libcublas-dev-11-4 \
    libcufft-dev-11-4

# Verify installation
ls /usr/local/cuda/lib64/libnpp*.so
```

**Invariant**: For OpenCV CUDA builds on Jetson, always install: libnpp-dev, libcublas-dev, libcufft-dev.

---

## Issue J6: PKG_CONFIG_PATH Version Mismatch

**Symptom**:
```
Package 'gio-2.0' has version '2.64.6', required version is '>= 2.82'
Package 'harfbuzz' has version '2.6.4', required version is '>= 2.7.4'
```

**Root Cause**:
- vcpkg builds GTK3 with dependencies on newer library versions
- CMake's `pkg_check_modules` searches system paths first
- System Ubuntu 20.04 packages are older than what vcpkg GTK3 expects
- vcpkg-built libraries have correct versions in `vcpkg_installed/<triplet>/lib/pkgconfig`

**Fix - Prepend vcpkg pkgconfig Path**:
```cmake
# In CMakeLists.txt
set(VCPKG_PKGCONFIG_PATH "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/pkgconfig")
set(ENV{PKG_CONFIG_PATH} "${VCPKG_PKGCONFIG_PATH}:$ENV{PKG_CONFIG_PATH}")
```

**Invariant**: When mixing vcpkg and system packages, PREPEND vcpkg's pkgconfig path to ensure consistent versions.

---

## Issue J7: JetPack 5.x Multimedia API Breaking Changes

**Symptom**:
```
Could not find NVBUFUTILSLIB using the following names: nvbuf_utils
error: nvbuf_utils.h: No such file or directory
```

**Root Cause**:
JetPack 5.0 renamed/replaced multimedia libraries from JetPack 4.x. See `knowledge_docs/jetson-jp5.md` for full API migration details.

**Fix - Accept Both Library Names in CMake**:
```cmake
# Accept either library name
find_library(NVBUFUTILSLIB NAMES nvbufsurface nvbuf_utils REQUIRED)

# Make removed libraries optional
find_library(EGLSTREAM_CAMCONSUMER_LIB nveglstream_camconsumer)
if(EGLSTREAM_CAMCONSUMER_LIB)
    list(APPEND JETSON_LIBS ${EGLSTREAM_CAMCONSUMER_LIB})
endif()
```

**Invariant**: Use NAMES keyword in find_library to accept both old and new library names.

**Technical Reference**: See `knowledge_docs/jetson-jp5.md` for NvBuffer→NvBufSurface API migration, compatibility layer implementation, and V4L2 mmap handling.

---

## Issue J8: CUDA Runtime Libraries Not in ldconfig

**Symptom**:
```
libnppicc.so.11 => not found
libnppidei.so.11 => not found
```

**Root Cause**:
- CUDA libraries exist at `/usr/local/cuda/targets/aarch64-linux/lib/`
- This path is NOT in ldconfig by default on JetPack 5.0
- Build succeeds (CMake finds libs), but runtime fails

**Fix - Add CUDA Path to ldconfig**:
```bash
sudo sh -c 'echo /usr/local/cuda/targets/aarch64-linux/lib > /etc/ld.so.conf.d/cuda-targets.conf'
sudo ldconfig
```

**Verify**:
```bash
ldconfig -p | grep npp
```

**Invariant**: On JetPack 5.0+, CUDA libs in `/usr/local/cuda/targets/aarch64-linux/lib` must be added to ldconfig.

---

## Issue J9: Headless Display for GTK/EGL Tests

**Symptom**:
- Tests hang indefinitely
- No test output generated
- Tests killed after timeout

**Root Cause**:
- GTK/EGL tests require a display for initialization
- Headless Jetson (no monitor attached) blocks waiting for display
- Tests never complete, producing no output

**Fix - Virtual Display with Xvfb**:
```bash
# Install Xvfb
sudo apt-get install -y xvfb

# Run tests with virtual display
xvfb-run -a --server-args="-screen 0 1920x1080x24" ./test_executable
```

In GitHub Actions:
```yaml
- name: Run Tests
  run: |
    if command -v xvfb-run &> /dev/null; then
      xvfb-run -a --server-args="-screen 0 1920x1080x24" ./tests
    else
      ./tests
    fi
```

**Invariant**: For headless Jetson runners, install Xvfb and use xvfb-run for GTK/EGL tests.

---

## Jetson Runner Setup Checklist

### Initial Setup (One-time)
- [ ] JetPack 5.0+ installed (Ubuntu 20.04, GLIBC 2.28+)
- [ ] GitHub Actions runner 2.330.0+ installed
- [ ] CUDA toolkit with development packages:
  ```bash
  sudo apt-get install -y libnpp-dev-11-4 libcublas-dev-11-4 libcufft-dev-11-4
  ```
- [ ] CUDA libraries in ldconfig:
  ```bash
  sudo sh -c 'echo /usr/local/cuda/targets/aarch64-linux/lib > /etc/ld.so.conf.d/cuda-targets.conf'
  sudo ldconfig
  ```
- [ ] Xvfb for headless testing:
  ```bash
  sudo apt-get install -y xvfb
  ```
- [ ] Swap space configured:
  ```bash
  sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  ```

### Disk Space Management
- [ ] Identify large partition for caches (typically `/data`)
- [ ] Set `XDG_CACHE_HOME=/data/.cache` in runner environment
- [ ] Set `VCPKG_DEFAULT_BINARY_CACHE=/data/.cache/vcpkg`
- [ ] Clean old caches periodically

### Diagnostic Commands
```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Check CUDA version
nvcc --version

# Check available disk space
df -h / /data

# Check GLIBC version
ldd --version

# Verify CUDA libraries in ldconfig
ldconfig -p | grep -E "(npp|cublas|cufft)"

# Check runner version
~/actions-runner/run.sh --version
```

---

## vcpkg ARM64 Configuration

### Triplet Settings
```cmake
# arm64-linux-release.cmake (custom triplet)
set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)  # Release-only builds use static
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_BUILD_TYPE release)      # Skip debug builds
```

### Required Environment Variables
```bash
export VCPKG_FORCE_SYSTEM_BINARIES=1  # Required for ARM64
export VCPKG_MAX_CONCURRENCY=4         # Limit parallelism (memory constraints)
```

### Platform Exclusions in vcpkg.json
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

---

## Common Error Quick Reference

| Error | Cause | Fix |
|-------|-------|-----|
| `No space left on device` | Root filesystem full | Set `XDG_CACHE_HOME=/data/.cache` |
| `nvbuf_utils not found` | JetPack 5.x API change | Use `NAMES nvbufsurface nvbuf_utils` |
| `libnpp*.so not found` | Missing ldconfig | Add CUDA path to ldconfig |
| Tests hang | Headless display | Use xvfb-run |
| `dlsym undefined` | Static linking | Add `${CMAKE_DL_LIBS}` |
| `libpng/png.h not found` | vcpkg path | Create libpng overlay with symlink |
| `gio-2.0 version mismatch` | pkg-config order | Prepend vcpkg pkgconfig path |

---

**Applies to**: CI-Linux-ARM64 workflow (self-hosted Jetson with JetPack 5.0+)
**Related Guides**: reference.md, troubleshooting.cuda.md (general CUDA concepts), troubleshooting.linux.md (Linux-specific patterns), methodology.md
