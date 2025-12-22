# CI/CD Learnings — Institutional Memory

> This file survives /clear. Read it at session start. Append new learnings.
> Don't repeat mistakes documented here.

## Entry Format
```
### YYYY-MM-DD | workflow-name | PASS/FAIL
**Tried:** What was attempted
**Error:** Error message (if any)
**Root cause:** Why it happened
**Fix:** What resolved it
**Rule:** One-line principle for future
```

---

## Entries

(Append new learnings below this line)

### 2025-12-21 | CI-Linux-ARM64 | FAIL (CRITICAL INFRASTRUCTURE ISSUE)
**Tried:** Run workflow on Jetson ARM64 self-hosted runner
**Error:** `GLIBC_2.28' not found (required by /mnt/disks/actions-runner/externals/node20/bin/node)`
**Root cause:**
- Jetson runs Ubuntu 18.04 (GLIBC 2.27), but GitHub Actions Node 20 requires GLIBC 2.28+
- GitHub Actions Node16 reached END OF LIFE on November 12, 2024
- Runner 2.330.0 no longer includes Node16, only Node20
- Main passed on Dec 16 with runner 2.329.0 (last version with Node16)
- Ubuntu 18.04 is now UNSUPPORTED by GitHub Actions
**Fix options:**
1. **Upgrade Jetson to Ubuntu 20.04+** (recommended - has GLIBC 2.28+)
2. Pin runner to 2.329.0 or earlier (not recommended - no security updates)
3. Use Docker container for build (wraps workflow in container with newer GLIBC)
**Rule:** Ubuntu 18.04 self-hosted runners are no longer supported by GitHub Actions as of Nov 2024

### 2025-12-21 | ACTIONS_RUNNER_FORCED_INTERNAL_NODE_VERSION | FAIL
**Tried:** Set `ACTIONS_RUNNER_FORCED_INTERNAL_NODE_VERSION=node16` as job env var
**Error:** Runner still uses Node 20, ignores env var
**Root cause:** This env var must be set at the RUNNER SERVICE level, not workflow level. Also, Node16 was removed from runner 2.330.0 entirely.
**Fix:** This workaround no longer works as of runner 2.330.0
**Rule:** Workflow-level env vars cannot override runner's internal Node version

### 2025-12-21 | Release-only triplets | LESSON
**Tried:** Implement release-only vcpkg triplets to reduce build times
**Error:** CMake couldn't find baresip library - `find_library(BARESIP_LIB NAMES libbaresip.so ...)` failed
**Root cause:** Release-only triplets build static libraries (`.a`) by default, not shared (`.so`)
**Fix:** Updated find_library to accept both: `NAMES libbaresip.so libbaresip.a baresip`
**Rule:** When using release-only triplets, expect static libraries - update find_library calls accordingly

### 2025-12-21 | CI-Linux-ARM64 | FAIL (DISK SPACE ISSUE)
**Tried:** Run ARM64 build on new Jetson (Ubuntu 20.04) after fixing GLIBC issue
**Error:** `fatal: write error: No space left on device` during vcpkg registry fetch
**Root cause:**
- Jetson has two filesystems: `/` (14GB, 100% full) and `/data` (117GB, 5% used)
- vcpkg binary cache was correctly on `/data/.cache/vcpkg`
- BUT vcpkg registries cache defaults to `~/.cache/vcpkg/registries` which is on root `/`
- Root filesystem only had 34MB free, vcpkg registry fetch failed
**Fix:**
1. Added `XDG_CACHE_HOME=/data/.cache` env var for self-hosted runners
2. This redirects ALL caches (including vcpkg registries) to the large `/data` partition
3. Added cleanup step to remove old `~/.cache/vcpkg/registries` before builds
**Rule:** For embedded devices with small root filesystems, set `XDG_CACHE_HOME` to redirect all caches to larger partitions

### 2025-12-21 | CI-Linux-ARM64 | FAIL (GCC VERSION MISMATCH)
**Tried:** Run ARM64 build after XDG_CACHE_HOME fix
**Error:** `vcpkg was unable to detect the active compiler's information` + `CMAKE_C_COMPILER not set`
**Root cause:**
- Workflow has "Set GCC-11" step that hardcodes `/usr/bin/gcc-11` for CUDA builds
- This is needed for x64 CUDA 11.8 builds (Ubuntu 24.04 has GCC 13, but CUDA 11.8 needs GCC <= 11)
- Jetson has CUDA 11.4 which works fine with GCC 9.4 (the default)
- Jetson doesn't have GCC-11 installed, so setting CC=/usr/bin/gcc-11 breaks the build
**Fix:** Made GCC-11 step conditional - only set CC/CXX if `/usr/bin/gcc-11` exists
**Rule:** When setting compiler overrides, always check if the compiler exists first to support different platforms

### 2025-12-21 | CI-Linux-ARM64 | FAIL (STATIC LINKING -ldl)
**Tried:** Build baresip with release-only triplet (arm64-linux-release)
**Error:** `undefined reference to symbol 'dlsym@@GLIBC_2.17'` when linking baresip
**Root cause:**
- `libre.a` static library uses `dlsym` for dynamic loading
- When linking statically, `-ldl` must be explicitly added
- The baresip patch set LINKLIBS but didn't include `${CMAKE_DL_LIBS}`
- This wasn't caught with shared library builds (.so) because they auto-resolve dependencies
**Fix:** Added `${CMAKE_DL_LIBS}` to LINKLIBS in fix-static-re-linking.patch
**Rule:** When switching from shared to static libraries, always check for missing `-ldl`, `-lpthread`, `-lm` etc.

### 2025-12-22 | CI-Linux-ARM64 | FAIL (LIBPNG PATH MISMATCH)
**Tried:** Build opencv4 with release-only triplet (arm64-linux-release)
**Error:** `png.h: No such file or directory` looking for `include/libpng/png.h`
**Root cause:**
- **NOT cache corruption** - vcpkg's libpng installs headers to `include/libpng16/` not `include/libpng/`
- OpenCV's CHECK_INCLUDE_FILE expects `${PNG_PNG_INCLUDE_DIR}/libpng/png.h`
- Docker builds work because system `libpng-dev` is installed (which has symlink `/usr/include/libpng -> libpng16`)
- Self-hosted runners without system libpng-dev fail
**Fix:** Created libpng overlay port that adds symlink: `include/libpng -> libpng16`
**Rule:** vcpkg libpng installs to libpng16/ but opencv expects libpng/ - create symlink for compatibility

### 2025-12-22 | CI-Linux-ARM64 | FAIL (MISSING CUDA LIBRARIES)
**Tried:** Build opencv4 with CUDA support on Jetson ARM64
**Error:** `CUDA_npps_LIBRARY-NOTFOUND`, `CUDA_cublas_LIBRARY-NOTFOUND`, `CUDA_cufft_LIBRARY-NOTFOUND`
**Root cause:**
- Jetson had minimal CUDA install (only cuda-cudart, cuda-nvcc)
- NPP (NVIDIA Performance Primitives), cuBLAS, and cuFFT libraries were NOT installed
- OpenCV4 with CUDA support requires these libraries for image processing accelerations
- These packages are NOT installed by default with JetPack's minimal CUDA setup
**Fix:** Installed the missing CUDA development packages:
```bash
sudo apt-get install -y libnpp-dev-11-4 libcublas-dev-11-4 libcufft-dev-11-4
```
**Rule:** For OpenCV CUDA builds on Jetson, ensure libnpp-dev, libcublas-dev, and libcufft-dev are installed

### 2025-12-22 | CI-Linux-ARM64 | FAIL (PKG_CONFIG_PATH VERSION MISMATCH)
**Tried:** ApraPipes CMake configure after opencv4 build succeeded
**Error:** `Package 'gio-2.0' has version '2.64.6', required version is '>= 2.82'` (and similar for harfbuzz, pixman)
**Root cause:**
- vcpkg built GTK3 with dependencies on newer library versions
- CMakeLists.txt had hardcoded PKG_CONFIG_PATH to system directories only: `/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig`
- System packages (Ubuntu 20.04) have older versions than what vcpkg GTK3 requires
- The vcpkg-built libraries have their own .pc files with correct versions in `vcpkg_installed/<triplet>/lib/pkgconfig`
**Fix:** Prepend vcpkg's pkgconfig path to PKG_CONFIG_PATH:
```cmake
set(VCPKG_PKGCONFIG_PATH "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/pkgconfig")
set(ENV{PKG_CONFIG_PATH} "${VCPKG_PKGCONFIG_PATH}:/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig")
```
**Rule:** When using vcpkg on ARM64, prepend vcpkg's pkgconfig path to PKG_CONFIG_PATH for consistent library versions

---