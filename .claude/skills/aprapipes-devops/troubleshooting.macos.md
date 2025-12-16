# macOS CI/CD Troubleshooting Guide

## macOS-Specific Build Issues

### Issue: libjpeg Undefined Symbols on GitHub Actions

**Symptom**:
```
Undefined symbols for architecture x86_64:
  "_jpeg_default_qtables", referenced from:
      _tjInitCompress in libturbojpeg.a(turbojpeg.c.o)
ld: warning: no platform load command found in libjpeg.a[3](jfdctflt-sse.asm.o)
```

**Affected Platform**: macOS-15-intel GitHub Actions runners

**Root Cause**:
- libjpeg-turbo 3.1.2 (vcpkg's current version) has NASM-built assembly objects that lack proper Mach-O platform load commands
- GitHub's macOS-15-intel runner has stricter linker requirements than local macOS builds
- The linker fails when linking OpenCV's JPEG encoder with these incomplete NASM objects

**Solution**:
Pin libjpeg-turbo to version 3.0.4 using `vcpkg-configuration.json` with custom registry:

```json
{
  "$schema": "https://raw.githubusercontent.com/microsoft/vcpkg-tool/main/docs/vcpkg-configuration.schema.json",
  "default-registry": {
    "kind": "git",
    "repository": "https://github.com/microsoft/vcpkg",
    "baseline": "3011303ba1f6586e8558a312d0543271fca072c6"
  },
  "registries": [
    {
      "kind": "git",
      "repository": "https://github.com/microsoft/vcpkg",
      "baseline": "20d1b778772da35c42c7729be82ad8dcf40b1e88",
      "packages": [
        "libjpeg-turbo"
      ]
    }
  ]
}
```

**How to Find the Baseline**:
```bash
# Find vcpkg commit with working libjpeg-turbo version
cd vcpkg
git log --all --oneline -- ports/libjpeg-turbo/vcpkg.json
git show <commit>:ports/libjpeg-turbo/vcpkg.json  # Check version field
```

**Testing**:
- Local builds may work with both 3.1.2 and 3.0.4 (macOS 15.7.2+)
- CI builds on GitHub Actions runners require 3.0.4
- Always test on actual CI runners, not just locally

**References**:
- Commit: ee2455c3b126d4d186c95967e31a41bb14d7570e
- GitHub Actions run: 20118166178 (failed with 3.1.2)

---

## vcpkg Cache Management Issues

### Issue: Failed Builds Waste 2+ Hours Rebuilding All Packages

**Symptom**:
- Every failed build starts vcpkg installation from scratch
- Takes ~2 hours to install 50+ dependencies
- Iterating on fixes is painfully slow

**Root Cause**:
- GitHub Actions cache was only saved on successful builds
- Failed builds (which are common during debugging) didn't save their vcpkg packages
- Used `actions/cache@v3` with hit/miss logic instead of separate restore/save

**Solution**:
Split cache operations into separate restore and save steps with `always()` condition:

```yaml
- name: Restore vcpkg cache
  id: cache-restore
  if: ${{ !inputs.is-selfhosted }}
  uses: actions/cache/restore@v3
  with:
      path: |
        ${{ inputs.cache-path }}
      key: ${{ inputs.flav }}-5-${{ hashFiles('base/vcpkg.json', 'base/vcpkg-configuration.json', 'submodule_ver.txt') }}
      restore-keys: ${{ inputs.flav }}-

- name: Configure CMake Common
  working-directory: ${{github.workspace}}/build
  run: '${{ inputs.cmake-conf-cmd }} -DCMAKE_TOOLCHAIN_FILE=${{env.CMAKE_TC_FILE}} ...'

- name: Save vcpkg cache (even if build fails later)
  if: ${{ always() && !inputs.is-selfhosted }}
  uses: actions/cache/save@v3
  with:
    path: |
      ${{ inputs.cache-path }}
    key: ${{ inputs.flav }}-5-${{ hashFiles('base/vcpkg.json', 'base/vcpkg-configuration.json', 'submodule_ver.txt') }}
```

**Key Insights**:
- GitHub Actions **automatically replaces** old cache when saving with same key
- No need to manually delete old cache using `gh cache delete`
- `always()` condition ensures save runs even if build fails later
- Cache is saved after CMake configure (when vcpkg install completes)

**Benefits**:
- First build: Downloads ~50+ packages, takes ~2 hours
- Second build (after failure): Reuses cached packages, much faster
- Incremental progress: Each run adds more packages to cache
- Debugging iterations become practical (minutes instead of hours)

**References**:
- Commit: 58e25a2075900a7c0bb8acdf44e669f00fec5052
- Similar pattern used in CI-Linux-CUDA-Docker workflow

---

## CMake Platform Configuration Issues

### Issue: ARM64 Build Failing After macOS Support Added

**Symptom**:
```
CMake Error at /usr/share/cmake/Modules/FindPkgConfig.cmake:634 (message):
  The following required packages were not found:
   - gdk-3.0

Package 'x11', required by 'gdk-3.0', not found
Package 'xext', required by 'gdk-3.0', not found
```

**Affected Platform**: ARM64 (Jetson) self-hosted builder

**Root Cause**:
- During macOS support refactoring, two `IF(ENABLE_LINUX)` blocks were consolidated
- This moved `pkg_check_modules(GDK3 REQUIRED gdk-3.0)` calls BEFORE `ARM64 PKG_CONFIG_PATH` setup
- ARM64 builds need `PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig` set BEFORE running pkg_check_modules
- Without correct PKG_CONFIG_PATH, pkg-config can't find the system libraries

**Debugging Approach**:
```bash
# ALWAYS compare with working version first
git diff main..HEAD -- base/CMakeLists.txt

# Look for what changed in the ordering of:
# 1. IF(ENABLE_LINUX) blocks
# 2. IF(ENABLE_ARM64) PKG_CONFIG_PATH setup
# 3. pkg_check_modules() calls
```

**Solution**:
Restore original two-block structure from main branch:

```cmake
# Block 1: Early Linux dependencies (don't need PKG_CONFIG_PATH)
IF(ENABLE_LINUX)
	find_package(GLEW REQUIRED)
	find_package(glfw3 CONFIG REQUIRED)
	find_package(FreeGLUT CONFIG REQUIRED)
	pkg_check_modules(GIO REQUIRED gio-2.0)
	pkg_check_modules(GOBJECT REQUIRED gobject-2.0)
	pkg_check_modules(GLFW REQUIRED glfw3)
ENDIF()

IF(ENABLE_MACOS)
	find_package(GLEW REQUIRED)
	find_package(glfw3 CONFIG REQUIRED)
	find_package(GLUT REQUIRED)
	# Skip GTK3/GDK/GIO/GOBJECT - not used in source code
ENDIF()

# ARM64 setup MUST come before GDK3/GTK3 checks
IF(ENABLE_ARM64)
	set(ENV{PKG_CONFIG_PATH} "/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig")
ENDIF(ENABLE_ARM64)

# Block 2: Late Linux dependencies (need PKG_CONFIG_PATH for ARM64)
IF(ENABLE_LINUX)
	pkg_check_modules(GDK3 REQUIRED gdk-3.0)
	pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
ENDIF()
```

**Multi-Platform Constraints**:
This structure balances three platform requirements:
1. **macOS**: GTK3 variables referenced in CMakeLists.txt:546-551 but can be empty (CMake tolerates undefined variables)
2. **ARM64**: PKG_CONFIG_PATH must be set BEFORE pkg_check_modules for GDK3/GTK3
3. **Linux x64**: Needs actual GTK3 libraries installed and found

**Why Consolidation Broke ARM64**:
- The "cleaner looking" consolidated IF(ENABLE_LINUX) block was NOT needed for macOS
- CMake tolerates `${GTK3_INCLUDE_DIRS}` being undefined (becomes empty string)
- Consolidation seemed harmless but violated ARM64's PKG_CONFIG_PATH ordering requirement
- This is why diff-based debugging is CRITICAL: working main branch had separate blocks

**Verification**:
```bash
# Compare structure with main branch
git diff main -- base/CMakeLists.txt

# Verify PKG_CONFIG_PATH comes before GDK3/GTK3 checks
grep -A5 "ENABLE_ARM64" base/CMakeLists.txt
grep -A3 "pkg_check_modules(GDK3" base/CMakeLists.txt
```

**References**:
- Commit: aaec46a0d41077d1ecb7dd8c8c395e4958446ebe
- ARM64 build run: 20140987796 (failed before fix)

---

## Workflow Triggering Issues

### Issue: Empty Commits Don't Trigger CI Workflows

**Symptom**:
- Created empty commit to trigger CI retry: `git commit --allow-empty -m "ci: retry"`
- Pushed successfully, but no GitHub Actions workflows triggered
- Waited 10+ minutes, still no runs

**Root Cause**:
- GitHub Actions workflows with `paths-ignore` configuration won't trigger on empty commits
- Even if no files match the ignore pattern, empty commits are treated specially
- This is undocumented behavior but confirmed through testing

**Wrong Approach**:
```bash
# ❌ WRONG: Empty commit to trigger build
git commit --allow-empty -m "ci: trigger build"
git push
```

**Correct Approach**:
```bash
# ✅ CORRECT: Manual trigger using gh CLI
gh workflow run CI-MacOSX-NoCUDA.yml --ref feat/ci-macos-new
```

**How to Know If a Commit Will Trigger**:
1. Check the workflow's `on:` section in the .yml file
2. Look for `paths-ignore:` configuration
3. If your changes match the ignore pattern OR commit is empty, use `gh workflow run`
4. Never create commits just to trigger builds

**Workflow Configuration Check**:
```bash
# Read the workflow trigger configuration
grep -A10 "^on:" .github/workflows/CI-MacOSX-NoCUDA.yml

# Look for paths-ignore patterns
grep -A5 "paths-ignore:" .github/workflows/CI-MacOSX-NoCUDA.yml
```

**References**:
- Commit: dab05aee0 (the ineffective empty commit - should be removed)
- Documented in methodology.md lines 77-90

---

## General macOS Build Best Practices

### Local Validation Before CI

**NEVER push fixes blindly to GitHub Actions**. Always validate locally first:

```bash
# 1. Clean build directory
rm -rf build && mkdir build && cd build

# 2. Configure with same flags as CI
cmake -B . -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DENABLE_MACOS=ON \
  -DENABLE_CUDA=OFF \
  -DENABLE_LINUX=OFF \
  -DENABLE_WINDOWS=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  ../base

# 3. Build
cmake --build . --config RelWithDebInfo -j 3

# 4. Run tests
./aprapipesut --log_level=test_suite
```

### Comparing Local vs CI Environments

**Local macOS** (developer machine):
- macOS 15.7.2+ with Xcode 16.2+
- May have additional system libraries installed via Homebrew
- More forgiving linker (accepts NASM objects without full Mach-O metadata)

**GitHub Actions macos-15-intel**:
- Clean runner image, minimal system libraries
- Stricter linker requirements
- May have different CPU architecture optimizations

**Key Differences**:
- A build working locally does NOT guarantee CI success
- Always test on actual CI runners for final verification
- Use manual workflow triggers for testing, not empty commits

### macOS Platform Dependencies

**Required Homebrew packages**:
```bash
brew install nasm yasm ninja cmake pkg-config automake autoconf \
  autoconf-archive libtool dos2unix gperf bison python3
```

**Not needed** (handled by vcpkg):
- GTK3/GDK3 (Linux-only in our build)
- GLEW, GLFW, FreeGLUT (vcpkg provides these)

**Native macOS frameworks**:
- GLUT (native macOS, not FreeGLUT)
- Accelerate framework (for BLAS/LAPACK)

---

## Quick Reference: macOS Troubleshooting Commands

```bash
# Check what symbols are missing in a library
nm -g libfoo.a | grep jpeg

# Inspect Mach-O load commands (check for platform commands)
otool -l libfoo.a

# Check library dependencies
otool -L ./aprapipesut

# Find which vcpkg package provides a file
vcpkg owns <file>

# Check vcpkg package version
vcpkg list | grep libjpeg

# Force vcpkg to rebuild a specific package
vcpkg remove libjpeg-turbo
vcpkg install libjpeg-turbo --recurse
```
