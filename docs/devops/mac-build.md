# macOS Build Documentation

**Status**: 🚧 In Progress

**Platform**: macOS 15.7.2 (x86_64)

**Target**: GitHub-hosted `macos-latest` runners

---

## Phase 0: Pre-Flight Checks ✅

### vcpkg Bootstrap
- ✅ vcpkg successfully bootstrapped for macOS
- Version: 2025-11-19-da1f056dc0775ac651bea7e3fbbf4066146a55f3
- Binary: macOS native

### GTK3 Analysis
- ✅ **Zero GTK3 usage found in source code** (grep returned 0 matches)
- Safe to skip GTK3 and all related packages (GDK, GIO, GOBJECT)
- No UI code depends on GTK3

### Platform-Specific Code
- ✅ No existing `__APPLE__` or `__MACH__` defines found
- Clean codebase - no macOS-specific code yet
- Will need to add platform guards as needed during build

### CMake Version
- Local: 4.1.2
- Required: 3.29+
- ✅ Compatible (newer version)

### Key Findings
1. **GTK3 is completely unused** - can skip entirely
2. **No existing macOS code** - starting fresh
3. **Build tools ready** - vcpkg, cmake, ninja all present

---

## Phase 1: CMake Configuration ✅

### Changes Made
- ✅ Added `ENABLE_MACOS` option to base/CMakeLists.txt
- ✅ Added macOS compile definitions
- ✅ Configured macOS package finding (GLEW, glfw3, GLUT)
- ✅ Skipped GTK3/GDK/GIO/GOBJECT on macOS (not used)
- ✅ Set `VCPKG_TARGET_TRIPLET` to `x64-osx`

---

## Phase 2: Dependency Management ✅

### vcpkg.json Platform Filters Added
- ✅ **whisper**: Split into CUDA (!osx) and NoCUDA (osx) variants
- ✅ **opencv4**: Split into CUDA (!osx) and NoCUDA (osx) variants
- ✅ **freeglut**: Linux-only (macOS uses native GLUT)
- ✅ **gtk3**: Linux-only (confirmed unused in source)
- ✅ **re (libre)**: Linux-only (only used in ARM64/CUDA builds)
- ✅ **baresip**: Linux-only (only used in ARM64/CUDA builds)

---

## Phase 3: Build Iteration 🚧

### First CMake Configure Attempt
_In progress..._

---

## Issues & Solutions

### Issue #1: baresip Cross-Platform Linkage Error
**Error**: `ninja: error: '/Users/akhil/git/ApraPipes2/build/vcpkg_installed/arm64-linux/lib/libre.so', needed by 'libbaresip.7.2.0.dylib', missing`

**Root Cause**: baresip was set to `platform: "!windows"`, causing it to install on macOS even though it's only used in ENABLE_ARM64 and ENABLE_LINUX CMake blocks.

**Investigation**:
```bash
# Check baresip usage in CMakeLists.txt
grep -n "BARESIP_LIB" base/CMakeLists.txt
# Found: Only referenced at lines 113, 162 inside ENABLE_ARM64 and ENABLE_LINUX blocks

# Check baresip usage in source code
grep -r "baresip" base/src base/include --include="*.cpp" --include="*.h" -i
# Found: Zero matches - not used in source
```

**Solution**: Changed vcpkg.json platform filter from `"!windows"` to `"linux"` for both `re` and `baresip`

**Status**: ✅ Fixed

---

### Issue #2: ffmpeg Missing nasm Assembler
**Error**: `Could not find nasm. Please install it via your package manager: brew install nasm`

**Root Cause**: ffmpeg requires nasm (Netwide Assembler) for x86 assembly optimizations, which wasn't installed on macOS.

**Solution**: Installed nasm via Homebrew:
```bash
brew install nasm
```

**Status**: ✅ Fixed

---

### Issue #3: libmp4 Compilation Errors on macOS
**Error**: Multiple compilation errors in `libmp4/src/mp4_demux.c`:
1. Lines 121, 146, 162: `void function should not return a value [-Wreturn-mismatch]`
2. Line 360: `type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int [-Wimplicit-int]` (C++ `auto` keyword in C code)

**Root Cause**: libmp4 source from Apra-Labs has C/C++ mixing issues:
- Void functions `mp4_demux_toggle_playback`, `mp4_set_reader_pos_lastframe`, `mp4_set_reader_pos_firstframe` trying to `return -ENOENT;`
- Using C++ `auto` keyword in C code: `for (auto i = 0; i < count; i++)`

**Solution**: Created patch file `thirdparty/custom-overlay/libmp4/fix-macos-build.patch`:
- Changed `return -ENOENT;` to `return;` in void functions (lines 121, 146, 162)
- Changed `for (auto i = 0; ...)` to `for (int i = 0; ...)` (line 360)
- Updated `portfile.cmake` to apply patch via `PATCHES` parameter

**Status**: ✅ Fixed, ready to retry build

---

## Build Commands

### Local Development Build
```bash
# Bootstrap vcpkg (one-time)
cd vcpkg && ./bootstrap-vcpkg.sh

# Configure
cmake -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DENABLE_MACOS=ON \
  -DENABLE_CUDA=OFF \
  -DENABLE_LINUX=OFF \
  -DENABLE_WINDOWS=OFF \
  ../base

# Build
cmake --build build -j $(sysctl -n hw.ncpu)

# Test
./build/aprapipesut
```

---

## Dependencies

### Homebrew Packages Needed
```bash
brew install pkg-config autoconf automake libtool nasm
```

### vcpkg Packages
_See base/vcpkg.json with platform filters_

---

## Test Results

_Will be populated after Phase 4_

---

## GitHub Actions Workflow

_Will be documented after Phase 5_

---

## Timeline

- Phase 0 (Pre-Flight): ✅ Completed
- Phase 1 (CMake): ✅ Completed
- Phase 2 (Dependencies): ✅ Completed
- Phase 3 (Build): 🚧 In progress (fixing vcpkg install errors)
- Phase 4 (Tests): ⏳ Pending
- Phase 5 (CI): ⏳ Pending
- Phase 6 (Docs): ⏳ Pending
