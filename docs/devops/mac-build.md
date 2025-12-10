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

## Phase 1: CMake Configuration 🚧

_In progress..._

---

## Issues & Solutions

### Issue Log
_Will be populated as issues are encountered_

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
brew install pkg-config autoconf automake libtool
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

- Phase 0 (Pre-Flight): ✅ Completed in 5 minutes
- Phase 1 (CMake): 🚧 In progress
- Phase 2 (Dependencies): ⏳ Pending
- Phase 3 (Build): ⏳ Pending
- Phase 4 (Tests): ⏳ Pending
- Phase 5 (CI): ⏳ Pending
- Phase 6 (Docs): ⏳ Pending
