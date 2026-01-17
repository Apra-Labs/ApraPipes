# Jetson Platform Known Issues

> Last Updated: 2026-01-16

This document details known issues specific to the Jetson (ARM64/L4T) platform when using the declarative pipeline system.

---

## Table of Contents

1. [Issue J1: libjpeg Version Conflict (L4TM Modules)](#issue-j1-libjpeg-version-conflict-l4tm-modules)
2. [Issue J2: Node.js Addon Boost.Serialization Linking](#issue-j2-nodejs-addon-boostserialization-linking)
3. [Issue J3: H264EncoderV4L2 Not Registered on ARM64](#issue-j3-h264encoderv4l2-not-registered-on-arm64)

---

## Issue J1: libjpeg Version Conflict (L4TM Modules)

### Status: ✅ RESOLVED (2026-01-16)

### Severity: ~~High~~ Resolved

### Affected Modules
- `JPEGEncoderL4TM` - ✅ Working
- `JPEGDecoderL4TM` - ✅ Working

### Resolution

The L4TM modules now work correctly thanks to the `L4TMJpegLoader` dlopen wrapper implementation (`base/src/L4TMJpegLoader.cpp`). This wrapper:

1. Uses `dlopen()` with `RTLD_LOCAL` to load `libnvjpeg.so` in isolation
2. Uses `dlsym()` to retrieve jpeg function pointers at runtime
3. Keeps NVIDIA's libjpeg symbols separate from vcpkg's libjpeg-turbo

**Test Status (CI-Linux-ARM64):**
- 7 L4TM tests PASSING
- 7 tests disabled with documented reasons (hardware limitations, deprecated patterns)

**Working Tests:**
- `jpegencoderl4tm_basic`, `basic_scale`, `basic_perf`, `basic_perf_scale`, `basic_width_notmultipleof32_2`
- `jpegdecoderl4tm_basic`, `jpegdecoderl4tm_rgb`

**Disabled Tests (legitimate reasons):**
- `jpegdecoderl4tm_mono` - NVIDIA hardware hangs on grayscale images
- `jpegencoderl4tm_rgb`, `rgb_perf` - Pre-existing NVIDIA RGB encoder crash
- `jpegencoderl4tm_basic_2`, `basic_width_notmultipleof32`, `basic_width_channels` - Deprecated "metadata after init" pattern
- `jpegencoderl4tm_basic_width_channels_2` - BGR not supported by L4TM encoder

### Original Error (Historical)

```
Wrong JPEG library version: library is 62, caller expects 80
```

Or at runtime:
```
JPEG parameter struct mismatch: library thinks size is 584, caller expects 680
```

### Original Description (Historical)

The L4T Multimedia (L4TM) JPEG encoder/decoder modules use NVIDIA's hardware-accelerated JPEG codec via the L4T Multimedia API (`libnvjpeg.so`). This library is dynamically linked against the **system libjpeg** (version 6.2, ABI version 62), which is part of the base Jetson L4T image.

However, ApraPipes is built with **vcpkg's libjpeg-turbo** (version 8.0, ABI version 80) for better performance and consistency across platforms. When the L4TM modules attempt to use libjpeg functions, the version mismatch causes struct size mismatches and crashes.

### Root Cause Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ApraPipes Application                        │
├─────────────────────────────────────────────────────────────────────┤
│  ImageEncoderCV      │  JPEGEncoderL4TM      │  JPEGEncoderNVJPEG   │
│  (OpenCV + vcpkg)    │  (L4T Multimedia API) │  (CUDA nvJPEG)       │
│         │            │         │             │         │            │
│         ▼            │         ▼             │         ▼            │
│  libjpeg-turbo 8.0   │  libnvjpeg.so (L4T)   │  nvjpeg (CUDA)       │
│  (vcpkg, static)     │         │             │  (no libjpeg dep)    │
│                      │         ▼             │                      │
│                      │  libjpeg.so.62        │                      │
│                      │  (system, dynamic)    │                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
1. vcpkg builds libjpeg-turbo as a **static library** and links it into OpenCV
2. The L4T Multimedia API (`libnvjpeg.so`) **dynamically links** against system libjpeg
3. When both are loaded in the same process, the version check fails
4. The struct layouts differ between versions (584 bytes vs 680 bytes)

### CI Status

L4TM tests now run in CI-Linux-ARM64:
- **7 tests enabled and passing**
- **7 tests disabled** with documented hardware/API limitations

### Jetson JSON Examples

| Example | Status |
|---------|--------|
| `examples/jetson/01_jpeg_decode_transform.json` | ✅ Should work |
| `examples/jetson/01_test_signal_to_jpeg.json` | ✅ Should work |

### Solution Implemented: dlopen Wrapper (L4TMJpegLoader)

The implemented solution uses runtime dynamic loading to isolate NVIDIA's libjpeg from vcpkg's libjpeg-turbo:

```cpp
// base/src/L4TMJpegLoader.cpp
void* handle = dlopen("libnvjpeg.so", RTLD_NOW | RTLD_LOCAL);
// RTLD_LOCAL keeps symbols private to this handle
```

This approach:
- **Pros:** No deployment changes, no LD_PRELOAD needed, works transparently
- **Cons:** Slightly more complex code, runtime lookup overhead (negligible)

### Alternative: JPEGEncoderNVJPEG (CUDA-based)

If L4TM modules don't meet your needs, `JPEGEncoderNVJPEG` is also available:

```json
{
  "modules": {
    "encoder": {
      "type": "JPEGEncoderNVJPEG",
      "props": { "quality": 90 }
    }
  }
}
```

Note: NVJPEG requires CUDA memory, so you'll need `HostCopyCuda` bridges.

### Related Files

- `base/src/JPEGEncoderL4TM.cpp` - L4TM encoder implementation
- `base/src/JPEGDecoderL4TM.cpp` - L4TM decoder implementation
- `base/test/jpegencoderl4tm_tests.cpp` - Disabled tests
- `base/test/jpegdecoderl4tm_tests.cpp` - Disabled tests
- `thirdparty/arm64-overlay/` - ARM64 vcpkg overlay ports

---

## Issue J2: Node.js Addon Boost.Serialization Linking

### Status: ✅ RESOLVED (2026-01-17)

### Severity: ~~Medium~~ Resolved

### Error Message

When loading the Node.js addon on Jetson:

```
Error: /path/to/aprapipes_node.node: undefined symbol: _ZTIN5boost7archive6detail17basic_iserializerE
```

Demangled symbol name:
```
typeinfo for boost::archive::detail::basic_iserializer
```

### Resolution

Added GCC version check in `base/CMakeLists.txt`:

```cmake
# GCC 9.x has stricter RTTI symbol handling - include Boost.Serialization in whole-archive
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
    message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION} detected - applying Boost.Serialization whole-archive workaround")
    target_link_libraries(${NODE_ADDON_NAME} PRIVATE
        -Wl,--whole-archive ${NODE_ADDON_APRAPIPES_LIB} ${Boost_SERIALIZATION_LIBRARY} -Wl,--no-whole-archive
    )
else()
    target_link_libraries(${NODE_ADDON_NAME} PRIVATE
        -Wl,--whole-archive ${NODE_ADDON_APRAPIPES_LIB} -Wl,--no-whole-archive
    )
endif()
```

**Key Points:**
- Version-gated to GCC < 10 (not platform-specific)
- Automatically stops being applied when upgrading to JetPack 6.x (GCC 11+)
- No impact on other platforms

**Verification (2026-01-17):**
```bash
node -e "require('./build/aprapipes.node')"
SUCCESS: Node addon loaded!
Methods: [ 'getVersion', 'listModules', 'describeModule', 'validatePipeline', ... ]
```

### Description (Historical)

The Node.js addon (`aprapipes_node.node`) fails to load on Jetson ARM64 due to missing RTTI (Run-Time Type Information) symbols from Boost.Serialization. The symbol `_ZTIN5boost7archive6detail17basic_iserializerE` is the typeinfo for a class used by Boost.Serialization's archive system.

### Root Cause Analysis

The issue stems from how the Node.js addon is linked:

```cmake
# base/CMakeLists.txt (lines 1227-1228)
target_link_libraries(${NODE_ADDON_NAME} PRIVATE
    -Wl,--whole-archive ${NODE_ADDON_APRAPIPES_LIB} -Wl,--no-whole-archive
)
# Later:
target_link_libraries(${NODE_ADDON_NAME} PRIVATE ${APRA_COMMON_LIBS})
```

**Link Order:**
1. `node-addon-api` (Node.js binding headers)
2. `aprapipes` static library (with `--whole-archive` to force all symbols)
3. `APRA_COMMON_LIBS` (includes `${Boost_LIBRARIES}`)

**The Problem:**

1. **RTTI symbols are special** - The C++ compiler generates typeinfo symbols for polymorphic classes. These are used by `dynamic_cast`, `typeid`, and exception handling.

2. **Static library linking** - When Boost.Serialization is built as a static library, its typeinfo symbols are only included if they're directly referenced.

3. **--whole-archive scope** - The `--whole-archive` flag only applies to `aprapipes`, not to Boost libraries. The linker may discard "unused" Boost symbols including typeinfo.

4. **ARM64 GCC behavior** - GCC 9.4 on Jetson (JetPack 5.x) has stricter symbol resolution than newer GCC versions on x86-64.

### Why This Only Affects Jetson

| Platform | GCC Version | Boost Build | Result |
|----------|-------------|-------------|--------|
| Linux x64 | 11+ | Static | ✅ Works |
| macOS | Clang 14+ | Static | ✅ Works |
| Windows | MSVC 2022 | Static | ✅ Works |
| Jetson ARM64 | 9.4 | Static | ❌ Fails |

The difference is likely due to:
- GCC 9's stricter handling of typeinfo symbols in shared libraries
- Different default visibility settings on ARM64
- Linker script differences

### Affected Functionality

- **CLI (`aprapipes_cli`)**: ✅ Works correctly
- **Node.js addon**: ❌ Cannot load on Jetson
- **C++ unit tests**: ✅ Work correctly

### Potential Solutions

#### Option A: Extend --whole-archive to Include Boost.Serialization

Force all Boost.Serialization symbols to be included.

```cmake
if(UNIX AND NOT APPLE)
    target_link_libraries(${NODE_ADDON_NAME} PRIVATE
        -Wl,--whole-archive
        ${NODE_ADDON_APRAPIPES_LIB}
        ${Boost_SERIALIZATION_LIBRARY}
        -Wl,--no-whole-archive
    )
endif()
```

**Pros:**
- Direct fix for the missing symbol

**Cons:**
- Increases binary size significantly
- May pull in unwanted symbols
- Need to identify exactly which Boost libraries need whole-archive

#### Option B: Use --no-as-needed Flag

Prevent the linker from discarding "unused" libraries.

```cmake
if(UNIX AND NOT APPLE)
    target_link_libraries(${NODE_ADDON_NAME} PRIVATE
        -Wl,--no-as-needed
        ${Boost_LIBRARIES}
        -Wl,--as-needed
    )
endif()
```

**Pros:**
- Simple flag addition
- Preserves all Boost symbols

**Cons:**
- May not resolve typeinfo specifically
- Increases binary size

#### Option C: Build Boost as Shared Libraries

Build Boost with shared library linkage on ARM64.

```cmake
# In arm64-linux-release triplet
set(VCPKG_LIBRARY_LINKAGE dynamic)
```

**Pros:**
- Typeinfo symbols properly exported from shared library
- Standard solution for this class of problems

**Cons:**
- Requires shipping Boost .so files
- More complex deployment
- May affect other platforms if not properly scoped

#### Option D: Explicit Symbol Export

Add explicit symbol visibility attributes to force typeinfo export.

**Cons:**
- Would require modifying Boost source code
- Not practical

#### Option E: Disable Boost.Serialization Usage in Declarative

Remove dependency on Boost.Serialization in the declarative pipeline code path.

**Pros:**
- Eliminates the dependency entirely

**Cons:**
- May require significant refactoring
- Boost.Serialization may be used by core ApraPipes code

### Recommended Path Forward

1. **Short-term:** Document that Node.js addon is not supported on Jetson; use CLI instead
2. **Medium-term:** Try Option A (extend --whole-archive) on a test branch
3. **Long-term:** Investigate why ARM64/GCC-9 behaves differently; may be fixed by JetPack 6.x with newer GCC

### Related Files

- `base/CMakeLists.txt` - Node addon linking (lines 1213-1245)
- `base/bindings/node/` - Node.js addon source code
- `thirdparty/triplets/arm64-linux-release.cmake` - ARM64 vcpkg triplet

### Diagnostic Commands

```bash
# Check which Boost libraries are linked
ldd aprapipes_node.node | grep boost

# Check for missing symbols
nm -u aprapipes_node.node | grep boost

# Check if symbol exists in Boost library
nm /path/to/libboost_serialization.a | grep basic_iserializer

# Try loading with verbose errors
LD_DEBUG=libs node -e "require('./aprapipes_node.node')"
```

---

## Issue J3: H264EncoderV4L2 Not Registered on ARM64

### Status: OPEN (Minor)

### Severity: Low

### Description

The `H264EncoderV4L2` module is not available in the declarative pipeline on ARM64 builds because it's only registered under the `#ifdef ENABLE_LINUX` flag, which is not defined for ARM64 builds.

### Current Registration

```cpp
// base/src/declarative/ModuleRegistrations.cpp

#ifdef ENABLE_LINUX
    // H264EncoderV4L2 - only on Linux x64 currently
    ModuleRegistrationBuilder("H264EncoderV4L2")
        .category(ModuleCategory::ENCODER)
        // ... rest of registration
        .finalize(registry);
#endif
```

### Root Cause

ARM64 builds use `ENABLE_ARM64` instead of `ENABLE_LINUX`. The V4L2 encoder should work on ARM64 Linux (Jetson), but the preprocessor guard excludes it.

### Solution

Either:

1. **Define ENABLE_LINUX for ARM64+Linux builds:**
   ```cmake
   if(ENABLE_ARM64 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
       add_compile_definitions(ENABLE_LINUX)
   endif()
   ```

2. **Or add ARM64 guard to registration:**
   ```cpp
   #if defined(ENABLE_LINUX) || defined(ENABLE_ARM64)
       // H264EncoderV4L2 registration
   #endif
   ```

### Impact

- `examples/jetson/02_h264_encode_demo.json` cannot run
- Users must use `H264EncoderNVCodec` instead (CUDA-based)

### Workaround

Use `H264EncoderNVCodec` which is registered on ARM64 CUDA builds.

---

## Summary Table

| Issue | Module(s) | Severity | Status | Notes |
|-------|-----------|----------|--------|-------|
| J1 | JPEGEncoder/DecoderL4TM | ~~High~~ | ✅ RESOLVED | dlopen wrapper isolates symbols |
| J2 | Node.js Addon | ~~Medium~~ | ✅ RESOLVED | GCC 9 whole-archive workaround |
| J3 | H264EncoderV4L2 | Low | ⚠️ OPEN | Use H264EncoderNVCodec |

---

## Jetson Device Development Rules

When working on the Jetson device (ssh to self-hosted runner), follow these **critical rules**:

### Protected Directories

| Path | Purpose | Action |
|------|---------|--------|
| `/data/action-runner/` | GitHub Actions runner | **NEVER modify** |
| `/data/.cache/` | vcpkg cache shared with CI | **NEVER delete en-masse** |
| `/data/ws/` | Development workspace | Use for all development work |

### Workspace Layout

```
/data/ws/ApraPipes/          # Main workspace (NVMe)
├── _build/                  # Build directory
├── base/                    # Source code
└── vcpkg/                   # vcpkg submodule

/data/.cache/                # Shared cache (DO NOT DELETE)
├── vcpkg_installed/         # vcpkg packages
└── tmp/                     # Temp for builds (TMPDIR)

/data/action-runner/         # GitHub Actions (DO NOT TOUCH)
├── _work/                   # Workflow workspace
└── ...
```

### CI Workflow Conflicts

**Before pushing changes** that will trigger CI-Linux-ARM64:
- Disable CI-Linux-ARM64.yml first (add to `branches-ignore` or disable workflow)
- Push your changes and test locally
- Re-enable CI-Linux-ARM64.yml when ready

**Why:** Development work and CI runs compete for the same hardware resources. Running both concurrently can cause OOM kills or unexpected failures.

### Build Commands

```bash
ssh akhil@192.168.1.18
cd /data/ws/ApraPipes

# Configure
cmake -B _build -S base \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_ARM64=ON \
  -DENABLE_CUDA=ON

# Build (use -j2 to avoid OOM on 8GB Jetson)
TMPDIR=/data/.cache/tmp cmake --build _build -j2

# Test
./_build/aprapipesut --run_test="ModuleRegistryTests/*" --log_level=test_suite
```

---

## References

- [Boost.Serialization Documentation](https://www.boost.org/doc/libs/release/libs/serialization/)
- [NVIDIA L4T Multimedia API](https://docs.nvidia.com/jetson/l4t-multimedia/)
- [libjpeg-turbo vs libjpeg](https://libjpeg-turbo.org/About/Compatibility)
- [GCC Symbol Visibility](https://gcc.gnu.org/wiki/Visibility)
