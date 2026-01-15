# Jetson Platform Known Issues

> Last Updated: 2026-01-15

This document details known issues specific to the Jetson (ARM64/L4T) platform when using the declarative pipeline system.

---

## Table of Contents

1. [Issue J1: libjpeg Version Conflict (L4TM Modules)](#issue-j1-libjpeg-version-conflict-l4tm-modules)
2. [Issue J2: Node.js Addon Boost.Serialization Linking](#issue-j2-nodejs-addon-boostserialization-linking)
3. [Issue J3: H264EncoderV4L2 Not Registered on ARM64](#issue-j3-h264encoderv4l2-not-registered-on-arm64)

---

## Issue J1: libjpeg Version Conflict (L4TM Modules)

### Status: OPEN (Blocker for L4TM modules)

### Severity: High

### Affected Modules
- `JPEGEncoderL4TM`
- `JPEGDecoderL4TM`

### Error Message

```
Wrong JPEG library version: library is 62, caller expects 80
```

Or at runtime:
```
JPEG parameter struct mismatch: library thinks size is 584, caller expects 680
```

### Description

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

### Why This Was Never Caught in CI

All L4TM unit tests in the codebase are **disabled** with Boost.Test's `disabled()` decorator:

```cpp
// base/test/jpegencoderl4tm_tests.cpp
BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic, * boost::unit_test::disabled())
{
    // Test code here - never executed
}
```

There are **11 disabled tests** for JPEGEncoderL4TM and similar counts for JPEGDecoderL4TM. These tests were likely disabled during initial development when the conflict was first discovered, but the issue was never resolved.

### Affected Examples

The following Jetson JSON examples are blocked by this issue:

| Example | Status |
|---------|--------|
| `examples/jetson/01_jpeg_decode_transform.json` | ❌ Blocked |
| `examples/jetson/01_test_signal_to_jpeg.json` | ❌ Blocked |

### Potential Solutions

#### Option A: Use System libjpeg for Entire Build (Not Recommended)

Configure vcpkg to use system libjpeg instead of libjpeg-turbo.

**Pros:**
- Simple configuration change
- Consistent libjpeg version

**Cons:**
- Loses libjpeg-turbo performance benefits
- May break other platforms
- System libjpeg (6.2) is very old

**Implementation:**
```cmake
# In arm64-linux-release triplet or vcpkg.json
# Would require significant vcpkg customization
```

#### Option B: Build libjpeg-turbo as Shared Library (Partial Fix)

Build libjpeg-turbo as a shared library and ensure it's loaded before system libjpeg.

**Pros:**
- Maintains libjpeg-turbo performance
- Standard approach for symbol resolution

**Cons:**
- Complex deployment (must ship .so file)
- LD_PRELOAD or rpath manipulation required
- May not fully resolve struct layout issues

**Implementation:**
```cmake
# In vcpkg triplet
set(VCPKG_LIBRARY_LINKAGE dynamic)
```

Then at runtime:
```bash
LD_PRELOAD=/path/to/libjpeg.so.62.3.0 ./aprapipes_cli run example.json
```

#### Option C: Isolate L4TM Modules in Separate Process (Complex)

Run L4TM modules in a separate process that only links against system libjpeg.

**Pros:**
- Complete isolation
- No symbol conflicts

**Cons:**
- Significant architecture change
- IPC overhead for frame data
- Complex implementation

#### Option D: Use JPEGEncoderNVJPEG Instead (Recommended Workaround)

Use NVIDIA's nvJPEG library (CUDA-based) instead of L4T Multimedia JPEG.

**Pros:**
- No libjpeg dependency (uses CUDA directly)
- Better performance (GPU-accelerated)
- Already registered in declarative pipeline

**Cons:**
- Requires CUDA memory (need CudaMemCopy bridges)
- Different API/quality settings
- Not a drop-in replacement

**Implementation:**
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

#### Option E: Patch L4T Multimedia to Use libjpeg-turbo (Invasive)

Rebuild the L4T Multimedia libraries against libjpeg-turbo.

**Pros:**
- Proper fix at the source

**Cons:**
- Requires NVIDIA source code (may not be available)
- Maintenance burden for each L4T version
- Complex build process

### Recommended Path Forward

1. **Short-term (Sprint 8):** Document the issue and recommend `JPEGEncoderNVJPEG` as the alternative
2. **Medium-term:** Investigate Option B (shared libjpeg-turbo with LD_PRELOAD)
3. **Long-term:** Work with NVIDIA to understand L4TM's libjpeg requirements

### Related Files

- `base/src/JPEGEncoderL4TM.cpp` - L4TM encoder implementation
- `base/src/JPEGDecoderL4TM.cpp` - L4TM decoder implementation
- `base/test/jpegencoderl4tm_tests.cpp` - Disabled tests
- `base/test/jpegdecoderl4tm_tests.cpp` - Disabled tests
- `thirdparty/arm64-overlay/` - ARM64 vcpkg overlay ports

---

## Issue J2: Node.js Addon Boost.Serialization Linking

### Status: OPEN (Blocker for Node.js addon on Jetson)

### Severity: Medium (CLI works, only addon affected)

### Error Message

When loading the Node.js addon on Jetson:

```
Error: /path/to/aprapipes_node.node: undefined symbol: _ZTIN5boost7archive6detail17basic_iserializerE
```

Demangled symbol name:
```
typeinfo for boost::archive::detail::basic_iserializer
```

### Description

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

| Issue | Module(s) | Severity | Workaround | Fix Complexity |
|-------|-----------|----------|------------|----------------|
| J1 | JPEGEncoder/DecoderL4TM | High | Use JPEGEncoderNVJPEG | High |
| J2 | Node.js Addon | Medium | Use CLI | Medium |
| J3 | H264EncoderV4L2 | Low | Use H264EncoderNVCodec | Low |

---

## References

- [Boost.Serialization Documentation](https://www.boost.org/doc/libs/release/libs/serialization/)
- [NVIDIA L4T Multimedia API](https://docs.nvidia.com/jetson/l4t-multimedia/)
- [libjpeg-turbo vs libjpeg](https://libjpeg-turbo.org/About/Compatibility)
- [GCC Symbol Visibility](https://gcc.gnu.org/wiki/Visibility)
