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

### Status: FIX IDENTIFIED (Simple header change)

### Severity: High → Low (once fix applied)

### Affected Modules
- `JPEGEncoderL4TM`
- `JPEGDecoderL4TM`

### Error Message

```
Wrong JPEG library version: library is 62, caller expects 80
```

### Root Cause (Verified via SSH Diagnostics)

**NVIDIA ships mismatched headers and library in JetPack 5.x:**

| Component | JPEG_LIB_VERSION | Location |
|-----------|------------------|----------|
| vcpkg jconfig.h | **62** | `_build/vcpkg_installed/arm64-linux/include/jconfig.h` |
| NVIDIA libjpeg-8b header | **80** | `/usr/src/jetson_multimedia_api/include/libjpeg-8b/jpeglib.h` |
| vcpkg libjpeg.a | **62** | Statically linked into binary |
| libnvjpeg.so internal | **62** | Runtime version check |

**What happens:**
1. L4TM code includes `"libjpeg-8b/jpeglib.h"` → compiles expecting version 80
2. At link time, vcpkg's `libjpeg.a` (version 62) provides `jpeg_*` symbols
3. Runtime: `jpeg_CreateCompress` checks version → **FAIL** (caller=80, library=62)

**The bug:** NVIDIA ships version 80 headers (`libjpeg-8b/jpeglib.h`) but `libnvjpeg.so` contains version 62 libjpeg internally. This is an NVIDIA header/library mismatch in JetPack 5.x.

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

#### Option F: CMake OBJECT Library Isolation (Recommended Fix)

Compile L4TM sources in a separate CMake OBJECT library with isolated include paths that use system libjpeg headers (ABI 62) instead of the mismatched `libjpeg-8b` headers from Jetson Multimedia API.

**Pros:**
- Clean separation of compilation contexts
- L4TM uses consistent headers + runtime (both ABI 62)
- Rest of codebase continues using vcpkg libjpeg-turbo (ABI 80)
- No LD_PRELOAD or deployment complexity
- Only affects ARM64 builds

**Cons:**
- Requires explicit linking to system libjpeg for L4TM
- Potential link-time symbol conflicts if vcpkg's static symbols take precedence

**Why This Should Work:**

The root cause is that L4TM code includes `libjpeg-8b/jpeglib.h` (ABI 80 struct sizes) but at runtime, `libnvjpeg.so` loads system libjpeg (ABI 62). By compiling L4TM with system `/usr/include/jpeglib.h` (ABI 62), the struct sizes match what the runtime library expects.

**Implementation:**

```cmake
if(ENABLE_ARM64)
    # Isolated OBJECT library for L4TM JPEG modules
    set(L4TM_JPEG_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/JPEGEncoderL4TM.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/JPEGEncoderL4TMHelper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/JPEGDecoderL4TM.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/JPEGDecoderL4TMHelper.cpp
    )

    find_path(SYSTEM_JPEG_INCLUDE jpeglib.h PATHS /usr/include REQUIRED)

    add_library(l4tm_jpeg_objects OBJECT ${L4TM_JPEG_SOURCES})

    # System libjpeg MUST come first
    target_include_directories(l4tm_jpeg_objects BEFORE PRIVATE
        ${SYSTEM_JPEG_INCLUDE}
    )

    target_include_directories(l4tm_jpeg_objects PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${JETSON_MULTIMEDIA_LIB_INCLUDE}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
    )

    target_link_libraries(l4tm_jpeg_objects PRIVATE jpeg)
endif()
```

Also change header includes from:
```cpp
#include "libjpeg-8b/jpeglib.h"
```
To:
```cpp
#include <jpeglib.h>
```

**Risks:**

| Risk | Severity | Mitigation |
|------|----------|------------|
| Link-time symbol conflicts | HIGH | If vcpkg's static libjpeg symbols take precedence, fall back to separate shared library |
| vcpkg toolchain overrides include paths | MEDIUM | Use absolute paths, CMAKE_C_FLAGS if needed |
| System libjpeg-dev not installed | LOW | Use REQUIRED in find_path |

### Cross-Platform Precedent

Similar libjpeg conflicts have been encountered and solved on other platforms:

**macOS (build-test-macosx.yml):**
```yaml
- name: CRITICAL - Unlink Homebrew JPEG BEFORE vcpkg (prevent path eclipsing)
  run: |
    brew unlink jpeg-turbo || true
```
Homebrew's jpeg-turbo headers were eclipsing vcpkg's, causing build failures. Solution: unlink before vcpkg runs.

**Linux Node.js Addon (INTEGRATION_TESTS.md):**
GTK brought in system libjpeg which conflicted with vcpkg's statically linked libjpeg-turbo, causing crashes in `cv::imencode`. Solution: created `aprapipes_node_headless` library excluding GTK modules.

### Recommended Path Forward

1. **Immediate (Option F):** Implement CMake OBJECT library isolation
   - Enable the 15 disabled L4TM tests
   - Compile L4TM sources with system libjpeg headers (ABI 62)
   - Verify no regression in ImageEncoderCV or JPEGEncoderNVJPEG tests

2. **Fallback:** If Option F fails due to link-time symbol conflicts, build L4TM as a separate shared library (`libl4tm_jpeg.so`) that explicitly links only system libjpeg

3. **Alternative:** Continue using `JPEGEncoderNVJPEG` as workaround for users who don't need L4TM specifically

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
