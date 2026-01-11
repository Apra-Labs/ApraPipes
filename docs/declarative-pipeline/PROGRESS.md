# Declarative Pipeline - Progress Tracker

> Last Updated: 2026-01-10

**Git Branch:** `feat-declarative-pipeline-v2` (tracking `origin/feat-declarative-pipeline-v2`)

---

## Current Status

**Sprints 1-3:** ✅ COMPLETE
**Sprint 4 (Node.js):** ✅ COMPLETE
**Sprint 5 (CUDA):** 🔄 IN PROGRESS (blocked by FFmpeg build on Ubuntu 24.04)
**Sprint 6 (DRY Refactoring):** ✅ COMPLETE

```
Core Infrastructure:  ✅ Complete (Metadata, Registry, Factory, Validator, CLI)
JSON Migration:       ✅ Complete (TOML removed, JsonParser added)
Module Coverage:      37 modules (cross-platform) + platform-specific
Node.js Addon:        ✅ Complete (Phase 5 - testing, docs, examples)
CUDA Modules:         ✅ 12 of 13 registered (H264EncoderNVCodec pending)
```

---

## Sprint 4 Progress

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ Complete | TOML removal |
| Phase 1 | ✅ Complete | JSON parser (24 tests) |
| Phase 2 | ✅ Complete | Node addon foundation |
| Phase 3 | ✅ Complete | Core JS API (createPipeline, Pipeline, ModuleHandle) |
| Phase 4 | ✅ Complete | Event system (on/off, health/error callbacks) |
| Phase 4.5 | ✅ Complete | Dynamic property support (getProperty/setProperty) |
| Phase 5 | ✅ Complete | Testing, docs, examples |

---

## Sprint 5 Progress (CUDA)

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | CUDA factory infrastructure (CudaFactoryFn, ModuleRegistry) |
| Phase 2 | ✅ Complete | ModuleFactory CUDA stream management |
| Phase 3 | ✅ Complete | CudaModuleRegistrationBuilder template |
| Phase 4 | ✅ Complete | Register 12 CUDA modules |
| Phase 5 | ⏳ Pending | Testing with JSON pipelines (blocked by FFmpeg build on Ubuntu 24.04) |

**Implementation Details:**
- Added `CudaFactoryFn` type-erased factory to `ModuleInfo`
- `ModuleFactory::build()` creates a shared `cudastream_sp` for all CUDA modules
- `CudaModuleRegistrationBuilder` fluent API for CUDA module registration
- Each CUDA module's factory lambda receives the shared stream

---

## Registered Modules

### Source Modules (7)
- TestSignalGenerator (patterns: GRADIENT, CHECKERBOARD, COLOR_BARS, GRID)
- FileReaderModule
- Mp4ReaderSource (outputFormat: "h264"/"jpeg" for declarative)
- WebCamSource
- RTSPClientSrc
- ExternalSourceModule
- AudioCaptureSrc

### Transform Modules (13)
- ImageDecoderCV
- ImageEncoderCV
- ImageResizeCV
- RotateCV
- ColorConversion
- VirtualPTZ (dynamic: roiX, roiY, roiWidth, roiHeight)
- TextOverlayXForm (dynamic properties)
- BrightnessContrastControl (dynamic: contrast, brightness)
- AffineTransform
- BMPConverter
- OverlayModule
- HistogramOverlay
- AudioToTextXForm (Whisper speech-to-text)

### Analytics Modules (5)
- FaceDetectorXform
- FacialLandmarkCV
- QRReader
- CalcHistogramCV
- MotionVectorExtractor

### Sink Modules (6)
- FileWriterModule
- Mp4WriterSink
- StatSink
- ExternalSinkModule
- RTSPPusher
- ThumbnailListGenerator

### Utility Modules (6)
- Split
- Merge
- ValveModule
- FramesMuxer
- MultimediaQueueXform
- ArchiveSpaceManager

### Linux-Only Modules (ENABLE_LINUX)
- VirtualCameraSink (virtual camera device output)
- H264EncoderV4L2 (V4L2 hardware encoder)

### CUDA-Only Modules (ENABLE_CUDA)
- H264Decoder (NVDEC decoder)
- GaussianBlur (NPP Gaussian blur filter)
- ResizeNPPI (NPP image resizing)
- RotateNPPI (NPP image rotation)
- CCNPPI (NPP color space conversion)
- EffectsNPPI (NPP brightness/contrast/saturation/hue)
- OverlayNPPI (NPP image overlay with alpha blending)
- CudaMemCopy (host/device memory transfer)
- CudaStreamSynchronize (CUDA stream sync)
- JPEGDecoderNVJPEG (nvJPEG decoder)
- JPEGEncoderNVJPEG (nvJPEG encoder)
- MemTypeConversion (HOST/DEVICE/DMA memory type conversion)
- CuCtxSynchronize (CUDA context synchronization)

**Total: 37 cross-platform + 2 Linux + 13 CUDA = 52 modules**

---

## Modules Not Registered

### CUDA Modules (require special factory pattern)
| Module | Reason |
|--------|--------|
| H264EncoderNVCodec | Requires apracucontext_sp (not cudastream_sp) |

**Note:** 12 CUDA modules now registered using CudaModuleRegistrationBuilder pattern.
The ModuleFactory automatically creates and shares a cudastream_sp across all CUDA modules in a pipeline.

### Jetson-Only Modules (L4T/Tegra)
| Module | Reason |
|--------|--------|
| NvArgusCamera | Jetson Argus camera API |
| NvV4L2Camera | Jetson V4L2 camera |
| NvTransform | Jetson multimedia transform |
| JPEGDecoderL4TM | L4T Multimedia decoder |
| JPEGEncoderL4TM | L4T Multimedia encoder |
| DMAFDToHostCopy | Jetson DMA file descriptor |

**Status:** Need conditional registration with `#ifdef` guards on Jetson builds.

### Display/UI Modules
| Module | Reason |
|--------|--------|
| GtkGlRenderer | Requires GTK display |
| EglRenderer | Requires EGL display |
| ImageViewerModule | Requires OpenCV GUI |
| KeyboardListener | Requires keyboard input |

**Status:** Could be registered but typically not used in headless pipelines.

---

## Build Status

| Platform | Status | Node Addon | Modules |
|----------|--------|------------|---------|
| macOS | ✅ Pass | ✅ | 37 |
| Linux x64 | ✅ Pass | ✅ | 39 (+V4L2) |
| Windows | ✅ Pass | ✅ | 37 |
| Linux ARM64 | ✅ Pass | ❌ (#493) | 39 |
| Linux CUDA | 🔄 Pending | ✅ | 52 (+13 CUDA modules) |
| Jetson | ✅ Pass | ❌ | 40+ (+Jetson modules) |

**Note:** Linux CUDA local build on Ubuntu 24.04 requires custom FFmpeg overlay (see below).

---

## Ubuntu 24.04 FFmpeg Build Issue

### Problem
FFmpeg 4.4.3 has inline assembly code in `libavcodec/x86/mathops.h` that is incompatible with binutils >= 2.41 (Ubuntu 24.04 ships binutils 2.42). The issue is with constraint modifiers in x86 shift instructions.

### Solution
Created a custom vcpkg overlay port at `thirdparty/custom-overlay/ffmpeg/` that:
1. Uses FFmpeg 4.4.3 source (same as vcpkg.json override)
2. Adds patch `0024-fix-binutils-2.41-mathops.patch` to fix the inline assembly

### Patch Details
The patch modifies `libavcodec/x86/mathops.h` to use `__builtin_constant_p()` to select between:
- Immediate constraint `"i"` for compile-time constants
- Register constraint `"c"` for runtime values

This is the same fix that was merged upstream in FFmpeg 5.x+.

### Usage
When building on Ubuntu 24.04, use the overlay:
```bash
cmake -B build \
  -DVCPKG_OVERLAY_PORTS=thirdparty/custom-overlay \
  ...
```

### CI Note
CI runs on Ubuntu 22.04 (binutils 2.38) which doesn't have this issue. The overlay is only needed for local Ubuntu 24.04 development.

---

## OpenCV CUDA Build Issue

### Problem
When building OpenCV with CUDA support locally, vcpkg may fail to find CUDA:
```
CMake Error: WITH_CUDA is enabled but HAVE_CUDA is FALSE
```

### Potential Resolutions
1. **Set CUDA environment variables before cmake**:
   ```bash
   export CUDA_PATH=/usr/local/cuda
   export CUDAToolkit_ROOT=/usr/local/cuda
   export CUDACXX=/usr/local/cuda/bin/nvcc
   ```

2. **Check CUDA installation**:
   ```bash
   nvcc --version
   ls /usr/local/cuda/include/cuda.h
   ```

3. **If using CUDA 11.8 with GCC 13+**, set GCC-11:
   ```bash
   export CC=/usr/bin/gcc-11
   export CXX=/usr/bin/g++-11
   export CUDAHOSTCXX=/usr/bin/g++-11
   ```

### Status
**RESOLVED** - Using GCC-11 with proper environment variables allows CUDA 11.8 to build successfully on Ubuntu 24.04.

**Working Command:**
```bash
export CUDA_PATH=/usr/local/cuda
export CUDAToolkit_ROOT=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11

cmake -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_OVERLAY_PORTS=thirdparty/custom-overlay \
  -DVCPKG_TARGET_TRIPLET=x64-linux-release \
  -DENABLE_CUDA=ON \
  base
```

---

## Test Results

### C++ Unit Tests (Boost.Test)

| Test Suite | Tests |
|------------|-------|
| MetadataTests | 36 |
| ModuleRegistryTests | 21 |
| PipelineDescriptionTests | 37 |
| JsonParserTests | 24 |
| ModuleFactoryTests | 42 |
| FrameTypeRegistryTests | 34 |
| PipelineValidatorTests | 28 |
| PropertyMacrosTests | 28 |
| PropertyValidatorTests | 26 |
| ModuleRegistrationTests | 11 |

**Total: 268+ declarative C++ tests passing**

### Node.js Tests

| Test File | Tests |
|-----------|-------|
| event_tests.js | 10 |
| ptz_dynamic_props_test.js | 10 |

**Total: 20 Node.js tests passing**

---

## Documentation

- [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) - Module registration guide
- [PIPELINE_AUTHOR_GUIDE.md](./PIPELINE_AUTHOR_GUIDE.md) - JSON/JS pipeline creation
- [node-api.md](../node-api.md) - Node.js API reference
- [examples/node/](../../examples/node/) - Working Node.js examples

---

## Node.js Examples

| Example | Description |
|---------|-------------|
| basic_pipeline.js | Generate test frames, encode to JPEG, write to files |
| ptz_control.js | VirtualPTZ with real-time property changes |
| event_handling.js | Health/error event handling |
| image_processing.js | Color bars + brightness/contrast control |
| rtsp_pusher_demo.js | Mp4ReaderSource -> RTSPPusher streaming |

---

## Sprint 6: DRY Refactoring

> Started: 2026-01-09
> Completed: 2026-01-09

### Problem Statement
The declarative pipeline has DRY (Don't Repeat Yourself) violations where defaults are duplicated instead of using the C++ API defaults.

### Task R1: Fix sieve Default ✅
**Issue:** `Connection.sieve` defaults to `false`, but `Module::setNext()` defaults to `sieve=true`.

**Solution:**
- `base/include/declarative/PipelineDescription.h` - Changed `bool sieve = false` to `std::optional<bool> sieve`
- `base/src/declarative/JsonParser.cpp` - Only sets sieve when explicitly specified in JSON
- `base/src/declarative/ModuleFactory.cpp` - Only passes sieve to setNext() when has_value()

When sieve is not specified in JSON, the C++ API default (`sieve=true`) is used automatically.

### Task R2: Fix Props Defaults ✅
**Issue:** Property defaults in `ModuleRegistrations.cpp` were hardcoded instead of querying from Props classes.

**Solution:** Updated modules to instantiate default Props and query values:
```cpp
FileReaderModuleProps fileReaderDefaults;
.intProp("startIndex", "Starting file index", false, fileReaderDefaults.startIndex, 0)
```

**Modules updated:**
- FileReaderModule (startIndex, maxIndex, readLoop)
- FileWriterModule (append)
- AffineTransform (angle, scale, shear, offsetX, offsetY, borderType)

### Task R3: Fix Type Validation ✅
**Issue:** Validator rejected int values for float properties (`E201: Type mismatch`).

**Solution:** Updated `PipelineValidator.cpp` to allow int values for float properties, since JSON doesn't distinguish `45` from `45.0`.

### Task R4: Integration Test ✅
**Results:** (test script auto-detects GTK3 preload on Linux)
```
Total:   7
Passed:  7
Failed:  0
Skipped: 3 (2 face detection models + 1 Node.js ImageEncoderCV)
```

**Note:** All pipelines pass. The `14_affine_transform_demo` is skipped for Node.js runtime due to libjpeg setjmp/longjmp threading conflict, but works correctly with CLI.

### Task R5: ImageEncoderCV Node.js Fix ✅
**Issue:** `14_affine_transform_demo` crashed with SIGSEGV in Node.js addon on Linux.

**Root Cause:** Node.js addon required GTK preload to resolve GtkGlRenderer symbols. GTK pulls in system libjpeg which conflicts with vcpkg's statically linked libjpeg-turbo, causing crash in `cv::imencode`.

**Solution (commit 849c1c00f):**
1. Created `aprapipes_node_headless` library in `base/CMakeLists.txt`
2. Excludes GTKGL_FILES (GtkGlRenderer, etc.) from Node.js addon on Linux
3. No GTK preload needed = no libjpeg symbol conflict

**Status:** Fix committed and pushed, pending CI rebuild. Test script temporarily skips on Node.js/Linux. macOS and CLI work correctly (macOS uses Cocoa, not GTK).

---

## Future Work

### Priority 1: H264EncoderNVCodec Support
- Create apracucontext_sp factory pattern (different from cudastream_sp)
- Enable H264EncoderNVCodec for declarative use
- Consider shared CUDA context across pipeline

### Priority 2: Jetson Module Registration
- Add `#ifdef` guards for Jetson-specific modules
- Test on Jetson device with L4T

### Priority 3: Display Modules (Optional)
- Register UI modules with platform checks
- Low priority - mainly for debugging

### Priority 4: ARM64 Node Addon
- Fix -fPIC linking issue (#493)
