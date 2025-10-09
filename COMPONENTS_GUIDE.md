# ApraPipes Component-Based Build Guide

**Build only what you need - Reduce build times from 60-90 minutes to 10-30 minutes**

## Table of Contents
- [Quick Start](#quick-start)
- [Component Overview](#component-overview)
- [Build Scripts Usage](#build-scripts-usage)
- [Component-Module Matrix](#component-module-matrix)
- [Infrastructure Backends](#infrastructure-backends)
- [Common Use Cases](#common-use-cases)
- [Build Time Comparison](#build-time-comparison)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Windows with CUDA
```bash
# Minimal pipeline (fastest build)
build_windows_cuda.bat --preset minimal

# Video processing (most common)
build_windows_cuda.bat --preset video

# GPU-accelerated processing
build_windows_cuda.bat --preset cuda
Nee
# Full build (backward compatible)
build_windows_cuda.bat --preset full
```

### Linux with CUDA
```bash
# Minimal pipeline
./build_linux_cuda.sh --preset minimal

# Video processing
./build_linux_cuda.sh --preset video

# GPU-accelerated
./build_linux_cuda.sh --preset cuda

# Full build
./build_linux_cuda.sh --preset full
```

### Jetson (ARM64)
```bash
# Jetson-optimized build
./build_jetson.sh --preset jetson

# Full Jetson build
./build_jetson.sh --preset full
```

---

## Component Overview

### CORE (Always Required)
**Build Time:** ~10-15 min
**Description:** Pipeline infrastructure and basic I/O
**Use When:** Building any ApraPipes application

**Key Capabilities:**
- Pipeline management and frame flow
- File reader/writer modules
- Basic control flow (split, merge, valve)
- Frame memory management
- Serialization and logging

**Infrastructure Notes:**
- Includes OpenCV (minimal) for image metadata
- Includes CUDA allocators when `ENABLE_CUDA=ON` (infrastructure only, no processing)

---

### VIDEO
**Build Time:** ~15-20 min additional
**Total Time:** ~25-30 min (with CORE)
**Description:** Video codecs and streaming

**Key Capabilities:**
- Mp4 reading and writing
- H264 encoding/decoding (CPU or GPU)
- RTSP client and pusher
- Video frame demuxing

**Dependencies:** CORE, FFmpeg, OpenH264, libmp4

---

### IMAGE_PROCESSING
**Build Time:** ~5-10 min additional
**Description:** OpenCV CPU-based image processing

**Key Capabilities:**
- Image resize, rotate, color conversion
- JPEG encoding/decoding
- Affine transformations (CPU)
- Brightness/contrast adjustments
- Image overlays and text rendering

**Dependencies:** CORE, OpenCV (CPU)

**Infrastructure Notes:**
- Works with or without CUDA
- AffineTransform has GPU acceleration when CUDA enabled (uses NPP)

---

### CUDA_COMPONENT
**Build Time:** ~10-15 min additional
**Total Time:** ~15-20 min (with CORE+VIDEO+IMAGE_PROCESSING, from cache)
**Description:** GPU-accelerated processing

**Key Capabilities:**
- NVIDIA NPP image processing (resize, rotate, color conversion)
- NVJPEG hardware JPEG encoding/decoding
- NVCODEC H264 hardware encoding/decoding
- CUDA memory operations
- GPU effects and overlays

**Dependencies:** CORE, IMAGE_PROCESSING, CUDA Toolkit, NPP, NVJPEG, NVCODEC, cuDNN, OpenCV (with CUDA)

**Infrastructure Requirements:**
- `ENABLE_CUDA=ON` required
- NVIDIA GPU with compute capability 5.2+ (Maxwell or newer)

---

### ARM64_COMPONENT (Jetson Only)
**Build Time:** ~10-15 min additional
**Description:** Jetson-specific hardware acceleration

**Key Capabilities:**
- NvArgus camera support
- V4L2 hardware codec support
- L4TM JPEG encoding/decoding
- DMA buffer management
- EGL rendering

**Dependencies:** CORE, CUDA_COMPONENT, Jetson L4T libraries

**Platform:** ARM64 Linux only (Jetson Nano, Xavier, Orin)

---

### WEBCAM
**Build Time:** ~2 min additional
**Description:** Webcam capture via OpenCV

**Dependencies:** CORE, IMAGE_PROCESSING, OpenCV (videoio)

---

### QR
**Build Time:** ~3 min additional
**Description:** QR code reading

**Dependencies:** CORE, IMAGE_PROCESSING, ZXing

---

### AUDIO
**Build Time:** ~30-40 min additional (due to Whisper)
**Description:** Audio capture and transcription

**Key Capabilities:**
- SFML-based audio capture
- Whisper speech-to-text (with CUDA acceleration)

**Dependencies:** CORE, SFML, Whisper (CUDA-enabled)

**Note:** Whisper is the longest-building dependency. Consider excluding if not needed.

---

### FACE_DETECTION
**Build Time:** ~5 min additional
**Description:** Face detection and facial landmarks

**Dependencies:** CORE, IMAGE_PROCESSING, OpenCV (DNN, contrib)

---

### GTK_RENDERING (Linux Only)
**Build Time:** ~10-15 min additional
**Description:** GUI rendering with GTK and OpenGL

**Dependencies:** CORE, IMAGE_PROCESSING, GTK3, GLEW, glfw3, OpenGL

**Platform:** Linux only

---

### THUMBNAIL
**Build Time:** ~2 min additional
**Description:** Thumbnail generation

**Dependencies:** CORE, IMAGE_PROCESSING

---

### IMAGE_VIEWER
**Build Time:** ~2 min additional
**Description:** Image viewing GUI

**Dependencies:** CORE, IMAGE_PROCESSING, OpenCV (highgui)

**Note:** Requires GUI support (X11/Windows)

---

## Build Scripts Usage

### Windows Scripts

#### `build_windows_cuda.bat` (CUDA-enabled builds)
```bash
# Usage
build_windows_cuda.bat [OPTIONS]

# Options
--help, -h              Display help information
--build-doc             Build documentation after compilation
--components "LIST"     Specify components (semicolon-separated)
--preset NAME           Use preset configuration

# Presets
--preset minimal        CORE only (~10-15 min)
--preset video          CORE + VIDEO + IMAGE_PROCESSING (~25-30 min)
--preset cuda           CORE + VIDEO + IMAGE_PROCESSING + CUDA_COMPONENT (~15-20 min)
--preset full           ALL components (~60-90 min)

# Custom component selection
build_windows_cuda.bat --components "CORE;VIDEO;CUDA_COMPONENT"

# Build with documentation
build_windows_cuda.bat --preset video --build-doc
```

#### `build_windows_no_cuda.bat` (CPU-only builds)
```bash
# Same options as CUDA build, but excludes CUDA_COMPONENT and ARM64_COMPONENT
build_windows_no_cuda.bat --preset video
```

---

### Linux Scripts

#### `build_linux_cuda.sh` (CUDA-enabled builds)
```bash
# Make executable (first time only)
chmod +x build_linux_cuda.sh

# Usage
./build_linux_cuda.sh [OPTIONS]

# Options
--help, -h              Display help information
--build-doc             Build documentation after compilation
--components "LIST"     Specify components (space or semicolon-separated)
--preset NAME           Use preset configuration

# Presets
--preset minimal        CORE only
--preset video          CORE + VIDEO + IMAGE_PROCESSING
--preset cuda           CORE + VIDEO + IMAGE_PROCESSING + CUDA_COMPONENT
--preset full           ALL components

# Examples
./build_linux_cuda.sh --preset video
./build_linux_cuda.sh --components "CORE VIDEO WEBCAM"
./build_linux_cuda.sh --preset cuda --build-doc
```

#### `build_linux_no_cuda.sh` (CPU-only builds)
```bash
chmod +x build_linux_no_cuda.sh
./build_linux_no_cuda.sh --preset video
```

---

### Jetson Script

#### `build_jetson.sh` (ARM64 with Jetson hardware)
```bash
chmod +x build_jetson.sh

# Jetson-optimized preset (recommended)
./build_jetson.sh --preset jetson

# Full build with all components
./build_jetson.sh --preset full

# Custom components
./build_jetson.sh --components "CORE VIDEO IMAGE_PROCESSING ARM64_COMPONENT"
```

**Jetson Preset Includes:**
- CORE + VIDEO + IMAGE_PROCESSING + CUDA_COMPONENT + ARM64_COMPONENT
- Optimized for Jetson hardware acceleration

---

## Component-Module Matrix

### Legend
- ✅ **Module is part of this component**
- 🔧 **Infrastructure module** (works with/without backend)
- 🎮 **Has GPU variant** (uses CUDA when available)
- 🦾 **Jetson-specific** (ARM64 only)

| Module | CORE | VIDEO | IMAGE_PROC | CUDA | ARM64 | Notes |
|--------|------|-------|------------|------|-------|-------|
| **Pipeline Infrastructure** |
| Module | ✅ | | | | | Base class for all modules |
| Frame | ✅ | | | | | Frame data structure |
| FrameFactory | ✅ 🔧 | | | | | Memory management (uses CUDA allocators if available) |
| FrameContainerQueue | ✅ | | | | | Frame queue management |
| PipeLine | ✅ | | | | | Pipeline orchestration |
| **Utilities** |
| Logger | ✅ | | | | | Boost.Log wrapper |
| Utils | ✅ 🔧 | | | | | Utility functions (requires OpenCV for image math) |
| ImageMetadata | ✅ 🔧 | | | | | Image metadata (requires OpenCV types) |
| APErrorObject | ✅ | | | | | Error handling |
| APHealthObject | ✅ | | | | | Health monitoring |
| **Basic I/O** |
| FileReaderModule | ✅ | | | | | Generic file reader |
| FileWriterModule | ✅ | | | | | Generic file writer |
| FileSequenceDriver | ✅ | | | | | File sequence handling |
| FilenameStrategy | ✅ | | | | | Filename pattern strategies |
| FIndexStrategy | ✅ | | | | | Frame index strategies |
| **Control Flow** |
| Split | ✅ | | | | | Split frame stream |
| Merge | ✅ | | | | | Merge frame streams |
| FramesMuxer | ✅ | | | | | Multiplex frames |
| ValveModule | ✅ | | | | | Control frame flow |
| SimpleControlModule | ✅ | | | | | Simple control logic |
| AbsControlModule | ✅ | | | | | Abstract control base |
| **Test Utilities** |
| TestSignalGeneratorSrc | ✅ | | | | | Generate test frames |
| **CUDA Infrastructure** (when ENABLE_CUDA=ON) |
| apra_cudamalloc_allocator | ✅ 🔧 | | | | | CUDA device memory allocator |
| apra_cudamallochost_allocator | ✅ 🔧 | | | | | CUDA pinned host memory allocator |
| **Video Codecs & Streaming** |
| Mp4ReaderSource | | ✅ | | | | Mp4 file reader |
| Mp4WriterSink | | ✅ | | | | Mp4 file writer |
| Mp4WriterSinkUtils | | ✅ | | | | Mp4 writing utilities |
| OrderedCacheOfFiles | | ✅ | | | | File cache for seeking |
| H264FrameDemuxer | | ✅ | | | | H264 frame demuxing |
| H264ParserUtils | | ✅ | | | | H264 parsing utilities |
| H264Utils | | ✅ | | | | H264 utilities |
| RTSPPusher | | ✅ | | | | RTSP stream pusher |
| RTSPClientSrc | | ✅ | | | | RTSP stream client |
| MultimediaQueueXform | | ✅ | | | | Multimedia queue |
| MotionVectorExtractor | | ✅ | | | | Motion vector extraction |
| VirtualCameraSink | | ✅ | | | | Virtual camera (Linux only) |
| **CPU Image Processing** |
| ImageDecoderCV | | | ✅ | | | OpenCV image decoder |
| ImageEncoderCV | | | ✅ | | | OpenCV image encoder |
| ImageResizeCV | | | ✅ | | | OpenCV resize |
| RotateCV | | | ✅ | | | OpenCV rotation |
| BMPConverter | | | ✅ | | | BMP conversion |
| AffineTransform | | | ✅ 🎮 | | | Affine transform (GPU-accelerated when CUDA enabled) |
| BrightnessContrastControlXform | | | ✅ | | | Brightness/contrast |
| VirtualPTZ | | | ✅ | | | Virtual pan-tilt-zoom |
| ColorConversionXForm | | | ✅ | | | Color space conversion |
| AbsColorConversionFactory | | | ✅ | | | Color conversion factory |
| Overlay | | | ✅ | | | Image overlay |
| OverlayFactory | | | ✅ | | | Overlay factory |
| OverlayModule | | | ✅ | | | Overlay module |
| TextOverlayXForm | | | ✅ | | | Text overlay |
| CalcHistogramCV | | | ✅ | | | Histogram calculation |
| HistogramOverlay | | | ✅ | | | Histogram overlay |
| ApraLines | | | ✅ | | | Line drawing |
| ArchiveSpaceManager | | | ✅ | | | Storage management |
| **GPU Processing (CUDA)** |
| CudaMemCopy | | | | ✅ | | CUDA memory copy |
| MemTypeConversion | | | | ✅ | | Host/Device/DMA conversion |
| CudaStreamSynchronize | | | | ✅ | | CUDA stream sync |
| CuCtxSynchronize | | | | ✅ | | CUDA context sync |
| CudaCommon | | | | ✅ | | CUDA utilities |
| ResizeNPPI | | | | ✅ | | NVIDIA NPP resize |
| RotateNPPI | | | | ✅ | | NVIDIA NPP rotation |
| OverlayNPPI | | | | ✅ | | NVIDIA NPP overlay |
| CCNPPI | | | | ✅ | | NVIDIA NPP color conversion |
| EffectsNPPI | | | | ✅ | | NVIDIA NPP effects |
| CCKernel | | | | ✅ | | Color conversion CUDA kernel |
| EffectsKernel | | | | ✅ | | Effects CUDA kernel |
| OverlayKernel | | | | ✅ | | Overlay CUDA kernel |
| build_point_list | | | | ✅ | | Point list CUDA kernel |
| JPEGEncoderNVJPEG | | | | ✅ | | NVIDIA NVJPEG encoder |
| JPEGDecoderNVJPEG | | | | ✅ | | NVIDIA NVJPEG decoder |
| H264EncoderNVCodec | | | | ✅ | | NVIDIA NVCODEC H264 encoder |
| H264EncoderNVCodecHelper | | | | ✅ | | NVCODEC encoder helper |
| H264Decoder | | | | ✅ | | NVIDIA H264 decoder |
| H264DecoderNvCodecHelper | | | | ✅ | | NVCODEC decoder helper |
| GaussianBlur | | | | ✅ | | CUDA Gaussian blur |
| **Jetson Hardware (ARM64)** |
| JPEGEncoderL4TM | | | | | ✅ 🦾 | L4TM JPEG encoder |
| JPEGEncoderL4TMHelper | | | | | ✅ 🦾 | L4TM encoder helper |
| JPEGDecoderL4TM | | | | | ✅ 🦾 | L4TM JPEG decoder |
| JPEGDecoderL4TMHelper | | | | | ✅ 🦾 | L4TM decoder helper |
| H264EncoderV4L2 | | | | | ✅ 🦾 | V4L2 H264 encoder |
| H264EncoderV4L2Helper | | | | | ✅ 🦾 | V4L2 encoder helper |
| H264DecoderV4L2Helper | | | | | ✅ 🦾 | V4L2 decoder helper |
| V4L2CUYUV420Converter | | | | | ✅ 🦾 | V4L2 YUV420 converter |
| AV4L2Buffer | | | | | ✅ 🦾 | V4L2 buffer wrapper |
| AV4L2ElementPlane | | | | | ✅ 🦾 | V4L2 plane wrapper |
| NvArgusCamera | | | | | ✅ 🦾 | Jetson camera (Argus) |
| NvArgusCameraHelper | | | | | ✅ 🦾 | Argus camera helper |
| NvV4L2Camera | | | | | ✅ 🦾 | Jetson camera (V4L2) |
| NvV4L2CameraHelper | | | | | ✅ 🦾 | V4L2 camera helper |
| EglRenderer | | | | | ✅ 🦾 | EGL renderer |
| NvEglRenderer | | | | | ✅ 🦾 | NVIDIA EGL renderer |
| ApraEGLDisplay | | | | | ✅ 🦾 | EGL display wrapper |
| DMAFDWrapper | | | | | ✅ 🦾 | DMA-BUF wrapper |
| DMAUtils | | | | | ✅ 🦾 | DMA utilities |
| DMAFDToHostCopy | | | | | ✅ 🦾 | DMA to host copy |
| NvTransform | | | | | ✅ 🦾 | Jetson transform |
| **Specialized Components** |
| WebCamSource | | | | | | See WEBCAM |
| QRReader | | | | | | See QR |
| AudioCaptureSrc | | | | | | See AUDIO |
| AudioToTextXForm | | | | | | See AUDIO |
| FaceDetectorXform | | | | | | See FACE_DETECTION |
| FacialLandmarksCV | | | | | | See FACE_DETECTION |
| GtkGlRenderer | | | | | | See GTK_RENDERING |
| GTKMatrix, GTKModel, GTKSetup, GTKView, Background | | | | | | See GTK_RENDERING |
| ThumbnailListGenerator | | | | | | See THUMBNAIL |
| ImageViewerModule | | | | | | See IMAGE_VIEWER |

---

## Infrastructure Backends

### CUDA Infrastructure (Conditional Compilation)

Some modules are "infrastructure" components that adapt based on whether CUDA is enabled:

#### Memory Allocators (CORE Component)
**Conditional Inclusion:** Only when `ENABLE_CUDA=ON`

```cpp
// These are part of CORE but only compiled with CUDA
apra_cudamalloc_allocator      // Device memory allocator
apra_cudamallochost_allocator  // Pinned host memory allocator
```

**Usage in FrameFactory:**
```cpp
// FrameFactory automatically uses CUDA allocators when available
// Falls back to standard allocators when CUDA is disabled
#ifdef APRA_CUDA_ENABLED
    // Use CUDA allocators
#else
    // Use standard host allocators
#endif
```

#### AffineTransform (IMAGE_PROCESSING Component)
**Dual Implementation:** CPU + GPU

```cpp
// IMAGE_PROCESSING module with optional GPU acceleration
AffineTransform:
  - CPU implementation: Uses OpenCV
  - GPU implementation: Uses NVIDIA NPP (when ENABLE_CUDA=ON)

// Runtime selection based on frame memory type
if (frame is on GPU) {
    use NPP implementation
} else {
    use OpenCV CPU implementation
}
```

**Build Requirements:**
- Always requires IMAGE_PROCESSING component
- NPP libraries linked when `ENABLE_CUDA=ON`
- Works without CUDA (CPU-only mode)

---

### Platform-Specific Infrastructure

#### Linux-Only Modules
```cpp
CORE:
  - KeyboardListener           // GTK keyboard events

VIDEO:
  - VirtualCameraSink          // V4L2 loopback device

GTK_RENDERING:
  - All GTK/OpenGL modules     // X11 GUI rendering
```

#### Windows-Only Considerations
```cpp
// No Windows-specific modules currently
// All CORE, VIDEO, IMAGE_PROCESSING modules work on Windows
```

#### Jetson-Only (ARM64_COMPONENT)
```cpp
// All ARM64_COMPONENT modules require:
// - ARM64 architecture
// - Jetson L4T libraries
// - ENABLE_ARM64=ON
```

---

## Common Use Cases

### 1. Video File Processing (No GPU)
**Scenario:** Read Mp4 files, process frames, write to new Mp4

```bash
# Windows
build_windows_no_cuda.bat --preset video

# Linux
./build_linux_no_cuda.sh --preset video
```

**Components:** CORE + VIDEO + IMAGE_PROCESSING
**Build Time:** ~25-30 min
**Modules Available:** Mp4 I/O, H264 codec (CPU), OpenCV processing

---

### 2. Real-Time RTSP Streaming with GPU Acceleration
**Scenario:** RTSP client, GPU processing, RTSP output

```bash
# Windows
build_windows_cuda.bat --preset cuda

# Linux
./build_linux_cuda.sh --preset cuda
```

**Components:** CORE + VIDEO + IMAGE_PROCESSING + CUDA_COMPONENT
**Build Time:** ~15-20 min (with vcpkg cache)
**Modules Available:** RTSP client/pusher, NVCODEC, NPP processing

---

### 3. Webcam Application with Face Detection
**Scenario:** Webcam capture, face detection, display

```bash
# Windows
build_windows_cuda.bat --components "CORE;IMAGE_PROCESSING;WEBCAM;FACE_DETECTION;IMAGE_VIEWER"

# Linux
./build_linux_cuda.sh --components "CORE IMAGE_PROCESSING WEBCAM FACE_DETECTION IMAGE_VIEWER"
```

**Components:** CORE + IMAGE_PROCESSING + WEBCAM + FACE_DETECTION + IMAGE_VIEWER
**Build Time:** ~20-25 min

---

### 4. Jetson Camera Application
**Scenario:** NvArgus camera, GPU processing, EGL display

```bash
./build_jetson.sh --preset jetson
```

**Components:** CORE + VIDEO + IMAGE_PROCESSING + CUDA_COMPONENT + ARM64_COMPONENT
**Build Time:** ~30-40 min
**Hardware:** Jetson Nano/Xavier/Orin

---

### 5. Audio Transcription Pipeline
**Scenario:** Audio capture with Whisper transcription

```bash
# Warning: Whisper build takes 30-40 minutes
build_windows_cuda.bat --components "CORE;AUDIO"
```

**Components:** CORE + AUDIO
**Build Time:** ~40-50 min (Whisper is slow to build)

---

### 6. Minimal Pipeline Development
**Scenario:** Testing pipeline logic, no media processing

```bash
build_windows_cuda.bat --preset minimal
```

**Components:** CORE only
**Build Time:** ~10-15 min
**Use Case:** Plugin development, pipeline testing, CI/CD

---

### 7. Thumbnail Generation Service
**Scenario:** Generate thumbnails from images/videos

```bash
build_windows_no_cuda.bat --components "CORE;IMAGE_PROCESSING;THUMBNAIL"
```

**Components:** CORE + IMAGE_PROCESSING + THUMBNAIL
**Build Time:** ~15-20 min

---

## Build Time Comparison

### Windows with CUDA (tested on Phase 5.5)

| Configuration | Components | Build Time | vcpkg Packages |
|--------------|------------|------------|----------------|
| **Minimal** | CORE | ~10-15 min | 42 |
| **Video** | CORE+VIDEO+IMAGE_PROCESSING | ~25-30 min | 48 |
| **CUDA** | CORE+VIDEO+IMAGE_PROCESSING+CUDA | ~15-20 min* | 117 |
| **Full** | ALL (12 components) | ~60-90 min | 120+ |

*Faster than VIDEO because most packages cached from previous builds

### Incremental Build Times

| Scenario | Time |
|----------|------|
| No changes | <1 min |
| Single file edit | <1 min |
| Module added | 2-5 min |
| Component added | Depends on dependencies |

### vcpkg Cache Benefits

**First Build:**
- Downloads and compiles all dependencies
- ~50% of total build time

**Subsequent Builds:**
- Restores packages from cache
- ~2-3 min for package restoration
- Significant time savings

**Tip:** Preserve `vcpkg/installed` directory to avoid redownloading packages

---

## Troubleshooting

### Build Fails: "Could NOT find OpenCV"

**Problem:** OpenCV not found during CMake configuration

**Solution:**
```bash
# Clean build directory
rm -rf _build _debugbuild

# Rebuild from scratch
./build_linux_cuda.sh --preset video
```

**Cause:** vcpkg cache corruption or incomplete installation

---

### Build Fails: "unresolved external symbol CUDA allocator"

**Problem:** CUDA allocators missing from CORE

**Solution:**
This should be fixed in Phase 5.5. Ensure you're on the latest code:
```bash
git pull origin feature/component-based-build
```

**Verification:**
```bash
# Check that CUDA allocators are in CORE
grep -n "apra_cudamalloc_allocator" base/CMakeLists.txt
# Should show allocators in CORE section (around line 698)
```

---

### Build Fails: "nppiWarpAffine" unresolved symbol

**Problem:** NPP libraries not linked for IMAGE_PROCESSING

**Solution:**
This should be fixed in Phase 5.5. Verify NPP linking:
```bash
# Check CMakeLists.txt has NPP linking for IMAGE_PROCESSING
grep -A5 "IMAGE_PROCESSING.*NPP" base/CMakeLists.txt
```

---

### Runtime Error: "Missing DLL" (Windows)

**Problem:** vcpkg DLLs not in PATH

**Solution:**
```bash
# Run from repository root
cd D:\dws\apra_fw

# vcpkg DLLs are in:
_build\vcpkg_installed\x64-windows\bin
_build\vcpkg_installed\x64-windows\debug\bin

# Or run executable from _build directory
cd _build\RelWithDebInfo
.\aprapipesut.exe
```

---

### Build Takes Too Long

**Problem:** Building unnecessary components

**Solution:**
```bash
# Use targeted presets instead of full build
# Bad:
build_windows_cuda.bat --preset full  # 60-90 min

# Good:
build_windows_cuda.bat --preset video  # 25-30 min
```

**Build Time Tips:**
1. Use presets for common configurations
2. Don't include AUDIO unless needed (Whisper is slow)
3. Leverage vcpkg cache (don't delete `vcpkg/installed`)
4. Use `--preset minimal` for development iteration

---

### Disk Space Issues

**Problem:** vcpkg cache fills disk

**Monitor disk usage:**
```bash
# Windows
dir _build\vcpkg_installed /s

# Linux
du -sh _build/vcpkg_installed
```

**Clean up:**
```bash
# Remove build directories (safe)
rm -rf _build _debugbuild

# Remove vcpkg packages (will require redownload)
rm -rf vcpkg/installed vcpkg/buildtrees
```

**vcpkg Disk Usage:**
- Minimal build: ~8-10 GB
- Video build: ~12-15 GB
- CUDA build: ~30-40 GB
- Full build: ~50-60 GB

---

### CMake Configuration Fails

**Problem:** Component dependencies not satisfied

**Example Error:**
```
Component CUDA_COMPONENT requires IMAGE_PROCESSING but it is not enabled
```

**Solution:**
```bash
# Include required dependencies
# Bad:
--components "CORE;CUDA_COMPONENT"

# Good:
--components "CORE;IMAGE_PROCESSING;CUDA_COMPONENT"
```

**Component Dependencies:**
- CUDA_COMPONENT requires: CORE, IMAGE_PROCESSING
- ARM64_COMPONENT requires: CORE, CUDA_COMPONENT
- VIDEO requires: CORE
- IMAGE_PROCESSING requires: CORE
- Most components require: CORE

---

## Advanced Usage

### Custom CMake Configuration

For fine-grained control, use CMake directly:

```bash
mkdir _build && cd _build

cmake -G "Visual Studio 16 2019" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DENABLE_CUDA=ON \
  -DENABLE_WINDOWS=ON \
  -DENABLE_COMPONENTS="CORE;VIDEO;IMAGE_PROCESSING;CUDA_COMPONENT" \
  -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
  -A x64 \
  ../base

cmake --build . --config RelWithDebInfo
```

---

### Verifying Component Selection

Check which components are enabled:

```bash
# After CMake configuration, check output:
-- Building selected components: CORE VIDEO IMAGE_PROCESSING CUDA_COMPONENT
--   - Enabling component: CORE
--   - Enabling component: VIDEO
--   - Enabling component: IMAGE_PROCESSING
--   - Enabling component: CUDA_COMPONENT
-- Component configuration complete
-- Building with 180 source files
```

**Source File Counts:**
- CORE only: 77 files
- CORE+VIDEO+IMAGE_PROCESSING: 139 files
- CORE+VIDEO+IMAGE_PROCESSING+CUDA: 180 files

---

### Running Tests by Component

```bash
# List all available tests
.\_build\RelWithDebInfo\aprapipesut.exe --list_content

# Run CORE tests only
.\_build\RelWithDebInfo\aprapipesut.exe --run_test=unit_tests/*

# Run VIDEO tests
.\_build\RelWithDebInfo\aprapipesut.exe --run_test=mp4_*

# Run CUDA tests
.\_build\RelWithDebInfo\aprapipesut.exe --run_test=*nppi*
.\_build\RelWithDebInfo\aprapipesut.exe --run_test=*nvjpeg*
```

---

## Next Steps

1. **Choose Your Configuration:**
   Review [Common Use Cases](#common-use-cases) and select the preset closest to your needs

2. **Run Build Script:**
   Use the appropriate script for your platform

3. **Verify Build:**
   ```bash
   # List tests to verify components
   ./_build/aprapipesut --list_content
   ```

4. **Develop Your Application:**
   - Instantiate modules from enabled components
   - Connect modules in pipelines
   - Process frames through the pipeline

5. **Iterate Quickly:**
   - Use `--preset minimal` for fast iteration
   - Add components as needed
   - Rebuild incrementally

---

## Additional Resources

- **Component Details:** See `COMPONENT_REFACTORING_LOG.md`
- **Testing Report:** See `TESTING_PHASE5.5_REPORT.md`
- **Module API:** See `base/include/*.h` header files
- **Examples:** See `base/test/*_tests.cpp` for usage patterns
- **Build Scripts:** `build_windows_cuda.bat`, `build_linux_cuda.sh`, `build_jetson.sh`

---

**Last Updated:** 2025-10-09
**Component System Version:** Phase 5.5 Complete
