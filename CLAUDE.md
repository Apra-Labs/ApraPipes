# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ApraPipes is a C++ pipeline framework for developing video and image processing applications with support for multiple GPUs and Machine Learning toolkits. The framework uses a modular architecture where processing modules are connected in pipelines to handle media streams.

## Initial Setup

**Important**: Clone with submodules and LFS:
```bash
git clone --recursive https://github.com/Apra-Labs/ApraPipes.git
```

Update submodules if needed:
```bash
git submodule update --init --recursive
```

## Component-Based Build System

ApraPipes now supports building only the components you need, significantly reducing build times and dependencies.

### Available Components
- **CORE** (always required): Pipeline infrastructure, basic I/O
- **VIDEO**: Mp4, H264, RTSP (requires FFmpeg)
- **IMAGE_PROCESSING**: OpenCV CPU-based processing
- **CUDA_COMPONENT**: GPU acceleration (requires CUDA)
- **ARM64_COMPONENT**: Jetson-specific modules
- **WEBCAM**: Webcam capture
- **QR**: QR code reading
- **AUDIO**: Audio capture and transcription
- **FACE_DETECTION**: Face detection and landmarks
- **GTK_RENDERING**: Linux GUI rendering
- **THUMBNAIL**: Thumbnail generation
- **IMAGE_VIEWER**: Image viewing GUI

### Component Selection Examples

**Minimal build** (pipeline only, ~5-10 min):
```bash
cmake -DENABLE_COMPONENTS=CORE -DENABLE_CUDA=OFF ../base
```

**Video processing** (no GPU, ~15-25 min):
```bash
cmake -DENABLE_COMPONENTS="CORE;VIDEO;IMAGE_PROCESSING" -DENABLE_CUDA=OFF ../base
```

**CUDA-enabled build**:
```bash
cmake -DENABLE_COMPONENTS="CORE;VIDEO;IMAGE_PROCESSING;CUDA_COMPONENT" -DENABLE_CUDA=ON ../base
```

**Full build** (backward compatible, default):
```bash
cmake ../base
# or explicitly:
cmake -DENABLE_COMPONENTS=ALL ../base
```

See `COMPONENT_REFACTORING_LOG.md` for detailed component information and `TESTING_VALIDATION.md` for testing guidelines.

## Build Commands

### Windows with CUDA
```bash
build_windows_cuda.bat
# Debug build: _debugbuild/Debug/aprapipesut.exe
# Release build: _build/RelWithDebInfo/aprapipesut.exe
```

**CUDA 11.8 Compatibility Note:**
The build script automatically detects CUDA 11.8 and selects a compatible Visual Studio version:
- **Priority 1**: Visual Studio 2019 (all editions) - Most compatible
- **Priority 2**: Visual Studio 2022 v17.0 - v17.3 - Compatible range
- **Warning**: Visual Studio 2022 v17.4+ is incompatible with CUDA 11.8

The script uses `vswhere.exe` to detect compatible VS installations and configures the build accordingly. No manual configuration needed.

### Windows without CUDA
```bash
build_windows_no_cuda.bat
```

### Linux with CUDA
```bash
chmod +x build_linux_cuda.sh
./build_linux_cuda.sh
# Output: _build/aprapipesut
```

### Linux without CUDA
```bash
chmod +x build_linux_no_cuda.sh
./build_linux_no_cuda.sh
```

### Jetson (ARM64)
```bash
chmod +x build_jetson.sh
./build_jetson.sh
```

## Test Commands

### List all tests
```bash
# Windows
_build/RelWithDebInfo/aprapipesut.exe --list_content

# Linux/Jetson
./_build/aprapipesut --list_content
```

### Run all tests
```bash
# Windows
_build/RelWithDebInfo/aprapipesut.exe -p -l all --detect_memory_leaks=0

# Linux/Jetson
./_build/aprapipesut -p -l all --detect_memory_leaks=0
```

### Run specific test suite/case
```bash
# Windows
_build/RelWithDebInfo/aprapipesut.exe --run_test=filenamestrategy_tests/boostdirectorystrategy

# Linux
./_build/aprapipesut --run_test=filenamestrategy_tests/boostdirectorystrategy
```

### Run test with arguments
```bash
# Windows
_build/RelWithDebInfo/aprapipesut.exe --run_test=unit_tests/params_test -- -ip 10.102.10.121 -data ArgusCamera

# Linux
./_build/aprapipesut --run_test=unit_tests/params_test -- -ip 10.102.10.121 -data ArgusCamera
```

## Code Formatting & Static Analysis

The project includes pre-commit hooks and GitHub Actions for CI checks.

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

The repository includes:
- **pre-commit config**: `.pre-commit-config.yaml`
- **GitHub Actions**: Automated CI checks in `.github/workflows/`

### Documentation

Build documentation using:
```bash
# Linux/Jetson
./build_documentation.sh

# Windows/Linux - Include docs in build
build_windows_cuda.bat --build-doc
./build_linux_cuda.sh --build-doc
```

## Architecture Overview

### Core Components

1. **Module System** (`base/include/Module.h`): Base class for all processing modules. Modules can be:
   - Source modules (generate frames)
   - Transform modules (process frames)
   - Sink modules (consume frames)
   - Control modules (manage pipeline flow)

2. **Frame Management** (`base/include/Frame.h`, `FrameFactory.h`): 
   - Frames carry data through the pipeline
   - FrameFactory manages memory allocation/pooling
   - Supports various memory types (Host, CUDA, DMA)

3. **Pipeline** (`base/include/PipeLine.h`): 
   - Manages module connections and execution
   - Handles data flow between modules
   - Supports multi-threaded execution

4. **Metadata System** (`base/include/FrameMetadata.h`):
   - Each frame type has associated metadata
   - Describes frame properties (dimensions, format, etc.)
   - Used for type checking and format negotiation

### Module Categories

- **Core Modules** (`src/`): FileReader/Writer, FramesMuxer, Split, Merge
- **Image Processing** (`src/`): ImageResize, ColorConversion, AffineTransform, BrightnessContrast
- **Video Codecs**: H264Encoder/Decoder (NVCODEC on GPU, V4L2 on Jetson)
- **Media I/O**: Mp4Reader/Writer, RTSPClient/Pusher, WebCamSource
- **CV/ML**: FaceDetector, QRReader, AudioToText

### Platform-Specific Code

- **CUDA**: Enabled with `ENABLE_CUDA` - NPP/NVJPEG processing, NVCODEC
- **ARM64/Jetson**: Enabled with `ENABLE_ARM64` - V4L2, L4T multimedia APIs
- **Windows**: Enabled with `ENABLE_WINDOWS` - Windows-specific implementations
- **Linux**: Enabled with `ENABLE_LINUX` - GTK/GL rendering, virtual camera

## Build System Details

### CMake Configuration

Main CMakeLists.txt is in `base/` directory with options:
- `ENABLE_CUDA`: GPU acceleration support
- `ENABLE_WINDOWS`: Windows platform build
- `ENABLE_LINUX`: Linux platform build  
- `ENABLE_ARM64`: Jetson/ARM64 build

### Dependencies

Managed through vcpkg (`base/vcpkg.json`):
- Boost (system, thread, filesystem, serialization, log, test, chrono, iostreams, dll, format, foreach)
- OpenCV 4.x with CUDA/cuDNN/contrib
- FFmpeg 4.4.3, OpenH264
- ZXing (QR codes)
- Whisper (audio transcription with CUDA)
- SFML (audio)
- Platform-specific: GTK3 (Linux), glib, hiredis/redis-plus-plus (non-ARM64)

**vcpkg Baseline**: Pinned to `4658624c5f19c1b468b62fe13ed202514dfd463e` for reproducible builds.

**Important**: The project includes a custom `base/cmake/FindCUDA.cmake` compatibility module that bridges legacy FindCUDA.cmake (used by vcpkg's OpenCV) with modern CMake CUDA support. This is required for CUDA 11.8+ builds.

### Build Output Structure

```
_build/                   # Release build directory
  aprapipesut(.exe)      # Test executable
  libaprapipes.a/.lib    # Static library
_debugbuild/             # Debug build directory
  Debug/
    aprapipesut(.exe)
```

## Testing Approach

Tests use Boost.Test framework. Each module typically has a corresponding test file in `base/test/`. Test naming convention: `<module>_tests.cpp`.

Test files are organized by test suites (BOOST_AUTO_TEST_SUITE) containing individual test cases (BOOST_AUTO_TEST_CASE).

## Key Development Patterns

1. **Module Creation**: Inherit from Module class, implement init(), term(), produce()/consume()/transform()
2. **Props Pattern**: Each module has associated Props class for configuration
3. **Metadata Handling**: Always set output metadata in init() or produce()
4. **Memory Management**: Use FrameFactory for allocation, proper memory type handling
5. **Error Handling**: Use AIPException for errors, return proper APErrorCode

## Common Development Tasks

When modifying modules:
1. Update both header (.h) and implementation (.cpp/.cu) files
2. Add/update tests in `base/test/`
3. Ensure proper metadata handling
4. Handle both CUDA and non-CUDA code paths where applicable
5. Follow existing patterns for Props classes and factory methods

## Troubleshooting

### CUDA 11.8 Build Issues on Windows

**Problem**: Build fails with "unsupported Microsoft Visual Studio version" error
**Cause**: CUDA 11.8 is incompatible with Visual Studio 2022 v17.4+

**Solution**: The build system automatically handles this, but if you encounter issues:

1. **Install Visual Studio 2019** (recommended) - Most compatible with CUDA 11.8
   - Download from: https://visualstudio.microsoft.com/vs/older-downloads/

2. **OR use Visual Studio 2022 v17.0 - v17.3**
   - The build script will automatically detect and use compatible versions
   - Versions 17.4+ are not compatible with CUDA 11.8

3. **How it works**:
   - `build_windows_cuda.bat` automatically detects CUDA 11.8
   - Searches for VS 2019 first (priority 1)
   - Falls back to compatible VS 2022 versions (v17.0-v17.3)
   - Uses `vswhere.exe` for version detection
   - Sets appropriate CMake generator and toolset

4. **Manual override** (if needed):
   ```bash
   # Force VS 2019
   cmake -G "Visual Studio 16 2019" -DENABLE_CUDA=ON ...

   # Force VS 2022 (only if compatible version)
   cmake -G "Visual Studio 17 2022" -DENABLE_CUDA=ON ...
   ```

### Technical Details: FindCUDA.cmake Compatibility

The project includes `base/cmake/FindCUDA.cmake` which provides compatibility between:
- **Legacy**: FindCUDA.cmake module (expected by vcpkg's OpenCV 4.8)
- **Modern**: CMake's native CUDA language support (CUDA 11.8+)

This module:
- Maps legacy `find_package(CUDA)` to modern `find_package(CUDAToolkit)`
- Provides `find_cuda_helper_libs()` function for library discovery
- Translates library names (e.g., `nppc`, `nppial`) to CUDA:: targets
- Automatically enabled via `CMAKE_MODULE_PATH` in base/CMakeLists.txt
- We are adding a new feature in this repository. The intention is to simplify the build structure of the repository.
- No actually, I had planned a Phase 5.5 called local testing. Here is the instruction for the phase: actually, before you go to phase 6, perform an extensive local testing phase for windows for all different combinations. Take as much     
time as you need. Make sure the disk space does not get full. Try different components. While testing make sure that not only the builds    
 are successful but also there are no runtime issues like linking issues, missing DLL issues. And the tests are running for 
ReleaseWithDebugInfo etc. Generate a report output of this testing phase. Once we are done with this, we will move to phase 6 and phase   7

We MUST generate a separate documentation which is like a developer guide to adding a module to this framework so as to navigate and decide which part of the CMakelists need to be edited for that specific module based on the COMPONENTS that we end up having finally. THIS MUST BE DONE at the end of all the phases. Lets call it THE LAST PHASE - generating a developement guide for future human developers.

CUDA preset is of high importanceI 
- CUDA Preset is important