# ApraPipes Developer Guide: Adding New Modules

This guide helps developers add new modules to the ApraPipes framework and integrate them properly with the component-based build system.

---

## Table of Contents

1. [Quick Start Checklist](#quick-start-checklist)
2. [Understanding the Component System](#understanding-the-component-system)
3. [Module Development Workflow](#module-development-workflow)
4. [CMakeLists.txt Integration](#cmakeliststxt-integration)
5. [vcpkg Dependency Management](#vcpkg-dependency-management)
6. [Writing Tests](#writing-tests)
7. [Platform-Specific Considerations](#platform-specific-considerations)
8. [Examples](#examples)
9. [Common Pitfalls](#common-pitfalls)

---

## Quick Start Checklist

When adding a new module, follow these steps:

- [ ] **Step 1**: Determine which component your module belongs to
- [ ] **Step 2**: Create your module files (`.h` and `.cpp`/`.cu`)
- [ ] **Step 3**: Add files to appropriate `COMPONENT_<NAME>_FILES` list in `base/CMakeLists.txt`
- [ ] **Step 4**: Add any new dependencies to `base/vcpkg.json`
- [ ] **Step 5**: Create unit tests in `base/test/`
- [ ] **Step 6**: Add tests to `COMPONENT_<NAME>_UT_FILES` list in `base/CMakeLists.txt`
- [ ] **Step 7**: Build and test your component
- [ ] **Step 8**: Document your module

---

## Understanding the Component System

### Component Overview

ApraPipes has 12 components:

| Component | Purpose | Typical Modules |
|-----------|---------|----------------|
| **CORE** | Pipeline infrastructure | Module, Frame, FrameFactory, FileReader/Writer |
| **VIDEO** | Video codecs & streaming | Mp4Reader/Writer, H264Encoder/Decoder, RTSPClient |
| **IMAGE_PROCESSING** | CPU image processing | ImageResize, ColorConversion, Overlay |
| **CUDA_COMPONENT** | GPU acceleration | NPP processors, NVJPEG codecs |
| **ARM64_COMPONENT** | Jetson hardware | V4L2 codecs, NvArgusCamera |
| **WEBCAM** | Webcam capture | WebCamSource |
| **QR** | QR code reading | QRReader |
| **AUDIO** | Audio capture/transcription | AudioCaptureSrc, AudioToTextXForm |
| **FACE_DETECTION** | Face detection | FaceDetectorXform |
| **GTK_RENDERING** | Linux GUI rendering | GtkGlRenderer |
| **THUMBNAIL** | Thumbnail generation | ThumbnailListGenerator |
| **IMAGE_VIEWER** | Image viewing | ImageViewerModule |

### Choosing the Right Component

Ask yourself these questions:

1. **What does your module do?**
   - Pipeline infrastructure → CORE
   - Video codec/streaming → VIDEO
   - CPU image processing → IMAGE_PROCESSING
   - GPU acceleration → CUDA_COMPONENT
   - Platform-specific → ARM64_COMPONENT / GTK_RENDERING

2. **What are your dependencies?**
   - OpenCV only → IMAGE_PROCESSING
   - FFmpeg → VIDEO
   - CUDA/NPP → CUDA_COMPONENT
   - ZXing → QR
   - Whisper → AUDIO

3. **Is it platform-specific?**
   - Jetson only → ARM64_COMPONENT
   - Linux GUI → GTK_RENDERING
   - Cross-platform → Choose based on functionality

---

## Module Development Workflow

### 1. Create Module Files

**Header File** (`base/include/YourModule.h`):
```cpp
#ifndef _YOUR_MODULE_H
#define _YOUR_MODULE_H

#include "Module.h"
#include "FrameMetadata.h"

class YourModuleProps : public ModuleProps {
public:
    YourModuleProps() {}
    // Add your properties
};

class YourModule : public Module {
public:
    YourModule(YourModuleProps _props);
    virtual ~YourModule();

protected:
    bool init() override;
    bool term() override;
    bool process(frame_container& frames) override;
    bool validateInputPins() override;
    bool validateOutputPins() override;

private:
    class Detail;
    boost::shared_ptr<Detail> mDetail;
};

#endif
```

**Implementation File** (`base/src/YourModule.cpp` or `.cu` for CUDA):
```cpp
#include "YourModule.h"

class YourModule::Detail {
public:
    Detail(YourModuleProps& props) {}
    ~Detail() {}
    // Implementation
};

YourModule::YourModule(YourModuleProps _props) : Module(TRANSFORM, "YourModule", _props) {
    mDetail.reset(new Detail(_props));
}

YourModule::~YourModule() {}

bool YourModule::init() {
    // Initialize your module
    return true;
}

bool YourModule::term() {
    // Cleanup
    return true;
}

bool YourModule::process(frame_container& frames) {
    // Process frames
    return true;
}

bool YourModule::validateInputPins() {
    // Validate input metadata
    return true;
}

bool YourModule::validateOutputPins() {
    // Validate output metadata
    return true;
}
```

---

## CMakeLists.txt Integration

### Location: `base/CMakeLists.txt`

The CMakeLists.txt is organized into sections:

1. **Component Option System** (lines 1-100)
2. **Component File Lists** (lines 250-850)
3. **Conditional Dependencies** (lines 900-1100)
4. **Source Aggregation** (lines 1150-1250)
5. **Test File Lists** (lines 1250-1500)

### Adding Your Module

#### Step 1: Find Your Component's File List

Search for `COMPONENT_<YOUR_COMPONENT>_FILES`:

```cmake
# Example: Adding to IMAGE_PROCESSING
set(COMPONENT_IMAGE_PROCESSING_FILES
    src/ImageDecoderCV.cpp
    src/ImageEncoderCV.cpp
    src/ImageResizeCV.cpp
    src/YourModule.cpp        # <-- ADD HERE
    # ... existing files
)

set(COMPONENT_IMAGE_PROCESSING_FILES_H
    include/ImageDecoderCV.h
    include/ImageEncoderCV.h
    include/ImageResizeCV.h
    include/YourModule.h      # <-- ADD HERE
    # ... existing files
)
```

#### Step 2: CUDA Files (if applicable)

If your module uses CUDA (`.cu` files):

```cmake
# Example: Adding to CUDA_COMPONENT
set(COMPONENT_CUDA_COMPONENT_FILES
    src/JPEGEncoderNVJPEG.cu
    src/ResizeNPPI.cu
    src/YourCudaModule.cu     # <-- ADD HERE
    # ... existing files
)
```

#### Step 3: Platform-Specific Files

If your module is platform-specific:

```cmake
# Example: Linux-only module
IF(ENABLE_LINUX)
    set(COMPONENT_GTK_RENDERING_FILES
        src/GtkGlRenderer.cpp
        src/YourLinuxModule.cpp   # <-- ADD HERE
    )
ENDIF(ENABLE_LINUX)

# Example: Windows-only module
IF(ENABLE_WINDOWS)
    set(COMPONENT_VIDEO_FILES
        src/Mp4ReaderSource.cpp
        src/YourWindowsModule.cpp  # <-- ADD HERE
    )
ENDIF(ENABLE_WINDOWS)

# Example: ARM64-only module
IF(ENABLE_ARM64)
    set(COMPONENT_ARM64_COMPONENT_FILES
        src/NvArgusCamera.cpp
        src/YourJetsonModule.cpp   # <-- ADD HERE
    )
ENDIF(ENABLE_ARM64)
```

---

## vcpkg Dependency Management

### Location: `base/vcpkg.json`

If your module requires external libraries, add them to the appropriate vcpkg feature.

### Adding a Dependency

**Example 1: Adding to existing feature**

```json
{
  "features": {
    "image-processing": {
      "description": "OpenCV CPU-based image processing",
      "dependencies": [
        {
          "name": "opencv4",
          "default-features": false,
          "features": ["jpeg", "png", "tiff", "webp"]
        },
        "your-new-library"  // <-- ADD HERE
      ]
    }
  }
}
```

**Example 2: Creating a new feature for your module**

```json
{
  "features": {
    "your-component": {
      "description": "Your module description",
      "dependencies": [
        "your-dependency-1",
        "your-dependency-2"
      ]
    }
  }
}
```

### CMake Integration for Dependencies

After adding to vcpkg.json, update CMakeLists.txt to find and link the package:

```cmake
# Find the package (add to dependencies section ~line 900)
if(APRAPIPES_ENABLE_YOUR_COMPONENT)
    find_package(YourLibrary CONFIG REQUIRED)
endif()

# Link the package (add to linking section ~line 1200)
if(APRAPIPES_ENABLE_YOUR_COMPONENT)
    target_link_libraries(aprapipes YourLibrary::YourLibrary)
endif()
```

---

## Writing Tests

### Test File Structure

Create a test file in `base/test/yourmodule_tests.cpp`:

```cpp
#include "boost/test/unit_test.hpp"
#include "YourModule.h"
#include "TestUtils.h"

BOOST_AUTO_TEST_SUITE(yourmodule_tests)

BOOST_AUTO_TEST_CASE(basic_functionality) {
    // Setup
    YourModuleProps props;
    auto module = boost::shared_ptr<YourModule>(new YourModule(props));

    // Create test pipeline
    auto source = boost::shared_ptr<TestSource>(new TestSource());
    source->setNext(module);

    // Initialize
    BOOST_TEST(source->init());
    BOOST_TEST(module->init());

    // Test
    source->step();

    // Verify results
    // ... your assertions
}

BOOST_AUTO_TEST_CASE(error_handling) {
    // Test error conditions
}

BOOST_AUTO_TEST_SUITE_END()
```

### Adding Tests to CMakeLists.txt

Find your component's test section (~line 1250):

```cmake
# Example: IMAGE_PROCESSING tests
set(COMPONENT_IMAGE_PROCESSING_UT_FILES
    test/imageresizecv_tests.cpp
    test/colorconversion_tests.cpp
    test/yourmodule_tests.cpp    # <-- ADD HERE
)
```

### Test Best Practices

1. **Test each module in isolation** using test sources/sinks
2. **Test edge cases**: null frames, invalid metadata, error conditions
3. **Test platform-specific code** conditionally
4. **Memory leak testing**: Use Boost.Test memory leak detection
5. **Performance testing**: Measure processing time for large datasets

---

## Platform-Specific Considerations

### Windows-Specific Code

```cpp
#ifdef _WIN32
    // Windows-specific implementation
#endif
```

```cmake
# In CMakeLists.txt
IF(ENABLE_WINDOWS)
    list(APPEND COMPONENT_YOUR_COMPONENT_FILES src/YourWindowsModule.cpp)
ENDIF(ENABLE_WINDOWS)
```

### Linux-Specific Code

```cpp
#ifdef __linux__
    // Linux-specific implementation
#endif
```

### ARM64/Jetson-Specific Code

```cpp
#ifdef __aarch64__
    // ARM64-specific implementation
#endif
```

```cmake
# In CMakeLists.txt
IF(ENABLE_ARM64)
    list(APPEND COMPONENT_ARM64_COMPONENT_FILES src/YourJetsonModule.cpp)
    # ARM64 dependencies
    target_link_libraries(aprapipes nvargus_socketclient)
ENDIF(ENABLE_ARM64)
```

### CUDA-Specific Code

**File extension**: Use `.cu` for CUDA files

```cuda
// YourCudaModule.cu
__global__ void yourKernel() {
    // CUDA kernel implementation
}

void YourCudaModule::process() {
    yourKernel<<<blocks, threads>>>();
}
```

```cmake
# In CMakeLists.txt
IF(ENABLE_CUDA)
    list(APPEND COMPONENT_CUDA_COMPONENT_FILES src/YourCudaModule.cu)
    target_link_libraries(aprapipes CUDA::cudart CUDA::npp)
ENDIF(ENABLE_CUDA)
```

---

## Examples

### Example 1: Adding a Simple Image Processing Module

**Scenario**: Add a GrayscaleConverter module to IMAGE_PROCESSING

**Step 1: Create files**
- `base/include/GrayscaleConverter.h`
- `base/src/GrayscaleConverter.cpp`

**Step 2: Update CMakeLists.txt** (line ~400)
```cmake
set(COMPONENT_IMAGE_PROCESSING_FILES
    # ... existing files
    src/GrayscaleConverter.cpp
)

set(COMPONENT_IMAGE_PROCESSING_FILES_H
    # ... existing files
    include/GrayscaleConverter.h
)
```

**Step 3: Dependencies** (already satisfied - OpenCV included in IMAGE_PROCESSING)

**Step 4: Create test**
- `base/test/grayscaleconverter_tests.cpp`

**Step 5: Add test to CMakeLists.txt** (line ~1300)
```cmake
set(COMPONENT_IMAGE_PROCESSING_UT_FILES
    # ... existing tests
    test/grayscaleconverter_tests.cpp
)
```

**Step 6: Build and test**
```bash
# Windows
build_windows_cuda.bat --preset video

# Linux
./build_linux_cuda.sh --preset video
```

---

### Example 2: Adding a CUDA-Accelerated Module

**Scenario**: Add a BlurNPPI module using NVIDIA NPP

**Step 1: Create files**
- `base/include/BlurNPPI.h`
- `base/src/BlurNPPI.cu` (note `.cu` extension)

**Step 2: Update CMakeLists.txt** (line ~700)
```cmake
set(COMPONENT_CUDA_COMPONENT_FILES
    # ... existing files
    src/BlurNPPI.cu
)

set(COMPONENT_CUDA_COMPONENT_FILES_H
    # ... existing files
    include/BlurNPPI.h
)
```

**Step 3: Dependencies** (NPP already included)

**Step 4: Create test**
- `base/test/blurnppi_tests.cpp`

**Step 5: Add test to CMakeLists.txt** (line ~1400)
```cmake
set(COMPONENT_CUDA_COMPONENT_UT_FILES
    # ... existing tests
    test/blurnppi_tests.cpp
)
```

**Step 6: Build and test**
```bash
build_windows_cuda.bat --preset cuda
```

---

### Example 3: Adding a New Dependency

**Scenario**: Add TensorRT support for an AIInference module

**Step 1: Update vcpkg.json**
```json
{
  "features": {
    "ai-inference": {
      "description": "AI inference with TensorRT",
      "dependencies": [
        "tensorrt"
      ]
    }
  }
}
```

**Step 2: Update CMakeLists.txt - Add new component**
```cmake
# Add to component list (line ~30)
set(APRAPIPES_ALL_COMPONENTS
    CORE VIDEO IMAGE_PROCESSING CUDA_COMPONENT
    AI_INFERENCE  # <-- NEW
    # ... rest
)

# Add dependency (line ~950)
if(APRAPIPES_ENABLE_AI_INFERENCE)
    find_package(TensorRT CONFIG REQUIRED)
endif()

# Add source files (line ~850)
set(COMPONENT_AI_INFERENCE_FILES
    src/AIInferenceModule.cpp
)

# Add linking (line ~1220)
if(APRAPIPES_ENABLE_AI_INFERENCE)
    target_link_libraries(aprapipes TensorRT::TensorRT)
endif()
```

---

## Common Pitfalls

### 1. ❌ Forgetting to Add Test Files

**Problem**: Module builds but tests don't run

**Solution**: Add your test file to `COMPONENT_<NAME>_UT_FILES`

### 2. ❌ Wrong Component Classification

**Problem**: Build fails with missing dependencies

**Solution**: Ensure your module is in the component that provides its dependencies

**Example**: A module using NPP must be in CUDA_COMPONENT, not IMAGE_PROCESSING

### 3. ❌ Platform-Specific Code Without Guards

**Problem**: Build fails on other platforms

**Solution**: Use platform guards

```cpp
#ifdef ENABLE_ARM64
    // Jetson-specific code
#endif
```

### 4. ❌ Missing vcpkg Feature Mapping

**Problem**: Dependencies not installed during build

**Solution**: Update both `vcpkg.json` AND the CMake vcpkg feature mapping (line ~60)

### 5. ❌ CUDA Files Without .cu Extension

**Problem**: CUDA code treated as C++, compilation fails

**Solution**: Use `.cu` extension for CUDA files

### 6. ❌ Not Testing Component Isolation

**Problem**: Module works in full build, fails in minimal build

**Solution**: Test your component in isolation

```bash
# Test with only your component's dependencies
build_windows_cuda.bat --components "CORE;YOUR_COMPONENT"
```

---

## Verification Checklist

After adding your module, verify:

- [ ] **Builds successfully** in component-only mode
- [ ] **Tests pass** in component-only mode
- [ ] **Builds successfully** in full mode
- [ ] **Tests pass** in full mode
- [ ] **No warnings** during compilation
- [ ] **Dependencies** correctly specified in vcpkg.json
- [ ] **Platform guards** for platform-specific code
- [ ] **Documentation** added to module header
- [ ] **Test coverage** for main functionality
- [ ] **Memory leaks** checked (use Boost.Test leak detection)

---

## Quick Reference: File Locations

| File | Purpose | Lines to Edit |
|------|---------|---------------|
| `base/CMakeLists.txt` | Build configuration | 250-850 (sources), 900-1100 (deps), 1250-1500 (tests) |
| `base/vcpkg.json` | Dependencies | Add to appropriate feature |
| `base/include/YourModule.h` | Module interface | Create new |
| `base/src/YourModule.cpp` | Module implementation | Create new |
| `base/test/yourmodule_tests.cpp` | Unit tests | Create new |

---

## Getting Help

### Documentation
- **Component Guide**: [COMPONENTS_GUIDE.md](COMPONENTS_GUIDE.md)
- **Dependency Diagram**: [COMPONENT_DEPENDENCY_DIAGRAM.md](COMPONENT_DEPENDENCY_DIAGRAM.md)
- **Migration Guide**: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Refactoring Log**: [COMPONENT_REFACTORING_LOG.md](COMPONENT_REFACTORING_LOG.md)

### Examples
Look at existing modules similar to yours:
- **Simple module**: FileReaderModule (`src/FileReaderModule.cpp`)
- **OpenCV module**: ImageResizeCV (`src/ImageResizeCV.cpp`)
- **CUDA module**: ResizeNPPI (`src/ResizeNPPI.cu`)
- **Platform-specific**: NvArgusCamera (`src/NvArgusCamera.cpp` - ARM64)

### Report Issues
- GitHub Issues: https://github.com/Apra-Labs/ApraPipes/issues

---

## Summary

**To add a new module:**

1. Choose the right component based on functionality and dependencies
2. Create `.h` and `.cpp`/`.cu` files
3. Add files to `COMPONENT_<NAME>_FILES` in CMakeLists.txt
4. Add dependencies to vcpkg.json (if needed)
5. Create unit tests
6. Add tests to `COMPONENT_<NAME>_UT_FILES`
7. Build and test in component-only mode
8. Verify in full build mode

**Key principle**: Keep components loosely coupled - modules should only depend on components listed in the dependency diagram.
