# ApraPipes Samples

Welcome to the ApraPipes samples! These samples demonstrate how to use the ApraPipes framework to build video/image processing pipelines.

---

## Overview

The samples are built as a **standalone project** that links against the already-built ApraPipes library. This approach:
- ✅ Keeps samples isolated from the main library build
- ✅ Demonstrates real-world usage patterns
- ✅ Allows independent building and testing
- ✅ Provides educational examples for new users

---

## Prerequisites

Before building samples, you must:

1. **Build the main ApraPipes library first**:
   ```powershell
   cd D:\dws\ApraPipes
   .\build_windows_cuda_vs19.ps1
   ```

2. **Verify the library exists**:
   - For Release/RelWithDebInfo: `_build\RelWithDebInfo\aprapipes.lib`
   - For Debug: `_build\Debug\aprapipesd.lib`

---

## Building Samples

### Windows (Visual Studio 2019)

```powershell
cd samples
.\build_samples.ps1
```

**Options**:
- Build Debug: `.\build_samples.ps1 -BuildType Debug`
- Build RelWithDebInfo (default): `.\build_samples.ps1 -BuildType RelWithDebInfo`
- Clean build: `.\build_samples.ps1 -Clean`

### Build Output

Executables are placed in:
```
samples/_build/<Configuration>/
```

For example:
- `samples/_build/RelWithDebInfo/hello_pipeline.exe`
- `samples/_build/Debug/hello_pipeline.exe`

---

## Running Samples

### Option 1: Direct Execution

```powershell
cd samples\_build\RelWithDebInfo
.\hello_pipeline.exe
```

### Option 2: Full Path

```powershell
.\samples\_build\RelWithDebInfo\hello_pipeline.exe
```

### Option 3: Open in Visual Studio

1. Open `samples/_build/ApraPipesSamples.sln`
2. Select configuration (Debug or RelWithDebInfo)
3. Right-click on a sample project → "Set as Startup Project"
4. Press F5 to run

---

## Available Samples

### 1. hello_pipeline (Basic)

**Location**: `basic/hello_pipeline/main.cpp`
**Difficulty**: ⭐ Beginner
**Category**: Basic / Introduction

**What it demonstrates**:
- Creating a basic pipeline with SOURCE → TRANSFORM → SINK modules
- Using `ExternalSourceModule` to inject frames
- Using `Split` module to duplicate frames
- Using `ExternalSinkModule` to receive processed frames
- Pipeline initialization, processing, and termination
- Cross-platform code (Windows/Linux)

**Key concepts**:
- Module creation and configuration
- Pipeline connections (`setNext()`)
- Frame creation and injection
- Frame metadata handling
- Round-robin frame distribution

**How to run**:
```powershell
.\samples\_build\RelWithDebInfo\hello_pipeline.exe
```

**Expected output**:
```
╔══════════════════════════════════════════════════════════════╗
║          ApraPipes Sample: Hello Pipeline                    ║
╚══════════════════════════════════════════════════════════════╝

Creating a simple pipeline:
  [ExternalSource] → [Split(x2)] → [ExternalSink]

...Frame processing output...

Pipeline completed successfully!
```

---

### 2. face_detection_cpu (Video Processing)

**Location**: `video/face_detection_cpu/main.cpp`
**Difficulty**: ⭐⭐ Intermediate
**Category**: Video / Computer Vision

**What it demonstrates**:
- Reading MP4 video files
- H264 video decoding
- Face detection using OpenCV Haar Cascades
- Drawing bounding boxes and overlays on detected faces
- Real-time video processing and display

**Key concepts**:
- MP4 video file reading with `Mp4ReaderSource`
- H264 decoding pipeline
- Computer vision integration with OpenCV
- Face detection algorithms
- Overlay drawing for visualization
- Frame-by-frame processing

**Pipeline**:
```
[Mp4Reader] → [H264Decoder] → [FaceDetector] → [Overlay] → [ImageViewer]
```

**How to run**:
```powershell
.\samples\_build\RelWithDebInfo\face_detection_cpu.exe <video_path>
```

**Example**:
```powershell
.\samples\_build\RelWithDebInfo\face_detection_cpu.exe video_with_faces.mp4
```

---

### 3. relay (Network Streaming)

**Location**: `network/relay/main.cpp`
**Difficulty**: ⭐⭐⭐ Advanced
**Category**: Network / Streaming

**What it demonstrates**:
- Dynamic source switching between RTSP and MP4
- RTSP network camera streaming
- MP4 file playback
- Relay pattern for source management
- Interactive keyboard control

**Key concepts**:
- **Relay pattern**: Switching between multiple input sources
- RTSP client for live camera streams
- MP4 file reading for recorded content
- Dynamic source enable/disable at runtime
- H264 video decoding from multiple sources

**Pipeline**:
```
[RTSPClientSrc]  ─┐
                   ├─> [H264Decoder] → [ColorConversion] → [ImageViewer]
[Mp4ReaderSource]─┘
```

**How to run**:
```powershell
.\samples\_build\RelWithDebInfo\relay.exe <rtsp_url> <mp4_file_path>
```

**Example**:
```powershell
.\samples\_build\RelWithDebInfo\relay.exe rtsp://camera:554/stream backup.mp4
```

**Interactive controls**:
- Press `r` to switch to RTSP source (live camera)
- Press `m` to switch to MP4 source (recorded video)
- Press `s` to stop and exit

**Use cases**:
- Security systems with live/playback switching
- Failover from live to backup footage
- Side-by-side comparison of live vs recorded
- Testing with recorded data

---

### 4. thumbnail_generator (Video Processing)

**Location**: `video/thumbnail_generator/main.cpp`
**Difficulty**: ⭐⭐ Intermediate
**Category**: Video / Image Processing

**What it demonstrates**:
- Extracting a single frame from MP4 video
- **ValveModule** for precise frame control
- GPU-accelerated JPEG encoding with NVJPEG
- CUDA memory operations (host to device transfer)
- File writing operations

**Key concepts**:
- **ValveModule**: Acts as a programmable gate to control frame flow
  - Initially closed (allows 0 frames)
  - Open to allow exactly N frames
  - Automatically closes after count reached
- CUDA pipeline optimization
- Single-frame extraction from long videos
- GPU-accelerated image encoding

**Pipeline**:
```
[Mp4Reader] → [H264Decoder] → [ValveModule] → [CudaMemCopy] →
[JPEGEncoder] → [FileWriter]
```

**How to run**:
```powershell
.\samples\_build\RelWithDebInfo\thumbnail_generator.exe <video_path> <output_path>
```

**Example**:
```powershell
.\samples\_build\RelWithDebInfo\thumbnail_generator.exe movie.mp4 thumbnail_????.jpg
```

Output: `thumbnail_0000.jpg` (first frame of video)

**Use cases**:
- Video library poster images
- Preview thumbnails for video galleries
- Video identification and cataloging
- Batch thumbnail generation for media management

---

### 5. timelapse (Video Processing)

**Location**: `video/timelapse/main.cpp`
**Difficulty**: ⭐⭐⭐ Advanced
**Category**: Video / Computer Vision

**What it demonstrates**:
- Motion-based frame extraction
- **MotionVectorExtractor** for intelligent frame filtering
- Multiple color space conversions
- GPU-accelerated H264 encoding
- MP4 video writing
- Video summarization techniques

**Key concepts**:
- **MotionVectorExtractor**: Analyzes motion in H264 compressed video
  - Uses motion vectors already present in H264 stream
  - Filters frames based on motion threshold
  - Outputs only frames with significant motion
- Color space pipeline: BGR → RGB → YUV420
- Hardware-accelerated video encoding
- Video compression by content (not time)

**Pipeline**:
```
[Mp4Reader] → [MotionExtractor] → [BGR→RGB] → [RGB→YUV420] →
[CudaCopy] → [CudaSync] → [H264Encoder] → [Mp4Writer]
```

**How to run**:
```powershell
.\samples\_build\RelWithDebInfo\timelapse.exe <input_video> <output_folder>
```

**Example**:
```powershell
.\samples\_build\RelWithDebInfo\timelapse.exe surveillance_8hrs.mp4 timelapse_out/
```

**Use cases**:
- Compress hours of surveillance footage into minutes
- Create time-lapse videos from long recordings
- Generate video summaries skipping static scenes
- Reduce storage requirements while preserving action
- Extract only interesting moments from long videos

**Typical results**:
- 8 hours of surveillance → 10 minutes of activity
- 60% - 95% reduction in video length (depends on motion threshold)

---

### 6. file_reader (Video Playback)

**Location**: `video/file_reader/main.cpp`
**Difficulty**: ⭐⭐ Intermediate
**Category**: Video / Playback

**What it demonstrates**:
- Basic MP4 video playback
- H264 video decoding
- **Timestamp-based seeking** with queue flushing
- Video display in OpenCV window
- Frame rate control

**Key concepts**:
- MP4 container format reading
- Pipeline queue management
- **Seek functionality**:
  - `flushAllQueues()`: Clears buffered frames
  - `randomSeek(timestamp)`: Jumps to specific time
  - Clean seeks without artifacts
- Color space conversion for display

**Pipeline**:
```
[Mp4Reader] → [H264Decoder] → [ColorConversion] → [ImageViewer]
```

**How to run**:
```powershell
.\samples\_build\RelWithDebInfo\file_reader.exe <video_path>
```

**Example**:
```powershell
.\samples\_build\RelWithDebInfo\file_reader.exe movie.mp4
```

**Behavior**:
1. Plays video for 3 seconds from start
2. Demonstrates seek to specific timestamp
3. Continues playing for 5 more seconds
4. Stops playback

**Use cases**:
- Video player applications
- Video analysis tools with navigation
- Frame-accurate video inspection
- Video debugging and testing
- Educational demonstrations of seek operations

---

## Adding New Samples

To add a new sample, edit `samples/CMakeLists.txt` and add one line:

```cmake
# Add your new sample here
add_apra_sample(your_sample_name path/to/main.cpp)
```

That's it! The build system automatically:
- ✅ Links against aprapipes library (debug or release)
- ✅ Links against correct Boost libraries (debug or release)
- ✅ Links against CUDA runtime
- ✅ Sets correct MSVC runtime library (/MD or /MDd)
- ✅ Copies required DLLs to output directory

### Example: Adding a webcam capture sample

```cmake
# In samples/CMakeLists.txt, add:
add_apra_sample(webcam_capture video/webcam_capture/main.cpp)
```

Then rebuild:
```powershell
cd samples
.\build_samples.ps1
```

---

## Architecture

### Directory Structure

```
ApraPipes/
├── base/                          # Main library (untouched)
│   ├── CMakeLists.txt
│   ├── src/
│   ├── include/
│   └── test/
│
├── samples/                       # Samples (standalone)
│   ├── CMakeLists.txt             # Standalone project config
│   ├── _build/                    # Samples build output
│   ├── build_samples.ps1          # Samples build script
│   ├── copy_dlls.cmake            # DLL copying script
│   ├── basic/
│   │   └── hello_pipeline/
│   │       ├── main.cpp
│   │       └── README.md
│   └── ApraPipesSamples.md        # This file
│
└── _build/                        # Main library build (untouched)
```

### Build Isolation

**Main Library Build**:
- Source: `base/`
- Build: `_build/`
- Script: `build_windows_cuda_vs19.ps1`
- Output: `_build/RelWithDebInfo/aprapipesut.exe` (unit tests)

**Samples Build**:
- Source: `samples/`
- Build: `samples/_build/`
- Script: `samples/build_samples.ps1`
- Output: `samples/_build/RelWithDebInfo/*.exe` (samples)

**Zero overlap. Zero conflicts.**

---

## How It Works

### 1. Library Discovery

The samples CMake configuration uses `find_library()` to locate the aprapipes library:

```cmake
find_library(APRAPIPES_LIB
    NAMES aprapipes aprapipesd
    PATHS
        ${APRAPIPES_BUILD_DIR}/${CMAKE_BUILD_TYPE}
        ${APRAPIPES_BUILD_DIR}/RelWithDebInfo
        ${APRAPIPES_BUILD_DIR}/Debug
    NO_DEFAULT_PATH
)
```

### 2. Multi-Configuration Support

Samples automatically build with the correct configuration:

| Configuration   | Library           | Runtime | Boost Libs        | Boost DLLs          |
|-----------------|-------------------|---------|-------------------|---------------------|
| Debug           | aprapipesd.lib    | /MDd    | *-vc140-mt-gd.lib | *-gd-*.dll          |
| RelWithDebInfo  | aprapipes.lib     | /MD     | *-vc140-mt.lib    | *-mt-*.dll          |
| Release         | aprapipes.lib     | /MD     | *-vc140-mt.lib    | *-mt-*.dll          |

### 3. Automatic Dependency Handling

The `add_apra_sample()` function handles:
- **Boost libraries**: Automatically selects debug (`-gd`) or release variants using CMake generator expressions
- **CUDA runtime**: Uses CMake imported targets (`CUDA::cudart`, `CUDA::cuda_driver`)
- **aprapipes library**: Searches correct build directory based on configuration
- **Runtime library**: Matches aprapipes library (/MD or /MDd)
- **DLL copying**: Post-build script copies correct Boost DLLs

### 4. Helper Function for Boost

```cmake
function(get_boost_libraries OUTPUT_VAR)
    set(BOOST_LIBS "")
    foreach(COMPONENT ${ARGN})
        list(APPEND BOOST_LIBS
            "$<$<CONFIG:Debug>:${BOOST_LIB_DIR_DEBUG}/boost_${COMPONENT}-vc140-mt-gd.lib>"
            "$<$<NOT:$<CONFIG:Debug>>:${BOOST_LIB_DIR_RELEASE}/boost_${COMPONENT}-vc140-mt.lib>"
        )
    endforeach()
    set(${OUTPUT_VAR} ${BOOST_LIBS} PARENT_SCOPE)
endfunction()
```

This ensures samples link against the correct debug or release Boost libraries automatically.

---

## Troubleshooting

### Error: "aprapipes library not found"

**Cause**: Main library hasn't been built yet.

**Solution**:
```powershell
cd D:\dws\ApraPipes
.\build_windows_cuda_vs19.ps1
```

### Error: "Cannot find vcpkg toolchain"

**Cause**: vcpkg hasn't been set up.

**Solution**: Build the main library first (it sets up vcpkg automatically).

### Error: "Missing boost_*.dll"

**Cause**: DLLs weren't copied correctly.

**Solution**: Rebuild samples:
```powershell
cd samples
.\build_samples.ps1 -Clean
```

### Error: "LNK2038: mismatch detected for 'RuntimeLibrary'"

**Cause**: Trying to link Debug sample against Release library (or vice versa).

**Solution**: Build both with same configuration:
```powershell
# Build main library in Debug
.\build_windows_cuda_vs19.ps1 -BuildType Debug

# Build samples in Debug
cd samples
.\build_samples.ps1 -BuildType Debug
```

### Error: "Cannot open file 'boost_system-vc140-mt-gd.lib'"

**Cause**: Boost libraries haven't been installed by vcpkg.

**Solution**: Build the main library first (it installs all dependencies via vcpkg).

---

## Dependencies

Samples depend on the same libraries as the main ApraPipes library:

- **Boost** (system, thread, filesystem, serialization, log, chrono)
- **CUDA Toolkit** (runtime and driver API)
- **ApraPipes library** (aprapipes.lib or aprapipesd.lib)

All dependencies are managed through:
- **vcpkg** for Boost and other third-party libraries
- **CUDA Toolkit** installed separately
- **aprapipes** built from source

---

## Build System Details

### Configuration-Aware DLL Copying

The `copy_dlls.cmake` script automatically detects build configuration:

```cmake
if(CONFIG MATCHES "Debug")
    set(BOOST_DLL_SUFFIX "-vc142-mt-gd-x64-1_84.dll")
    set(VCPKG_BIN_DIR "${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows/debug/bin")
else()
    set(BOOST_DLL_SUFFIX "-vc142-mt-x64-1_84.dll")
    set(VCPKG_BIN_DIR "${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows/bin")
endif()
```

This ensures correct DLLs are copied for each configuration.

### Runtime Library Matching

Samples use the same runtime library as aprapipes:

```cmake
set_property(TARGET ${SAMPLE_NAME} PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
)
```

This prevents runtime library mismatches (LNK2038 errors).

---

## Sample Categories

### By Difficulty
- ⭐ **Beginner**: hello_pipeline
- ⭐⭐ **Intermediate**: face_detection_cpu, thumbnail_generator, file_reader
- ⭐⭐⭐ **Advanced**: relay, timelapse

### By Category
- **Basic/Introduction**: hello_pipeline
- **Video Processing**: face_detection_cpu, thumbnail_generator, timelapse, file_reader
- **Network/Streaming**: relay
- **Computer Vision**: face_detection_cpu, timelapse
- **Image Processing**: thumbnail_generator

---

## Learning Path

We recommend exploring samples in this order:

### 1. Getting Started (Beginner)
**Start here**: `hello_pipeline`
- Learn basic pipeline concepts
- Understand module connections
- See SOURCE → TRANSFORM → SINK pattern
- Master pipeline lifecycle (init, run, stop)

### 2. Video Basics (Intermediate)
**Next**: `file_reader`
- MP4 file reading
- H264 decoding
- Video playback
- Seeking operations

**Then**: `thumbnail_generator`
- Frame extraction
- ValveModule usage
- CUDA operations
- JPEG encoding

### 3. Computer Vision (Intermediate to Advanced)
**Start with**: `face_detection_cpu`
- OpenCV integration
- Real-time detection
- Overlay drawing
- Frame-by-frame processing

**Advance to**: `timelapse`
- Motion analysis
- Intelligent frame filtering
- Video summarization
- H264 encoding

### 4. Network Streaming (Advanced)
**Finally**: `relay`
- RTSP streaming
- Multi-source management
- Dynamic source switching
- Interactive control

---

## Sample Comparison Matrix

| Sample | Video I/O | Processing | GPU | Network | Difficulty |
|--------|-----------|------------|-----|---------|------------|
| hello_pipeline | ❌ | ✅ Basic | ❌ | ❌ | ⭐ |
| face_detection_cpu | ✅ Read | ✅ CV | ❌ | ❌ | ⭐⭐ |
| relay | ✅ Both | ✅ Decode | ❌ | ✅ RTSP | ⭐⭐⭐ |
| thumbnail_generator | ✅ Read | ✅ Encode | ✅ CUDA | ❌ | ⭐⭐ |
| timelapse | ✅ Both | ✅ Motion | ✅ CUDA | ❌ | ⭐⭐⭐ |
| file_reader | ✅ Read | ✅ Decode | ❌ | ❌ | ⭐⭐ |

---

## Future Samples (Planned)

- **video/webcam_capture** - Capture frames from webcam
- **image/resize** - Resize images using CUDA (ResizeNPPI)
- **image/encode** - Encode frames to JPEG/PNG variants
- **transform/overlay** - Advanced overlay text/graphics on frames
- **advanced/multi_pipeline** - Multiple pipelines running concurrently
- **network/rtsp_pusher** - Push video streams to RTSP server
- **video/frame_writer** - Extract and save individual frames

---

## Contributing

When adding new samples:

1. **Create sample directory**:
   ```
   samples/<category>/<sample_name>/
   ├── main.cpp
   └── README.md
   ```

2. **Add to CMakeLists.txt**:
   ```cmake
   add_apra_sample(sample_name category/sample_name/main.cpp)
   ```

3. **Write clear README.md** explaining:
   - What the sample demonstrates
   - Key concepts used
   - Expected output
   - Any special requirements

4. **Test both configurations**:
   ```powershell
   .\build_samples.ps1 -BuildType Debug
   .\build_samples.ps1 -BuildType RelWithDebInfo
   ```

---

## Documentation

For more information:

- **ApraPipes Main Documentation**: `../README.md`
- **Individual Sample READMEs**: `basic/hello_pipeline/README.md`, etc.

---

## Questions?

If you encounter issues:

1. Check this README's Troubleshooting section
2. Verify main library builds correctly
3. Ensure you're using matching configurations (Debug with Debug, Release with Release)
4. Check that vcpkg dependencies are installed

---

**Last Updated**: 2025-10-24
**ApraPipes Version**: Compatible with current main branch
**Samples Version**: 2.0 (Added relay fix, thumbnail_generator, timelapse, file_reader)
