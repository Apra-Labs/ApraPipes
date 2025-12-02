# ApraPipes Samples

Welcome to the ApraPipes samples! These examples demonstrate how to use the ApraPipes framework for building video processing pipelines.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Available Samples](#available-samples)
- [Building Samples](#building-samples)
- [Running Samples](#running-samples)
- [Learning Path](#learning-path)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

ApraPipes is a high-performance, modular video processing framework. These samples demonstrate:
- Pipeline construction and module connections
- Video capture and playback
- Real-time video processing
- Frame transformations and analysis
- Hardware acceleration (CUDA, NVENC, NVDEC)

Each sample is a standalone executable that showcases specific ApraPipes capabilities.

---

## Quick Start

### Prerequisites

- Windows 10/11 with Visual Studio 2019 or later
- CUDA 11.8+ (for GPU-accelerated samples)
- Webcam (for webcam-based samples)
- MP4 video files (for video processing samples)

### Build All Samples

```powershell
cd D:\dws\ApraPipes\samples
.\build_samples.ps1
```

**Output**: All samples built to `samples\_build\RelWithDebInfo\`

### Run Your First Sample

```powershell
cd samples\_build\RelWithDebInfo
.\hello_pipeline.exe
```

---

## Available Samples

### 🔰 Basic Samples

#### 1. hello_pipeline
**Difficulty**: ⭐ Beginner
**Purpose**: Learn pipeline fundamentals
**Requirements**: None (no external dependencies)

Demonstrates:
- Creating and connecting modules
- Pipeline initialization and lifecycle
- Frame processing basics
- Clean termination

```powershell
.\hello_pipeline.exe
```

📚 [Read hello_pipeline documentation](basic/hello_pipeline/README.md)

---

### 🎥 Video Processing Samples

#### 2. face_detection_cpu
**Difficulty**: ⭐⭐ Intermediate
**Purpose**: Real-time face detection using DNN
**Requirements**: Webcam, Caffe model files

Demonstrates:
- Webcam video capture
- CPU-based face detection (Caffe DNN)
- Bounding box overlay
- Real-time visualization

```powershell
.\face_detection_cpu.exe
```

📚 [Read face_detection_cpu documentation](video/face_detection_cpu/README.md)

---

#### 3. file_reader (MP4 Playback)
**Difficulty**: ⭐⭐ Intermediate
**Purpose**: Play MP4 videos with seek functionality
**Requirements**: MP4 video file

Demonstrates:
- MP4 file reading and demuxing
- H264 hardware decoding
- Video playback with seeking
- Frame rate control

```powershell
.\file_reader.exe <path_to_video.mp4>
```

**Example**:
```powershell
.\file_reader.exe "C:\Videos\sample.mp4"
```

📚 [Read file_reader documentation](video/file_reader/README.md)

---

#### 4. thumbnail_generator
**Difficulty**: ⭐⭐ Intermediate
**Purpose**: Extract thumbnail from video
**Requirements**: MP4 video file

Demonstrates:
- MP4 video reading
- H264 decoding
- Frame extraction (ValveModule)
- JPEG encoding with NVJPEG
- File writing

```powershell
.\thumbnail_generator.exe <input_video.mp4> <output_thumbnail.jpg>
```

**Example**:
```powershell
.\thumbnail_generator.exe "input.mp4" "thumbnail.jpg"
```

📚 [Read thumbnail_generator documentation](video/thumbnail_generator/README.md)

---

#### 5. timelapse (Motion-Based Summarization)
**Difficulty**: ⭐⭐⭐ Advanced
**Purpose**: Create motion-based video summaries
**Requirements**: MP4 video file

Demonstrates:
- Motion detection
- Frame filtering based on movement
- Video summarization
- H264 encoding and MP4 writing

```powershell
.\timelapse.exe <input_video.mp4> <output_summary.mp4>
```

**Example**:
```powershell
.\timelapse.exe "long_video.mp4" "summary.mp4"
```

📚 [Read timelapse documentation](video/timelapse/README.md)

---

### 🌐 Network Samples

#### 6. relay (Dynamic Source Switching)
**Difficulty**: ⭐⭐⭐ Advanced
**Purpose**: Switch between live and recorded sources
**Requirements**: RTSP camera stream, MP4 video file

Demonstrates:
- RTSP client for live camera feeds
- MP4 file playback
- Dynamic source switching (relay pattern)
- Shared decoder for multiple sources

```powershell
.\relay.exe
```

📚 [Read relay documentation](network/relay/README.md)

---

## Building Samples

### Automatic Build (Recommended)

The easiest way to build all samples:

```powershell
cd D:\dws\ApraPipes\samples
.\build_samples.ps1
```

This script:
1. ✅ Verifies prerequisites (aprapipes library, vcpkg)
2. ✅ Configures CMake with correct dependencies
3. ✅ Builds all samples in RelWithDebInfo mode
4. ✅ Copies all required DLLs (85 total: Boost, OpenCV, FFmpeg, etc.)

**Build Output**: `samples\_build\RelWithDebInfo\`

### Manual Build

If you prefer manual control:

```powershell
# Configure
cd samples
cmake -B _build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DVCPKG_INSTALLED_DIR="..\\_build\\vcpkg_installed\\x64-windows"

# Build
cmake --build _build --config RelWithDebInfo

# Output in: _build\RelWithDebInfo\
```

### Build Requirements

The samples build requires:
- ✅ ApraPipes library (`../_build/RelWithDebInfo/aprapipes.lib`)
- ✅ Boost 1.84+ (system, thread, filesystem, log, serialization, chrono)
- ✅ OpenCV 4.8+ with CUDA support
- ✅ CUDA 11.8+
- ✅ FFmpeg libraries (avcodec, avformat, avutil, swscale, swresample)
- ✅ NVIDIA Video Codec SDK (NVENC, NVDEC, CUVID)

All dependencies are automatically managed via vcpkg.

---

## Running Samples

### Sample Executables Location

After building, all samples are in:
```
D:\dws\ApraPipes\samples\_build\RelWithDebInfo\
```

### Running from Command Line

```powershell
# Navigate to build output
cd D:\dws\ApraPipes\samples\_build\RelWithDebInfo

# Run any sample
.\<sample_name>.exe [arguments]
```

### Running from File Explorer

Simply double-click the `.exe` files in:
```
D:\dws\ApraPipes\samples\_build\RelWithDebInfo\
```

### Sample Arguments

| Sample | Arguments | Example |
|--------|-----------|---------|
| hello_pipeline | None | `.\hello_pipeline.exe` |
| face_detection_cpu | None (uses webcam 0) | `.\face_detection_cpu.exe` |
| file_reader | `<video_path>` | `.\file_reader.exe video.mp4` |
| thumbnail_generator | `<video_path> <output_jpg>` | `.\thumbnail_generator.exe in.mp4 thumb.jpg` |
| timelapse | `<input_video> <output_video>` | `.\timelapse.exe in.mp4 out.mp4` |
| relay | None (uses default paths) | `.\relay.exe` |

---

## Learning Path

We recommend learning ApraPipes samples in this order:

### Level 1: Fundamentals 🌱

**Start here if you're new to ApraPipes!**

1. **hello_pipeline** (5 minutes)
   - Understand basic pipeline concepts
   - Learn module connections
   - See frame flow visualization
   - **No external dependencies required**

### Level 2: Video Basics 🎬

2. **file_reader** (10 minutes)
   - Learn video file reading
   - Understand H264 decoding
   - See video playback implementation
   - **Requires**: MP4 video file

3. **thumbnail_generator** (10 minutes)
   - Learn frame extraction
   - Understand ValveModule for filtering
   - See JPEG encoding
   - **Requires**: MP4 video file

### Level 3: Computer Vision 👁️

4. **face_detection_cpu** (15 minutes)
   - Learn webcam capture
   - Understand DNN integration
   - See real-time processing
   - **Requires**: Webcam + model files

### Level 4: Advanced Patterns 🚀

5. **relay** (20 minutes)
   - Learn source switching
   - Understand relay pattern
   - See RTSP streaming
   - **Requires**: RTSP camera + MP4 file

6. **timelapse** (20 minutes)
   - Learn motion detection
   - Understand frame filtering logic
   - See video encoding
   - **Requires**: MP4 video file

---

## Troubleshooting

### Common Issues

#### ❌ "aprapipes.lib not found"

**Solution**: Build the main ApraPipes library first:
```powershell
cd D:\dws\ApraPipes\base
cmake -B ../_build -S . --preset windows-cuda
cmake --build ../_build --config RelWithDebInfo
```

#### ❌ "Missing DLL" errors

**Solution**: DLLs should be copied automatically. If not:
```powershell
cd samples
.\build_samples.ps1  # Rebuild to copy DLLs
```

#### ❌ "Failed to open camera" (face_detection_cpu)

**Causes**:
- Webcam not connected
- Webcam in use by another application
- No camera permissions

**Solution**:
- Connect webcam
- Close other applications using camera (Zoom, Teams, etc.)
- Grant camera permissions in Windows Settings

#### ❌ "Model file not found" (face_detection_cpu)

**Solution**: Download Caffe models and place in `./data/assets/`:
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000_fp16.caffemodel`

#### ❌ "Cannot open file" (video samples)

**Solution**: Provide full path to video file:
```powershell
# Use full path
.\file_reader.exe "C:\Users\YourName\Videos\test.mp4"

# Or use relative path from samples directory
.\file_reader.exe "..\..\data\test_video.mp4"
```

#### ❌ Sample crashes or hangs

**Debug steps**:
1. Check if running from correct directory (where DLLs are)
2. Verify all input files exist and are readable
3. Check system resources (GPU memory, disk space)
4. Review console output for error messages

### Getting Help

- 📖 **Sample-specific README**: Check each sample's README.md
- 🧪 **Testing Guide**: See [TESTING.md](TESTING.md) for detailed test results
- 🐛 **Report Issues**: [GitHub Issues](https://github.com/Apra-Labs/ApraPipes/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Apra-Labs/ApraPipes/discussions)

---

## Sample Architecture

### Pipeline Structure

All samples follow this pattern:

```
┌─────────┐    ┌───────────┐    ┌──────┐
│ Source  │ -> │ Transform │ -> │ Sink │
│ Module  │    │  Module   │    │Module│
└─────────┘    └───────────┘    └──────┘
```

**Source Modules**: Generate or capture frames
- WebCamSource
- Mp4ReaderSource
- RTSPClientSrc
- ExternalSourceModule

**Transform Modules**: Process frames
- FaceDetectorXform
- H264Decoder
- ColorConversion
- OverlayModule
- ValveModule

**Sink Modules**: Output or display frames
- ImageViewerModule
- FileWriterModule
- JPEGEncoderNVJPEG
- ExternalSinkModule

### Module Connection

Modules are connected using `setNext()`:

```cpp
source->setNext(transform);
transform->setNext(sink);
```

### Pipeline Lifecycle

Every pipeline follows this lifecycle:

```cpp
// 1. Create modules
auto source = boost::shared_ptr<Module>(...);
auto sink = boost::shared_ptr<Module>(...);

// 2. Connect modules
source->setNext(sink);

// 3. Add to pipeline
PipeLine pipeline("my_pipeline");
pipeline.appendModule(source);

// 4. Initialize
pipeline.init();

// 5. Run
pipeline.run_all_threaded();  // Or run_all_threaded_withpause()

// 6. Process (happens automatically)

// 7. Stop
pipeline.stop();
pipeline.term();
pipeline.wait_for_all();
```

---

## Performance Tips

### GPU Acceleration

Most samples use CUDA for hardware acceleration:
- ✅ H264Decoder uses NVDEC (GPU video decoding)
- ✅ JPEGEncoderNVJPEG uses NVJPEG (GPU JPEG encoding)
- ✅ ColorConversion uses CUDA kernels
- ✅ Face detection uses DNN inference

**Tip**: Ensure NVIDIA GPU drivers are up to date for best performance.

### Memory Management

ApraPipes uses boost::shared_ptr for automatic memory management:
```cpp
auto module = boost::shared_ptr<Module>(new Module(props));
// No manual delete needed!
```

### Threading

- Most samples use `run_all_threaded()` for automatic threading
- Each module runs in its own thread
- Frame queues handle inter-module communication
- No manual thread management needed

---

## File Structure

```
samples/
├── README.md                    # This file
├── TESTING.md                   # Test results and procedures
├── build_samples.ps1            # Automated build script
├── CMakeLists.txt              # Build configuration
├── copy_dlls.cmake             # DLL copying script
├── test_runner.cpp             # Unit test entry point
│
├── basic/
│   └── hello_pipeline/         # Beginner: Pipeline fundamentals
│       ├── main.cpp
│       ├── README.md
│       └── test_hello_pipeline.cpp
│
├── video/
│   ├── face_detection_cpu/     # Intermediate: Face detection
│   │   ├── main.cpp
│   │   ├── README.md
│   │   └── test_face_detection_cpu.cpp
│   │
│   ├── file_reader/            # Intermediate: MP4 playback
│   │   ├── main.cpp
│   │   ├── README.md
│   │   └── test_file_reader.cpp
│   │
│   ├── thumbnail_generator/    # Intermediate: Thumbnail extraction
│   │   ├── main.cpp
│   │   ├── README.md
│   │   └── test_thumbnail_generator.cpp
│   │
│   └── timelapse/              # Advanced: Motion-based summary
│       ├── main.cpp
│       ├── README.md
│       └── test_timelapse.cpp
│
└── network/
    └── relay/                  # Advanced: Source switching
        ├── main.cpp
        ├── README.md
        └── test_relay.cpp
```

---

## Testing

### Running Tests

Unit tests are built alongside samples:

```powershell
cd samples\_build\RelWithDebInfo
.\sample_tests.exe
```

### Test Coverage

Tests verify:
- ✅ Module creation without errors
- ✅ Pipeline construction
- ✅ Initialization success
- ✅ Basic property validation

**Note**: Full integration testing requires external resources (webcam, video files).

See [TESTING.md](TESTING.md) for detailed test results.

---

## Contributing

### Adding New Samples

1. **Create sample directory**:
   ```
   samples/category/sample_name/
   ├── main.cpp
   ├── README.md
   └── test_sample_name.cpp
   ```

2. **Add to CMakeLists.txt**:
   ```cmake
   if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/category/sample_name/main.cpp)
       add_apra_sample(sample_name category/sample_name/main.cpp)
   endif()
   ```

3. **Follow sample structure**:
   - Use class to encapsulate pipeline
   - Provide `setupPipeline()`, `startPipeline()`, `stopPipeline()` methods
   - Add comprehensive comments
   - Handle errors gracefully
   - Return proper exit codes

4. **Create README.md** with:
   - Purpose and what it demonstrates
   - Requirements
   - Usage instructions
   - Expected output
   - Troubleshooting

### Sample Guidelines

- ✅ Use descriptive variable names
- ✅ Add comprehensive comments
- ✅ Print clear status messages
- ✅ Handle errors gracefully
- ✅ Clean up resources properly
- ✅ Document all requirements
- ✅ Provide usage examples
- ❌ Don't use hardcoded paths
- ❌ Don't require specific hardware without alternatives
- ❌ Don't leave zombie threads or memory leaks

---

## License

These samples are part of the ApraPipes project and follow the same license as the main library.

---

## Additional Resources

- 📚 [ApraPipes Documentation](https://github.com/Apra-Labs/ApraPipes)
- 🎓 [ApraPipes Wiki](https://github.com/Apra-Labs/ApraPipes/wiki)
- 💬 [Community Discussions](https://github.com/Apra-Labs/ApraPipes/discussions)
- 🐛 [Report Issues](https://github.com/Apra-Labs/ApraPipes/issues)
- 📧 [Contact Support](mailto:support@apra.ai)

---

**Happy coding with ApraPipes! 🚀**
