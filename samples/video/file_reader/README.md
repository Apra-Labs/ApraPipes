# File Reader Sample - MP4 Video Playback with Seeking

## Overview

This sample demonstrates how to play MP4 video files using ApraPipes with support for seeking to specific timestamps. It showcases:
- MP4 file reading and demuxing
- H264 hardware video decoding (NVDEC)
- Color space conversion (YUV420 to RGB)
- Real-time video display
- Seek functionality for timestamp navigation
- Pipeline queue flushing for clean seeks

## What You'll Learn

- How to use `Mp4ReaderSource` for video file reading
- How to configure loop playback and seeking
- How to use `H264Decoder` for hardware-accelerated decoding
- How to convert color spaces with `ColorConversion`
- How to display video frames with `ImageViewerModule`
- How to implement seek functionality with queue flushing

## Pipeline Structure

```
┌──────────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Mp4ReaderSource  │ --> │ H264Decoder │ --> │ ColorConversion │ --> │ ImageViewerModule│
│  (Read MP4)      │     │ (GPU Decode)│     │ (YUV420->RGB)   │     │   (Display)      │
└──────────────────┘     └─────────────┘     └─────────────────┘     └──────────────────┘
```

### Module Details

1. **Mp4ReaderSource**
   - Reads and demuxes MP4 container
   - Extracts H264 encoded frames
   - Supports loop playback
   - Enables seeking to timestamps
   - Configurable playback speed (FPS)

2. **H264Decoder**
   - Hardware-accelerated H264 decoding using NVIDIA NVDEC
   - Outputs raw YUV420 frames
   - Efficient GPU-based processing

3. **ColorConversion**
   - Converts YUV420 planar to RGB
   - Required for display on most systems
   - Uses CUDA kernels for fast conversion

4. **ImageViewerModule**
   - Displays video frames in OpenCV window
   - Updates at specified frame rate
   - Handles window events

## Requirements

### Software
- Windows 10/11
- Visual Studio 2019 or later
- CUDA 11.8+
- NVIDIA GPU with NVDEC support
- ApraPipes library built

### Hardware
- NVIDIA GPU (for hardware H264 decoding)

### Input Files
- **MP4 video file** with H264 encoding

## Building

The sample is built automatically with other samples:

```powershell
cd D:\dws\ApraPipes\samples
.\build_samples.ps1
```

**Output**: `samples\_build\RelWithDebInfo\file_reader.exe`

## Usage

### Basic Usage

```powershell
cd D:\dws\ApraPipes\samples\_build\RelWithDebInfo
.\file_reader.exe <path_to_video.mp4>
```

### Examples

```powershell
# Play video from current directory
.\file_reader.exe test_video.mp4

# Play video with full path
.\file_reader.exe "C:\Videos\sample.mp4"

# Play video from relative path
.\file_reader.exe "..\..\..\data\test_video.mp4"
```

### Runtime Controls

While the video is playing:
- **ESC** or **Q**: Quit playback
- Video displays in OpenCV window titled "MP4 Player"

## Configuration

You can modify the pipeline configuration in `main.cpp`:

```cpp
// Playback configuration
const int targetFPS = 30;              // Playback speed
const bool enableLoop = true;          // Loop video when it ends
const bool rewindOnLoop = true;        // Restart from beginning
const int runDurationSeconds = 60;     // Auto-stop after N seconds
```

### Mp4ReaderSource Configuration

```cpp
auto mp4ReaderProps = Mp4ReaderSourceProps(
    videoPath,       // Input MP4 file path
    parseFS,         // Parse filesystem for frame numbers
    0,               // Start frame index
    enableLoop,      // Enable looping
    rewindOnLoop,    // Rewind to start on loop
    false            // Direction (false = forward)
);
mp4ReaderProps.fps = targetFPS;  // Playback frame rate
```

### Seeking Configuration

The sample includes a `seekToTimestamp()` method for jumping to specific times:

```cpp
bool seekToTimestamp(uint64_t timestampMs) {
    // 1. Flush queues to discard old frames
    mMp4Reader->flushQueues();
    mDecoder->flushQueues();
    mColorConv->flushQueues();

    // 2. Seek to timestamp
    return mMp4Reader->randomSeek(timestampMs, false);
}
```

## Expected Output

### Console Output

```
============================================================
     ApraPipes Sample: MP4 Video Player
============================================================

Setting up MP4 playback pipeline...
  Video file: C:\Videos\sample.mp4
  Target FPS: 30
  Loop playback: Enabled
  Auto-stop: 60 seconds

Pipeline setup completed successfully!

Pipeline structure:
  [Mp4Reader] → [H264Decoder] → [ColorConversion] → [ImageViewer]

Initializing and starting pipeline...
Pipeline started successfully!

Playing video...
Press ESC or Q in the viewer window to stop, or wait for automatic shutdown.
```

### Video Window

A window titled "MP4 Player" will open displaying the video:
- Video plays at the configured frame rate
- Window updates in real-time
- Press ESC or Q to close

### Termination

```
Stopping pipeline...
Pipeline stopped successfully!

╔══════════════════════════════════════════════════════════════╗
║     MP4 Player Sample Completed Successfully!                ║
╚══════════════════════════════════════════════════════════════╝
```

## Features Demonstrated

### 1. MP4 File Reading

```cpp
// Create MP4 reader with loop support
auto mp4ReaderProps = Mp4ReaderSourceProps(
    videoPath, false, 0, true, true, false
);
mp4ReaderProps.fps = 30;

auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(
    new Mp4ReaderSource(mp4ReaderProps)
);
```

### 2. H264 Metadata Configuration

```cpp
// Configure H264 metadata for decoder
auto h264ImageMetadata = framemetadata_sp(
    new H264Metadata(1920, 1080)  // Resolution
);
mp4Reader->addOutputPin(h264ImageMetadata);

// Add MP4 metadata pin
auto mp4Metadata = framemetadata_sp(
    new Mp4VideoMetadata("v_1")
);
mp4Reader->addOutputPin(mp4Metadata);
```

### 3. Color Space Conversion

```cpp
// Convert YUV420 from decoder to RGB for display
auto colorConv = boost::shared_ptr<ColorConversion>(
    new ColorConversion(
        ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)
    )
);
```

### 4. Video Seeking

```cpp
// Seek to specific timestamp
uint64_t targetTime = 30000;  // 30 seconds in milliseconds

// Flush queues first
mp4Reader->flushQueues();
decoder->flushQueues();
colorConv->flushQueues();

// Perform seek
mp4Reader->randomSeek(targetTime, false);
```

### 5. Frame Rate Control

```cpp
// Control playback speed
mp4ReaderProps.fps = 30;  // Play at 30 FPS
// OR
mp4ReaderProps.fps = 60;  // Play at 60 FPS (fast)
// OR
mp4ReaderProps.fps = 15;  // Play at 15 FPS (slow motion)
```

## Troubleshooting

### ❌ "Cannot open file: [path]"

**Cause**: File not found or no read permissions

**Solutions**:
1. Verify file exists:
   ```powershell
   Test-Path "path\to\video.mp4"
   ```
2. Use absolute path instead of relative path
3. Check file permissions
4. Ensure file is not locked by another application

### ❌ "Failed to initialize Mp4ReaderSource"

**Causes**:
- File is not a valid MP4 file
- File is corrupted
- Unsupported codec (not H264)

**Solutions**:
1. Verify MP4 file is valid:
   ```powershell
   ffprobe video.mp4
   ```
2. Re-encode video to H264 if needed:
   ```powershell
   ffmpeg -i input.mp4 -c:v libx264 -preset fast output.mp4
   ```

### ❌ "H264Decoder initialization failed"

**Causes**:
- No NVIDIA GPU available
- GPU drivers outdated
- NVDEC not supported on GPU

**Solutions**:
1. Check NVIDIA GPU is present:
   ```powershell
   nvidia-smi
   ```
2. Update GPU drivers to latest version
3. Verify GPU supports NVDEC (most GTX 900+ and all RTX cards)

### ❌ Window doesn't open / No video display

**Causes**:
- OpenCV display issue
- Graphics driver problem
- Running in headless environment

**Solutions**:
1. Ensure running on machine with display
2. Update graphics drivers
3. Try running with administrator privileges

### ❌ Video plays too fast/slow

**Solution**: Adjust FPS in configuration:
```cpp
mp4ReaderProps.fps = 30;  // Set to video's native FPS
```

To find video's native FPS:
```powershell
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 video.mp4
```

### ❌ "Pipeline failed to start"

**Solutions**:
1. Check all DLLs are present in executable directory
2. Verify CUDA runtime is installed
3. Check video file is H264 encoded (not H265/HEVC)
4. Review console output for specific error messages

## Advanced Usage

### Custom Frame Processing

You can replace `ImageViewerModule` with your own processing:

```cpp
// Instead of ImageViewerModule, use ExternalSinkModule
auto sink = boost::shared_ptr<ExternalSinkModule>(
    new ExternalSinkModule()
);
colorConv->setNext(sink);

// Then in processing loop:
auto frames = sink->pop();
for (auto& frame : frames) {
    // Your custom processing here
    processFrame(frame.second);
}
```

### Seeking to Specific Frame

```cpp
// Calculate timestamp from frame number
uint64_t frameNumber = 100;
uint64_t fps = 30;
uint64_t timestampMs = (frameNumber * 1000) / fps;

// Seek
mp4Reader->randomSeek(timestampMs, false);
```

### Backward Playback

```cpp
// Enable reverse playback
auto mp4ReaderProps = Mp4ReaderSourceProps(
    videoPath, false, 0, true, true,
    true  // direction = true for backward
);
```

## Performance Notes

### Hardware Acceleration

- **NVDEC**: GPU-accelerated H264 decoding
  - ~10-50x faster than CPU decoding
  - Minimal CPU usage
  - Supports multiple concurrent streams

- **CUDA Color Conversion**: Fast YUV to RGB conversion on GPU
  - Eliminates PCIe transfer overhead
  - Efficient memory usage

### Memory Usage

- Pipeline uses frame queues for buffering
- Typical memory usage: 100-500 MB depending on resolution
- GPU memory usage: ~200-500 MB for decoder buffers

### Optimization Tips

1. **Match FPS to display refresh rate** to avoid frame drops
2. **Use GPU throughout pipeline** to minimize transfers
3. **Adjust queue sizes** if experiencing lag or stuttering
4. **Consider frame skipping** for real-time requirements

## Related Samples

- **thumbnail_generator**: Extract single frame from video
- **timelapse**: Motion-based video summarization
- **relay**: Switch between live and recorded sources

## Code Structure

```cpp
class FileReaderSample {
public:
    FileReaderSample();

    // Setup pipeline with video file
    bool setupPipeline(const std::string& videoPath);

    // Start video playback
    bool startPipeline();

    // Stop playback
    bool stopPipeline();

    // Seek to timestamp
    bool seekToTimestamp(uint64_t timestampMs);

private:
    PipeLine pipeline;
    boost::shared_ptr<Mp4ReaderSource> mMp4Reader;
    boost::shared_ptr<H264Decoder> mDecoder;
    boost::shared_ptr<ColorConversion> mColorConv;
    boost::shared_ptr<ImageViewerModule> mImageViewer;
};
```

## Next Steps

- ✅ Modify FPS for different playback speeds
- ✅ Implement seeking logic based on user input
- ✅ Add frame export functionality
- ✅ Process frames instead of just displaying
- ✅ Integrate with relay sample for live/recorded switching
- ✅ Add audio playback (requires audio pipeline)

## Learn More

- 📚 [Mp4ReaderSource Documentation](../../base/include/Mp4ReaderSource.h)
- 📚 [H264Decoder Documentation](../../base/include/H264Decoder.h)
- 📚 [ColorConversion Documentation](../../base/include/ColorConversionXForm.h)
- 📚 [Main Samples README](../../README.md)
- 🧪 [Testing Guide](../../TESTING.md)
