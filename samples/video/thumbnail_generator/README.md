# Thumbnail Generator Sample

**Category**: Video Processing
**Difficulty**: Beginner
**Dependencies**: MP4 video file, CUDA, NVJPEG

## Overview

This sample demonstrates how to extract a single frame from an MP4 video file and save it as a JPEG thumbnail. This is a common operation in video management systems, media libraries, and video streaming platforms where you need preview images for video content.

### What You'll Learn

- How to read MP4 video files
- H264 video decoding
- **Frame filtering with ValveModule** (key concept)
- CUDA-accelerated JPEG encoding
- File writing operations

## Pipeline Structure

```
[Mp4ReaderSource] → [H264Decoder] → [ValveModule] → [CudaMemCopy] →
[JPEGEncoderNVJPEG] → [FileWriterModule]
```

### Module Description

1. **Mp4ReaderSource**: Reads H264 compressed frames from MP4 container
2. **H264Decoder**: Decodes H264 frames to raw YUV420 format
3. **ValveModule**: Acts as a programmable gate - allows exactly N frames to pass
4. **CudaMemCopy**: Transfers frame data from host (CPU) to device (GPU) memory
5. **JPEGEncoderNVJPEG**: NVIDIA GPU-accelerated JPEG encoder
6. **FileWriterModule**: Writes the encoded JPEG file to disk

### Key Concept: ValveModule

The **ValveModule** is the critical component in this pipeline. Think of it as a water valve that you can open to let a specific amount of water through, then automatically close.

```cpp
// Create valve initially closed (0 frames allowed)
auto valve = new ValveModule(ValveModuleProps(0));

// After pipeline starts, open it to allow exactly 1 frame
valve->allowFrames(1);
```

**How it works:**
- Initially configured to allow 0 frames (closed)
- After pipeline starts, we call `allowFrames(1)` to open it
- The first frame passes through
- The valve automatically closes
- All subsequent frames are blocked
- This ensures we capture exactly one thumbnail, no more, no less

**Why use ValveModule instead of just stopping after one frame?**
- Clean pipeline design - no need to manually stop modules
- Thread-safe operation
- Prevents race conditions
- Can be used for batch operations (e.g., every Nth frame)

## Prerequisites

### Software Requirements

- ApraPipes library built with CUDA support
- NVIDIA GPU with CUDA Toolkit
- NVJPEG library (part of CUDA Toolkit)

### Input Requirements

- MP4 video file with H264 codec
- Video can be any resolution
- Video can be any length (only first frame is used)

## Building

The thumbnail_generator sample is built as part of the samples build system:

```powershell
# From repository root
cd samples
.\build_samples.ps1 -BuildType RelWithDebInfo
```

The executable will be created at:
```
samples\_build\RelWithDebInfo\thumbnail_generator.exe
```

## Running the Sample

### Basic Usage

```powershell
.\thumbnail_generator.exe <video_path> <output_path>
```

### Example

```powershell
# Extract thumbnail from video
.\thumbnail_generator.exe input.mp4 thumbnail_????.jpg

# The output will be saved as:
# thumbnail_0000.jpg
```

### Output Path Pattern

The output path uses `????` as a placeholder for frame numbering:
- `thumbnail_????.jpg` → `thumbnail_0000.jpg`
- `output/frame_????.jpg` → `output/frame_0000.jpg`
- `preview_????.png` → Won't work, output is always JPEG

### Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║     ApraPipes Sample: Thumbnail Generator                   ║
╚══════════════════════════════════════════════════════════════╝

Setting up thumbnail generation pipeline...
  Input video: input.mp4
  Output path: thumbnail_????.jpg

[1/6] Setting up MP4 reader...
  ✓ MP4 reader configured
[2/6] Setting up H264 decoder...
  ✓ H264 decoder configured
[3/6] Setting up valve module...
  ✓ Valve module configured (initially closed)
[4/6] Setting up CUDA memory copy...
  ✓ CUDA memory copy configured
[5/6] Setting up JPEG encoder...
  ✓ JPEG encoder configured
[6/6] Setting up file writer...
  ✓ File writer configured

✓ Pipeline setup completed successfully!

Initializing and starting pipeline...
✓ Pipeline started successfully!

Opening valve to capture 1 frame...
✓ Valve opened (1 frame allowed)

Waiting for thumbnail generation...

Stopping pipeline...
✓ Pipeline stopped successfully!

╔══════════════════════════════════════════════════════════════╗
║     Thumbnail Generated Successfully!                        ║
╚══════════════════════════════════════════════════════════════╝

Output saved to: thumbnail_????.jpg
(Replace ???? with 0000 to see the actual filename)
```

## Understanding the Code

### ThumbnailGenerator Class

```cpp
class ThumbnailGenerator {
public:
    ThumbnailGenerator();
    bool setupPipeline(const std::string &videoPath, const std::string &outPath);
    bool startPipeline();
    bool stopPipeline();

private:
    PipeLine pipeline;
    boost::shared_ptr<Mp4ReaderSource> mp4Reader;
    boost::shared_ptr<H264Decoder> decoder;
    boost::shared_ptr<ValveModule> valve;
    boost::shared_ptr<CudaMemCopy> cudaCopy;
    boost::shared_ptr<JPEGEncoderNVJPEG> jpegEncoder;
    boost::shared_ptr<FileWriterModule> fileWriter;
};
```

### Valve Control Pattern

```cpp
// 1. Create valve initially closed
valve = boost::shared_ptr<ValveModule>(
    new ValveModule(ValveModuleProps(0))  // 0 = initially closed
);

// 2. Start pipeline
pipeline.run_all_threaded();

// 3. Open valve to allow exactly 1 frame
valve->allowFrames(1);

// 4. Wait for processing
boost::this_thread::sleep_for(boost::chrono::seconds(5));

// 5. Stop pipeline
pipeline.stop();
```

## Use Cases

### 1. Video Library Thumbnails

Generate preview images for a video library:

```powershell
# Process all videos in a directory
for video in *.mp4; do
    thumbnail_generator.exe "$video" "thumbnails/${video%.mp4}_????.jpg"
done
```

### 2. Video Streaming Platform

Create poster images for video players:

```cpp
// In production code, integrate into video upload pipeline
ThumbnailGenerator gen;
gen.setupPipeline(uploadedVideo, posterImagePath);
gen.startPipeline();
// Store poster image path in database
```

### 3. Video Surveillance

Extract first frame from recorded footage for quick review:

```powershell
# Generate thumbnails for surveillance recordings
thumbnail_generator.exe camera1_20240115.mp4 preview_????.jpg
```

### 4. Video Analysis

Extract frames for computer vision preprocessing:

```cpp
// Extract every Nth frame using ValveModule
valve->allowFrames(10);  // Extract 10 frames
// Process extracted frames with CV algorithms
```

## Configuration Options

### MP4 Reader Configuration

```cpp
Mp4ReaderSourceProps(
    filePath,      // Path to MP4 file
    parseFS,       // Parse filesystem (usually false for single frame)
    startFrame,    // Starting frame number (0 = beginning)
    readLoop,      // Loop playback (doesn't matter for single frame)
    rewindOnLoop,  // Rewind to start on loop
    direction      // false = forward, true = reverse
)
```

### JPEG Encoder Quality

The `JPEGEncoderNVJPEG` uses default quality settings. To customize:

```cpp
JPEGEncoderNVJPEGProps props(stream);
// props.quality = 95;  // Set quality (check actual API)
auto jpegEncoder = boost::shared_ptr<JPEGEncoderNVJPEG>(
    new JPEGEncoderNVJPEG(props)
);
```

### Extracting Different Frames

To extract a frame other than the first:

```cpp
// Modify Mp4ReaderSourceProps
mp4ReaderProps.startFrame = 100;  // Start at frame 100

// Or use seek functionality
mp4Reader->randomSeek(timestamp, false);
```

## Troubleshooting

### Error: "Failed to setup pipeline"

**Possible causes:**

1. **Input file not found**
   - Verify file path is correct
   - Use absolute path if relative path fails

2. **Unsupported codec**
   - Only H264 video is supported
   - Check codec: `ffmpeg -i video.mp4`
   - Re-encode if needed: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

3. **CUDA not available**
   - Verify CUDA Toolkit is installed
   - Check GPU is detected: `nvidia-smi`
   - Ensure NVJPEG library is available

### Error: "Failed to start pipeline"

- Check console output for detailed error messages
- Verify CUDA device has sufficient memory
- Ensure video file is not corrupted

### No Output File Created

1. **Check output path permissions**
   - Ensure directory exists
   - Verify write permissions

2. **Check valve opened correctly**
   - Look for "Valve opened (1 frame allowed)" message
   - If not present, valve didn't open

3. **Wait longer**
   - Large videos may take more time to decode first frame
   - Increase sleep duration if needed

### Poor Thumbnail Quality

- JPEG encoder uses default quality settings
- For higher quality, modify encoder properties
- Consider PNG output for lossless compression (requires different encoder)

### Wrong Frame Extracted

- By default, extracts first frame
- To extract different frame, use `startFrame` property or seek

## Advanced Usage

### Batch Thumbnail Generation

Extract thumbnails from multiple videos:

```cpp
std::vector<std::string> videos = {"vid1.mp4", "vid2.mp4", "vid3.mp4"};
for (const auto& video : videos) {
    ThumbnailGenerator gen;
    std::string output = video + "_thumb_????.jpg";
    gen.setupPipeline(video, output);
    gen.startPipeline();
    // Wait and stop
}
```

### Extract Multiple Frames

Modify valve to allow multiple frames:

```cpp
// Allow 5 frames (creates 5 thumbnails)
valve->allowFrames(5);

// Wait longer for processing
boost::this_thread::sleep_for(boost::chrono::seconds(10));
```

### Different Output Format

To use CPU-based JPEG encoder (OpenCV):

```cpp
// Replace JPEGEncoderNVJPEG with JPEGEncoderCV
#include "JPEGEncoderCV.h"

auto jpegEncoder = boost::shared_ptr<JPEGEncoderCV>(
    new JPEGEncoderCV(JPEGEncoderCVProps())
);
// No need for CUDA copy in this case
```

## Performance Considerations

### GPU vs CPU Encoding

- **NVJPEG (GPU)**: Faster for batch operations, requires CUDA
- **OpenCV (CPU)**: Slower but works without GPU

### Memory Usage

- Minimal: Only one frame in memory at a time
- GPU memory: ~50-100 MB depending on frame size
- Host memory: ~10-50 MB

### Processing Time

- Typical: 100-500ms for first frame extraction
- Factors:
  - Video resolution (1080p vs 4K)
  - H264 keyframe distance
  - GPU performance

## Learning Points

### For New Users

1. **ValveModule**: Powerful frame filtering mechanism
2. **MP4 Reading**: How to extract frames from video containers
3. **CUDA Pipeline**: Host-to-device memory transfers
4. **JPEG Encoding**: GPU-accelerated image encoding

### For Advanced Users

1. **Frame Control**: Precise control over frame flow
2. **Pipeline Optimization**: When to use GPU vs CPU
3. **Memory Management**: Efficient single-frame processing
4. **Batch Processing**: Scaling to multiple videos

## Related Samples

- **face_detection_cpu**: Video processing with frame-by-frame analysis
- **file_reader**: Full video playback and seeking
- **timelapse**: Multi-frame extraction with motion filtering

## Technical Details

### Module Specifications

| Module | Input | Output | Notes |
|--------|-------|--------|-------|
| Mp4ReaderSource | None | H264 | Reads from file |
| H264Decoder | H264 | YUV420 | CPU decode |
| ValveModule | YUV420 | YUV420 | Filters frames |
| CudaMemCopy | YUV420 (host) | YUV420 (device) | H2D transfer |
| JPEGEncoderNVJPEG | YUV420 (device) | JPEG | GPU encode |
| FileWriterModule | JPEG | File | Writes to disk |

### Memory Flow

```
[Disk] → [Host Memory] → [GPU Memory] → [GPU Encode] → [Host Memory] → [Disk]
         (MP4/H264)       (YUV420)       (JPEG)         (JPEG)
```

## FAQ

**Q: Can I extract a frame from the middle of the video?**
A: Yes, use `mp4ReaderProps.startFrame` or `mp4Reader->randomSeek(timestamp)`.

**Q: Does this work with non-H264 videos?**
A: No, this sample specifically uses H264Decoder. For other codecs, use appropriate decoder modules.

**Q: Can I extract multiple thumbnails?**
A: Yes, call `valve->allowFrames(N)` with N > 1.

**Q: Why use CUDA for a single frame?**
A: For batch operations with many videos, GPU acceleration provides significant speedup.

**Q: Can I use this without NVIDIA GPU?**
A: Yes, replace JPEGEncoderNVJPEG with JPEGEncoderCV and remove CudaMemCopy.

**Q: What's the output quality?**
A: NVJPEG uses default quality (typically 90-95). Check encoder properties for customization.

## References

- [H.264/AVC Standard](https://en.wikipedia.org/wiki/Advanced_Video_Coding)
- [NVJPEG Documentation](https://docs.nvidia.com/cuda/nvjpeg/index.html)
- [MP4 Container Format](https://en.wikipedia.org/wiki/MPEG-4_Part_14)

## License

This sample code is provided as part of the ApraPipes project for educational and demonstration purposes.
