# Relay Sample - Dynamic Source Switching

**Category**: Network/Streaming
**Difficulty**: Intermediate
**Dependencies**: RTSP camera, MP4 file, H264 decoder

## Overview

This sample demonstrates the **relay pattern** in ApraPipes, which allows dynamically switching between multiple input sources without stopping the pipeline. This is useful for applications that need to switch between live camera feeds and recorded video, or between multiple camera sources.

### What You'll Learn

- How to use the relay pattern for source switching
- RTSP streaming from network cameras
- MP4 file reading and playback
- H264 video decoding
- Interactive pipeline control
- Managing multiple input sources

## Pipeline Structure

```
[RTSPClientSrc]  ─┐
                   ├─> [H264Decoder] → [ColorConversion] → [ImageViewer]
[Mp4ReaderSource]─┘
```

### Module Description

1. **RTSPClientSrc**: Connects to network camera via RTSP protocol
2. **Mp4ReaderSource**: Reads H264 video from MP4 file
3. **H264Decoder**: Decodes H264 compressed video (receives from both sources)
4. **ColorConversion**: Converts YUV420 to RGB for display
5. **ImageViewerModule**: Displays the video frames in a window

### Key Concept: Relay Pattern

The relay pattern allows multiple sources to be connected to a single processing module, but only one source is active at a time:

- **relay(module, true)**: Enable this source (disable others)
- **relay(module, false)**: Disable this source
- Switching is done at runtime without stopping the pipeline
- No frame loss during switching

## Prerequisites

### Hardware Requirements

- Network camera with RTSP support (e.g., IP camera, security camera)
- OR RTSP test server (e.g., MediaMTX, RTSP Simple Server)

### Software Requirements

- ApraPipes library built with RTSP support
- H264 decoder (included in ApraPipes)
- MP4 video file with H264 codec

### Test Resources

If you don't have an RTSP camera, you can use test servers:

**Option 1: MediaMTX (recommended)**
```bash
# Download and run MediaMTX
# https://github.com/bluenviron/mediamtx

# Stream a file via RTSP
ffmpeg -re -i video.mp4 -c copy -f rtsp rtsp://localhost:8554/mystream
```

**Option 2: RTSP Simple Server**
```bash
# https://github.com/aler9/rtsp-simple-server
# Provides test RTSP streams
```

## Building

The relay sample is built as part of the samples build system:

```powershell
# From repository root
cd samples
.\build_samples.ps1 -BuildType RelWithDebInfo
```

The executable will be created at:
```
samples\_build\RelWithDebInfo\relay.exe
```

## Running the Sample

### Basic Usage

```powershell
.\relay.exe <rtsp_url> <mp4_file_path>
```

### Example with Real Camera

```powershell
.\relay.exe rtsp://192.168.1.100:554/stream video.mp4
```

### Example with Test Server

```powershell
.\relay.exe rtsp://localhost:8554/mystream test_video.mp4
```

### Example with Authentication

```powershell
# For cameras requiring authentication, embed credentials in URL
.\relay.exe rtsp://user:password@192.168.1.100:554/stream video.mp4
```

**⚠️ Security Warning**: Never commit RTSP URLs with credentials to version control!

## Interactive Controls

Once running, use keyboard controls to switch sources:

| Key | Action |
|-----|--------|
| `r` | Switch to **RTSP** source (live camera) |
| `m` | Switch to **MP4** source (recorded video) |
| `s` | **Stop** and exit |

### Expected Behavior

```
Setting up relay pipeline...
  RTSP URL: rtsp://localhost:8554/mystream
  MP4 Path: video.mp4

[1/5] Setting up RTSP source...
  ✓ RTSP source configured
[2/5] Setting up MP4 source...
  ✓ MP4 source configured
[3/5] Setting up H264 decoder...
  ✓ H264 decoder configured with dual inputs
[4/5] Setting up color conversion...
  ✓ Color conversion configured
[5/5] Setting up image viewer...
  ✓ Image viewer configured

✓ Pipeline started successfully!

Default source: RTSP

Keyboard Controls:
  'r' - Switch to RTSP source
  'm' - Switch to MP4 source
  's' - Stop and exit

Waiting for keyboard input...
→ Switched to MP4 source
→ Switched to RTSP source
```

## Understanding the Code

### RelayPipeline Class

```cpp
class RelayPipeline {
public:
    bool setupPipeline(const std::string &rtspUrl, const std::string &mp4VideoPath);
    bool startPipeline();
    bool stopPipeline();
    void addRelayToRtsp(bool open);  // Switch to RTSP
    void addRelayToMp4(bool open);   // Switch to MP4

private:
    PipeLine pipeline;
    boost::shared_ptr<RTSPClientSrc> rtspSource;
    boost::shared_ptr<Mp4ReaderSource> mp4ReaderSource;
    boost::shared_ptr<H264Decoder> h264Decoder;
    boost::shared_ptr<ColorConversion> colorConversion;
    boost::shared_ptr<ImageViewerModule> imageViewer;
};
```

### Source Switching Logic

```cpp
// Switch to RTSP source
pipelineInstance.addRelayToRtsp(false);  // Disable MP4
pipelineInstance.addRelayToMp4(true);    // Enable RTSP

// Switch to MP4 source
pipelineInstance.addRelayToMp4(false);   // Disable RTSP
pipelineInstance.addRelayToRtsp(true);   // Enable MP4
```

**Note**: The function names might seem inverted, but this is the correct usage pattern. The `relay()` method takes a target module and an enable flag.

## Configuration Options

### RTSP Source Configuration

```cpp
RTSPClientSrcProps(url, username, password)
```

- **url**: RTSP URL (e.g., `rtsp://192.168.1.100:554/stream`)
- **username**: Authentication username (or empty string)
- **password**: Authentication password (or empty string)

### MP4 Source Configuration

```cpp
Mp4ReaderSourceProps(
    filePath,      // Path to MP4 file
    parseFS,       // Parse filesystem (usually false)
    startFrame,    // Starting frame number (0 = beginning)
    readLoop,      // Loop playback (true/false)
    rewindOnLoop,  // Rewind to start on loop (true/false)
    direction      // false = forward, true = reverse
)
```

Additional properties:
- `mp4ReaderProps.fps = 9;`  // Playback frame rate

### H264 Metadata Configuration

```cpp
H264Metadata(width, height)
```

Example:
- `H264Metadata(1280, 720)` - 720p HD
- `H264Metadata(1920, 1080)` - 1080p Full HD
- `H264Metadata(3840, 2160)` - 4K UHD

**Important**: The resolution must match your camera/video file resolution!

## Troubleshooting

### Error: "Failed to setup pipeline"

**Possible causes:**

1. **RTSP camera not reachable**
   - Check network connectivity: `ping camera_ip`
   - Verify RTSP port is open (default: 554)
   - Check firewall settings

2. **MP4 file not found**
   - Verify file path is correct
   - Use absolute path if relative path fails
   - Check file permissions

3. **Wrong codec**
   - This sample only works with H264 video
   - Check video codec: `ffmpeg -i video.mp4`
   - Re-encode if needed: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

4. **Resolution mismatch**
   - Camera resolution doesn't match metadata configuration
   - Check actual resolution and update H264Metadata accordingly

### Error: "Failed to start pipeline"

- Check console output for detailed error messages
- Verify both RTSP and MP4 sources are valid
- Ensure H264 decoder is available

### No Video Display

1. **Check source status**
   - Is the correct source active?
   - Try switching sources with 'r' and 'm' keys

2. **RTSP stream issues**
   - Test RTSP URL in VLC player first
   - Check authentication credentials
   - Verify stream is H264 (not MJPEG or other codecs)

3. **MP4 playback issues**
   - Verify file is not corrupted
   - Check codec with: `ffmpeg -i video.mp4`
   - Ensure file contains H264 video stream

### Keyboard Input Not Working

- Make sure terminal/console window has focus
- On Windows, terminal might need to be in raw input mode
- Press Enter after each key on some systems

### Connection Timeout (RTSP)

- Camera may require specific RTSP parameters
- Try different RTSP transport modes (TCP vs UDP)
- Check camera's RTSP URL format in documentation
- Some cameras use non-standard ports (not 554)

## Use Cases

### 1. Security System with Playback

Switch between live camera feed and recorded incidents:
```cpp
// Live monitoring mode
switchToRTSP();

// Review recorded incident
switchToMP4("incident_2024_01_15.mp4");
```

### 2. Live vs Recorded Comparison

Compare live feed with reference recording:
```cpp
// Show live behavior
switchToRTSP();

// Compare with reference
switchToMP4("reference_behavior.mp4");
```

### 3. Testing and Development

Test algorithms with both live and recorded data:
```cpp
// Test with live data
switchToRTSP();

// Replay edge cases from files
switchToMP4("edge_case_001.mp4");
```

### 4. Backup/Fallback System

Automatically switch to backup when primary source fails:
```cpp
if (rtspSourceFailed) {
    switchToMP4("backup_stream.mp4");
}
```

## Extending the Sample

### Add More Sources

```cpp
// Add a third source
boost::shared_ptr<WebCamSource> webcamSource;
webcamSource = boost::shared_ptr<WebCamSource>(new WebCamSource(...));
webcamSource->setNext(h264Decoder);

// Switch to webcam
webcamSource->relay(h264Decoder, true);
```

### Add Frame Processing

```cpp
// Insert processing between decoder and display
auto faceDetector = boost::shared_ptr<FaceDetectorXform>(...);
h264Decoder->setNext(faceDetector);
faceDetector->setNext(colorConversion);
```

### Save Active Source to File

```cpp
// Record the current active source
auto mp4Writer = boost::shared_ptr<Mp4WriterSink>(...);
colorConversion->setNext(mp4Writer);
```

## Performance Considerations

### CPU Usage

- H264 decoding is CPU-intensive
- Use hardware acceleration if available
- Consider limiting frame rate for lower CPU usage

### Network Bandwidth

- RTSP streaming requires continuous network bandwidth
- 1080p H264 @ 30fps ≈ 2-8 Mbps depending on quality
- Monitor network quality for live streams

### Switching Latency

- Source switching is typically instantaneous
- May see 1-2 frame delay during switch
- No pipeline restart required

## Learning Points

### For New Users

1. **Relay Pattern**: How to manage multiple input sources
2. **RTSP Streaming**: Working with network cameras
3. **MP4 Playback**: Reading video files
4. **Interactive Control**: User input handling in pipelines
5. **Source Management**: Enabling/disabling sources dynamically

### For Advanced Users

1. **Multiple Source Architecture**: Designing pipelines with redundancy
2. **Network Protocols**: Understanding RTSP, RTP, RTCP
3. **Buffer Management**: How relay handles frame buffers
4. **Synchronization**: Timing considerations when switching sources
5. **Error Recovery**: Handling source failures gracefully

## Related Samples

- **face_detection_cpu**: Video processing with OpenCV
- **play_mp4_from_beginning**: Simple MP4 playback
- **timelapse**: Time-based video processing

## Technical Details

### Module Specifications

| Module | Input | Output | Notes |
|--------|-------|--------|-------|
| RTSPClientSrc | None | H264 | Network stream |
| Mp4ReaderSource | None | H264 | File read |
| H264Decoder | H264 | YUV420 | CPU decode |
| ColorConversion | YUV420 | RGB | Format conversion |
| ImageViewerModule | RGB | Display | OpenCV window |

### Memory Usage

- RTSP buffering: ~10-50 MB depending on stream
- MP4 file reader: ~10-100 MB depending on file size and caching
- H264 decoder: ~50-200 MB for frame buffers
- Total: ~100-400 MB typical

### Thread Model

- Each source module runs in its own thread
- Decoder runs in separate thread
- Display runs in main thread
- Total: 4-5 threads typical

## FAQ

**Q: Can I switch between more than 2 sources?**
A: Yes! The relay pattern supports any number of sources. Just add more sources and connect them to the decoder.

**Q: Does switching sources cause frame loss?**
A: Minimal. You may lose 1-2 frames during the switch, but it's typically seamless.

**Q: Can I use non-H264 video?**
A: No, this sample specifically uses H264Decoder. You would need to use appropriate decoders for other codecs.

**Q: How do I get an RTSP URL from my camera?**
A: Check your camera's documentation. Common formats:
- `rtsp://camera_ip:554/stream1`
- `rtsp://camera_ip/live/main`
- `rtsp://camera_ip:8554/h264`

**Q: Can I use HTTPS or HTTP streams?**
A: Not with RTSPClientSrc. You would need HTTPClientSrc or similar module.

**Q: Why doesn't the sample support GPU decoding?**
A: For simplicity, this sample uses CPU decoding. GPU decoding can be added by replacing H264Decoder with a CUDA-based decoder module.

## References

- [RTSP Protocol](https://en.wikipedia.org/wiki/Real_Time_Streaming_Protocol)
- [H.264/AVC Standard](https://en.wikipedia.org/wiki/Advanced_Video_Coding)
- [MP4 Container Format](https://en.wikipedia.org/wiki/MPEG-4_Part_14)

## License

This sample code is provided as part of the ApraPipes project for educational and demonstration purposes.
