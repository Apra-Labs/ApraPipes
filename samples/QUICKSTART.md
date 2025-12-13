# ApraPipes Samples - Quick Start Guide

Get up and running with ApraPipes samples in 5 minutes!

## ⚡ 1-Minute Setup

### Prerequisites Check

```powershell
# Verify you have built ApraPipes library
Test-Path "D:\dws\ApraPipes\_build\RelWithDebInfo\aprapipes.lib"
# Should return: True
```

### Build All Samples

```powershell
cd D:\dws\ApraPipes\samples
.\build_samples.ps1
```

**Expected output**: All 6 samples build successfully
```
✓ hello_pipeline.exe
✓ face_detection_cpu.exe
✓ relay.exe
✓ thumbnail_generator.exe
✓ file_reader.exe
✓ timelapse.exe
```

---

## 🚀 Run Your First Sample (30 seconds)

```powershell
cd samples\_build\RelWithDebInfo
.\hello_pipeline.exe
```

**You should see**:
```
=====================================
  ApraPipes Hello Pipeline Sample
=====================================

✓ Created 3 modules
✓ Connected modules in pipeline
✓ Initialized all modules
✓ Processed 5 frames successfully
✓ Clean termination
```

**Success!** ✅ Your ApraPipes samples are working!

---

## 📚 What's Next?

### Choose Your Learning Path

#### 🌱 **New to ApraPipes?** Start Here

1. **hello_pipeline** (5 min) - Learn the basics
   ```powershell
   .\hello_pipeline.exe
   ```

2. Read the [Main README](README.md) for detailed overview

#### 🎥 **Want Video Processing?** Try These

3. **file_reader** (10 min) - Play MP4 videos
   ```powershell
   # Need: An MP4 video file
   .\file_reader.exe "path\to\video.mp4"
   ```

4. **thumbnail_generator** (10 min) - Extract video thumbnails
   ```powershell
   # Need: An MP4 video file
   .\thumbnail_generator.exe "input.mp4" "thumbnail.jpg"
   ```

#### 👁️ **Interested in Computer Vision?**

5. **face_detection_cpu** (15 min) - Real-time face detection
   ```powershell
   # Need: Webcam + model files
   .\face_detection_cpu.exe
   ```
   📖 See [face_detection_cpu README](video/face_detection_cpu/README.md) for model setup

#### 🚀 **Ready for Advanced Features?**

6. **relay** (20 min) - Switch between video sources
   ```powershell
   .\relay.exe
   ```

7. **timelapse** (20 min) - Motion-based video summaries
   ```powershell
   .\timelapse.exe "input.mp4" "summary.mp4"
   ```

---

## 🛠️ Quick Reference

### Sample Requirements

| Sample | Webcam | Video File | Model Files | Notes |
|--------|--------|------------|-------------|-------|
| hello_pipeline | ❌ | ❌ | ❌ | No dependencies! |
| file_reader | ❌ | ✅ | ❌ | Any H264 MP4 |
| thumbnail_generator | ❌ | ✅ | ❌ | Any H264 MP4 |
| timelapse | ❌ | ✅ | ❌ | Any H264 MP4 |
| face_detection_cpu | ✅ | ❌ | ✅ | Caffe models needed |
| relay | ❌ | ✅ | ❌ | RTSP optional |

### Common Commands

```powershell
# Navigate to samples
cd D:\dws\ApraPipes\samples\_build\RelWithDebInfo

# List all samples
dir *.exe

# Run sample without arguments
.\hello_pipeline.exe
.\face_detection_cpu.exe
.\relay.exe

# Run sample with file input
.\file_reader.exe "C:\Videos\test.mp4"

# Run sample with input and output
.\thumbnail_generator.exe "input.mp4" "output.jpg"
.\timelapse.exe "input.mp4" "output.mp4"
```

---

## 🐛 Quick Troubleshooting

### ❌ "aprapipes.lib not found"

Build the main library first:
```powershell
cd D:\dws\ApraPipes\base
cmake --preset windows-cuda -B ../_build
cmake --build ../_build --config RelWithDebInfo
```

### ❌ "Missing DLL" errors

Rebuild samples (DLLs are copied automatically):
```powershell
cd D:\dws\ApraPipes\samples
.\build_samples.ps1
```

### ❌ "Cannot open file"

Use full path to video file:
```powershell
.\file_reader.exe "C:\Users\YourName\Videos\test.mp4"
```

### ❌ "Failed to open camera"

- Ensure webcam is connected
- Close other apps using camera (Zoom, Teams, etc.)
- Grant camera permissions in Windows Settings

### More Help?

- 📖 [Full README](README.md)
- 🧪 [Testing Guide](TESTING.md)
- 🐛 [Sample-specific READMEs](README.md#available-samples)

---

## 📋 Sample Cheat Sheet

### hello_pipeline
```powershell
.\hello_pipeline.exe
# No arguments needed
# Runtime: ~1 second
# What it does: Demonstrates basic pipeline operations
```

### file_reader
```powershell
.\file_reader.exe <video_path>
# Example: .\file_reader.exe "C:\Videos\sample.mp4"
# Runtime: Plays until video ends or you press ESC
# What it does: Plays MP4 video in window
```

### thumbnail_generator
```powershell
.\thumbnail_generator.exe <input_video> <output_jpg>
# Example: .\thumbnail_generator.exe "video.mp4" "thumb.jpg"
# Runtime: ~2-5 seconds
# What it does: Extracts first frame as JPEG
```

### timelapse
```powershell
.\timelapse.exe <input_video> <output_video>
# Example: .\timelapse.exe "long.mp4" "summary.mp4"
# Runtime: Depends on input video length
# What it does: Creates motion-based summary
```

### face_detection_cpu
```powershell
.\face_detection_cpu.exe
# No arguments (uses webcam 0)
# Runtime: 50 seconds (auto-stops)
# What it does: Detects faces in webcam feed
# NOTE: Requires model files - see README
```

### relay
```powershell
.\relay.exe
# No arguments (uses default paths)
# Runtime: Until you press ESC
# What it does: Switches between RTSP and MP4 sources
# NOTE: Requires video file or RTSP stream
```

---

## 🎯 Quick Goals

**Next 5 minutes**:
- ✅ Run hello_pipeline
- ✅ Read [Main README](README.md) overview

**Next 30 minutes**:
- ✅ Get a test MP4 video file
- ✅ Run file_reader with your video
- ✅ Generate a thumbnail with thumbnail_generator

**Next hour**:
- ✅ Setup face detection models (if you have a webcam)
- ✅ OR create a timelapse summary from a long video

**Next 2 hours**:
- ✅ Read sample-specific READMEs
- ✅ Modify sample code to fit your use case
- ✅ Experiment with pipeline configurations

---

## 💡 Tips for Success

1. **Start Simple**: Run hello_pipeline first to verify everything works
2. **Use Test Data**: Have some MP4 video files ready for testing
3. **Read READMEs**: Each sample has detailed documentation
4. **Check Console Output**: Samples print helpful status messages
5. **Experiment**: Modify code and rebuild to learn how it works

---

## 🔗 Important Links

- [📖 Main Samples README](README.md) - Comprehensive documentation
- [🧪 Testing Guide](TESTING.md) - Test results and procedures
- [📚 Sample-Specific Docs](README.md#available-samples) - Detailed guides for each sample
- [🌐 ApraPipes Repository](https://github.com/Apra-Labs/ApraPipes)

---

## ✅ Next Steps

You're ready to start using ApraPipes! Here's what to do:

1. ✅ **Completed**: Built and ran hello_pipeline
2. **Now**: Pick a sample that matches your interest
3. **Then**: Read its specific README
4. **Finally**: Start building your own pipelines!

---

**Happy coding with ApraPipes! 🚀**

Got questions? Check the [Main README](README.md) or open an issue on GitHub.
