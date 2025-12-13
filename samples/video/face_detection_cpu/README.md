# Face Detection CPU Sample

## Overview

This sample demonstrates real-time face detection from webcam input using CPU-based processing with a Caffe deep neural network model.

## What This Sample Demonstrates

- **Webcam Integration**: Capturing live video from a system webcam
- **DNN-Based Face Detection**: Using a Caffe SSD (Single Shot Detector) model for face detection
- **Visual Feedback**: Drawing bounding boxes around detected faces
- **Pipeline Architecture**: Building a multi-stage processing pipeline

## Pipeline Structure

```
[WebCamSource] → [FaceDetectorXform] → [OverlayModule] → [ColorConversion] → [ImageViewerModule]
```

### Module Breakdown

1. **WebCamSource**: Captures video frames from the default webcam (camera ID 0)
2. **FaceDetectorXform**: Applies face detection using Caffe DNN model
3. **OverlayModule**: Draws bounding boxes around detected faces
4. **ColorConversion**: Converts RGB to BGR format for display
5. **ImageViewerModule**: Displays the processed frames in a window

## Prerequisites

### Hardware
- Webcam connected to your system

### Software
- ApraPipes library (already built)
- OpenCV with DNN module (included via vcpkg)

### Model Files

This sample requires Caffe model files to be present in the `./data/assets/` directory (relative to where the executable runs):

1. **deploy.prototxt** - Model architecture definition
2. **res10_300x300_ssd_iter_140000_fp16.caffemodel** - Pre-trained model weights

**Where to get the models:**
- These are standard OpenCV DNN face detection models
- Available from OpenCV's GitHub repository or DNN model zoo
- Or download from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

**File structure:**
```
samples/_build/RelWithDebInfo/
├── face_detection_cpu.exe
└── data/
    └── assets/
        ├── deploy.prototxt
        └── res10_300x300_ssd_iter_140000_fp16.caffemodel
```

## Building

From the `samples/` directory:

```powershell
.\build_samples.ps1
```

Or for Debug build:
```powershell
.\build_samples.ps1 -BuildType Debug
```

## Running

### Windows

```powershell
cd samples\_build\RelWithDebInfo
.\face_detection_cpu.exe
```

### Expected Behavior

1. The sample will open a window titled "Face Detection - CPU"
2. Your webcam feed will appear with bounding boxes around detected faces
3. The pipeline will run for **50 seconds** then automatically stop
4. Press Ctrl+C to stop early

### Console Output

```
╔══════════════════════════════════════════════════════════════╗
║     ApraPipes Sample: Face Detection CPU                    ║
╚══════════════════════════════════════════════════════════════╝

Setting up face detection pipeline...
  Camera ID: 0
  Scale Factor: 1.0
  Detection Threshold: 0.7
  Model Config: ./data/assets/deploy.prototxt
  Model Weights: ./data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel

✓ Pipeline setup completed successfully!

Pipeline structure:
  [WebCam] → [FaceDetector] → [Overlay] → [ColorConversion] → [ImageViewer]

Initializing and starting pipeline...
✓ Pipeline started successfully!

Processing webcam feed with face detection...
Running for 50 seconds...

Stopping pipeline...
✓ Pipeline stopped successfully!
```

## Configuration Parameters

You can modify these constants in `main.cpp` to change behavior:

```cpp
const int cameraId = 0;                    // Webcam device ID
const double scaleFactor = 1.0;            // Image scaling (1.0 = no scaling)
const double confidenceThreshold = 0.7;    // Detection confidence (0.0-1.0)
const int runDurationSeconds = 50;         // How long to run
```

### Detection Threshold

- **0.5**: More detections, some false positives
- **0.7**: Balanced (default)
- **0.9**: Fewer detections, high confidence only

## Troubleshooting

### Error: "Failed to setup pipeline"

**Possible causes:**
1. **Webcam not available**
   - Check if another application is using the webcam
   - Try changing `cameraId` to 1 or 2
   - Verify webcam permissions

2. **Model files not found**
   - Ensure model files are in `./data/assets/` directory
   - Check file paths are correct
   - Download models if missing

3. **OpenCV DNN module not available**
   - Rebuild ApraPipes library with OpenCV support
   - Verify vcpkg installed opencv correctly

### Error: "Failed to start pipeline"

- Check console output for detailed error messages
- Ensure all modules initialized correctly
- Verify webcam is not in use by another app

### No Faces Detected

- Ensure adequate lighting
- Position yourself clearly in front of webcam
- Try lowering `confidenceThreshold` to 0.5
- Check that overlay is working (bounding boxes appear)

### Poor Performance

- The model is CPU-based and may be slow on older hardware
- Consider using GPU-based face detection (different sample)
- Reduce input resolution by adjusting `scaleFactor` (e.g., 0.5 for half size)

### Error: "Missing DLL" (e.g., opencv_core4.dll not found)

**This should NOT happen** - the build system automatically copies all required DLLs.

If you encounter missing DLL errors:

1. **Verify DLLs were copied during build**:
   ```powershell
   dir samples\_build\RelWithDebInfo\*.dll
   ```
   You should see ~81 DLL files (4 Boost + 62 OpenCV + 14 dependencies)

2. **If DLLs are missing, rebuild samples**:
   ```powershell
   cd samples
   .\build_samples.ps1 -BuildType RelWithDebInfo
   ```

3. **Check build output** for DLL copy messages:
   ```
   -- Copying required DLLs for RelWithDebInfo configuration
   --   [1/2] Copying Boost DLLs...
   --   [2/2] Copying OpenCV DLLs and dependencies...
   ```

**Note**: The build system uses `copy_dlls.cmake` to automatically copy:
- Boost runtime DLLs (filesystem, log, serialization, thread)
- All OpenCV DLLs (core, imgproc, highgui, dnn, etc.)
- Image codec DLLs (jpeg, png, tiff, webp)
- Compression DLLs (zlib, zstd, liblzma)
- Serialization DLLs (libprotobuf, libprotobuf-lite, libprotoc)

## Learning Points

### For New Users

1. **Module Pattern**: Each processing step is a separate module
2. **Pipeline Chaining**: Modules are connected using `setNext()`
3. **Lifecycle Management**: init() → run() → stop() → term()
4. **Boost Smart Pointers**: Using `boost::shared_ptr` for memory management
5. **Error Handling**: Try-catch blocks for graceful failure

### Code Structure

- **Class Encapsulation**: Pipeline logic wrapped in `FaceDetectionCPU` class
- **Configuration Parameters**: Passed to constructors via props structs
- **Threading**: Pipeline runs on separate threads (`run_all_threaded()`)
- **Resource Cleanup**: Proper shutdown in `stopPipeline()`

## Extending This Sample

### Ideas to Try

1. **Save detected faces to disk**
   - Add a `FileWriterModule` after the FaceDetector
   - Crop face regions and save as separate images

2. **Count faces**
   - Access face detection metadata
   - Display count in overlay text

3. **Add facial landmarks**
   - Use `FacialLandmarksCV` module (commented in original)
   - Detect eyes, nose, mouth positions

4. **Multiple cameras**
   - Create separate pipelines for different camera IDs
   - Display multiple streams simultaneously

5. **Record to video**
   - Add an encoder module (H.264 or H.265)
   - Save processed video with face boxes to MP4

## Related Samples

- **face_detection_gpu** - GPU-accelerated face detection (faster)
- **facial_landmarks** - Detect facial feature points
- **face_recognition** - Identify specific individuals

## Technical Details

### Caffe Model Specifications

- **Architecture**: SSD (Single Shot Detector)
- **Input Size**: 300x300 pixels
- **Precision**: FP16 (half precision)
- **Framework**: Caffe
- **Trained On**: WIDER FACE dataset

### Performance

- **CPU-based processing** (no GPU required)
- **Frame rate**: Depends on CPU speed (typically 5-15 FPS on modern CPUs)
- **Detection range**: Works best at 0.5m - 3m from camera
- **Multiple faces**: Can detect multiple faces in a single frame

## References

- [OpenCV DNN Module Documentation](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html)
- [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [SSD Paper](https://arxiv.org/abs/1512.02325)

---

**Next Steps**: Try modifying the detection threshold or add face counting functionality!
