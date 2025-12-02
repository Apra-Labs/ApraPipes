# How to Run Face Detection CPU Sample

## Location
The executable is located at:
```
D:\dws\ApraPipes\samples\_build\RelWithDebInfo\face_detection_cpu.exe
```

## Prerequisites
- Webcam connected to your computer
- Caffe model files (already in place at `./data/assets/`)

## Running the Sample

### Option 1: From PowerShell/Command Prompt
```powershell
cd D:\dws\ApraPipes\samples\_build\RelWithDebInfo
.\face_detection_cpu.exe
```

### Option 2: From Project Root
```powershell
cd D:\dws\ApraPipes\samples\_build\RelWithDebInfo
.\face_detection_cpu.exe
```

### Option 3: Double-click
You can also double-click the executable in Windows Explorer:
```
D:\dws\ApraPipes\samples\_build\RelWithDebInfo\face_detection_cpu.exe
```

## What to Expect

When you run the sample:
1. Console output will show pipeline setup information
2. A window will open showing your webcam feed
3. **Green rectangles** will appear around detected faces
4. **White text** will show the confidence score (as a percentage) above each face
5. Press `ESC` or `Q` to quit

## Sample Output
```
============================================================
     ApraPipes Sample: Face Detection CPU
=============================================================

Setting up face detection pipeline...
  Camera ID: 0
  Scale Factor: 1
  Detection Threshold: 0.7
  Model paths (hardcoded in FaceDetectorXform):
    Config: ./data/assets/deploy.prototxt
    Weights: ./data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel

2025-Oct-13 XX:XX:XX [info] Registering built-in metadata conversions...
2025-Oct-13 XX:XX:XX [info] <OverlayModule_3>::validateInputPins Auto-conversion available from 17 to 11
✓ Pipeline setup completed successfully!
✓ Pipeline initialized successfully!

Pipeline running! Press ESC or Q in the viewer window to stop...
```

## Features Demonstrated

This sample demonstrates the **automatic metadata type conversion** feature:

1. **FaceDetectorXform** outputs `FACEDETECTS_INFO` (type 17) metadata
2. **OverlayModule** expects `OVERLAY_INFO_IMAGE` (type 11) metadata
3. **MetadataRegistry** automatically converts between these types
4. The conversion creates visualization with:
   - Green bounding boxes around faces
   - White text showing confidence percentages

No manual converter module is needed - the conversion happens automatically!

## Troubleshooting

### Camera Not Found
If you see "Failed to open camera", check:
- Webcam is connected
- No other application is using the webcam
- Camera permissions are granted

### Model Files Missing
If you see model file errors:
```bash
# Copy model files to the correct location
cd D:\dws\ApraPipes
cp -r data samples\_build\RelWithDebInfo\
```

### No Window Appears
- Make sure you're running from the correct directory
- Check that all DLLs were copied (should happen automatically during build)
