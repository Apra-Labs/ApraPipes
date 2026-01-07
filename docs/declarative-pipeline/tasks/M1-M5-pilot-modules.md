# M1: Add Metadata to FileReaderModule

**Sprint:** 1-2 (Week 2-3)  
**Priority:** P1 - High  
**Effort:** 1 day  
**Depends On:** A2 (Module Registry)  
**Blocks:** Integration tests  

## Description

Add `Metadata` struct to `FileReaderModule` and register it with `REGISTER_MODULE`. This is the first **Source** module in the pilot.

## Files to Modify
```
base/include/FileReaderModule.h
base/src/FileReaderModule.cpp
```

## Metadata to Add

```cpp
class FileReaderModule : public Module {
public:
    struct Metadata {
        static constexpr std::string_view name = "FileReaderModule";
        static constexpr ModuleCategory category = ModuleCategory::Source;
        static constexpr std::string_view version = "1.0";
        static constexpr std::string_view description = 
            "Reads video frames from a file. Supports MP4, MKV, and other "
            "formats via FFmpeg demuxing.";
        
        static constexpr std::array<std::string_view, 3> tags = {
            "reader", "file", "ffmpeg"
        };
        
        static constexpr std::array inputs = {};  // Source has no inputs
        
        static constexpr std::array outputs = {
            PinDef{
                .name = "output",
                .frame_types = {"H264Frame", "H265Frame"},
                .required = true,
                .description = "Encoded video frames from file"
            }
        };
        
        static constexpr std::array properties = {
            PropDef::String("path", "", "Path to video file", ".*"),
            PropDef::Bool("loop", false, "Loop playback"),
            PropDef::Int("start_frame", 0, 0, INT_MAX, "Starting frame number"),
            PropDef::DynamicBool("paused", false, "Pause playback")
        };
    };
    
    // ... existing implementation
};

// In .cpp file:
REGISTER_MODULE(FileReaderModule, FileReaderModuleProps)
```

## Acceptance Criteria
- [ ] Metadata struct exists with all fields
- [ ] REGISTER_MODULE in .cpp file
- [ ] `ModuleRegistry::instance().hasModule("FileReaderModule")` returns true
- [ ] Module appears in `list-modules` output
- [ ] Schema generator includes this module
- [ ] Unit test verifies metadata is correct

---

# M2: Add Metadata to H264Decoder

**Sprint:** 1-2 (Week 2-3)  
**Priority:** P1 - High  
**Effort:** 1 day  
**Depends On:** A2 (Module Registry)  
**Blocks:** Integration tests  

## Description

Add `Metadata` struct to `H264Decoder` (or `H264DecoderNvCodec` for CUDA builds). This is the first **Transform** module in the pilot.

## Files to Modify
```
base/include/H264Decoder.h (or H264DecoderNvCodec.h)
base/src/H264Decoder.cpp
```

## Metadata to Add

```cpp
class H264Decoder : public Module {
public:
    struct Metadata {
        static constexpr std::string_view name = "H264Decoder";
        static constexpr ModuleCategory category = ModuleCategory::Transform;
        static constexpr std::string_view version = "1.0";
        static constexpr std::string_view description = 
            "Decodes H.264/AVC video frames to raw planar format. "
            "Uses NvCodec on CUDA platforms, FFmpeg otherwise.";
        
        static constexpr std::array<std::string_view, 4> tags = {
            "decoder", "h264", "nvidia", "cuda_optional"
        };
        
        static constexpr std::array inputs = {
            PinDef{
                .name = "input",
                .frame_types = {"H264Frame"},
                .required = true,
                .description = "H.264 encoded NAL units"
            }
        };
        
        static constexpr std::array outputs = {
            PinDef{
                .name = "output",
                .frame_types = {"RawImagePlanar"},
                .required = true,
                .description = "Decoded NV12 frames"
            }
        };
        
        static constexpr std::array properties = {
            PropDef::Int("device_id", 0, 0, 7, "CUDA device ID"),
            PropDef::Bool("low_latency", false, "Low latency decode mode")
        };
    };
};

REGISTER_MODULE(H264Decoder, H264DecoderProps)
```

## Acceptance Criteria
- [ ] Metadata struct exists with all fields
- [ ] REGISTER_MODULE in .cpp file
- [ ] Tags include platform info (cuda_optional or cuda_required)
- [ ] Input frame types match actual module behavior
- [ ] Output frame types match actual module behavior

---

# M3: Add Metadata to FaceDetectorXform

**Sprint:** 2 (Week 3)  
**Priority:** P1 - High  
**Effort:** 1 day  
**Depends On:** A2 (Module Registry)  
**Blocks:** Integration tests  

## Description

Add `Metadata` struct to `FaceDetectorXform`. This is the first **Analytics** module in the pilot.

## Files to Modify
```
base/include/FaceDetectorXform.h
base/src/FaceDetectorXform.cpp
```

## Metadata to Add

```cpp
class FaceDetectorXform : public Module {
public:
    struct Metadata {
        static constexpr std::string_view name = "FaceDetectorXform";
        static constexpr ModuleCategory category = ModuleCategory::Analytics;
        static constexpr std::string_view version = "1.0";
        static constexpr std::string_view description = 
            "Detects faces in video frames using OpenCV DNN or Haar cascades. "
            "Outputs detection results with bounding boxes and landmarks.";
        
        static constexpr std::array<std::string_view, 3> tags = {
            "detector", "face", "opencv"
        };
        
        static constexpr std::array inputs = {
            PinDef{
                .name = "input",
                .frame_types = {"RawImagePlanar", "RawImagePacked"},
                .required = true,
                .description = "Video frames to analyze"
            }
        };
        
        static constexpr std::array outputs = {
            PinDef{
                .name = "output",
                .frame_types = {"RawImagePlanar", "RawImagePacked"},
                .required = true,
                .description = "Passthrough frames with detection metadata"
            },
            PinDef{
                .name = "detections",
                .frame_types = {"DetectionResultFrame"},
                .required = false,
                .description = "Face detection results"
            }
        };
        
        static constexpr std::array properties = {
            PropDef::String("model_path", "", "Path to detection model"),
            PropDef::DynamicFloat("confidence_threshold", 0.5, 0.0, 1.0, 
                "Minimum confidence for detection"),
            PropDef::DynamicBool("enabled", true, "Enable/disable detection")
        };
    };
};

REGISTER_MODULE(FaceDetectorXform, FaceDetectorXformProps)
```

## Acceptance Criteria
- [ ] Metadata struct exists with all fields
- [ ] Detection output pin is optional (required = false)
- [ ] Dynamic properties for runtime control
- [ ] Model path is configurable

---

# M4: Add Metadata to QRReader

**Sprint:** 2 (Week 3)  
**Priority:** P1 - High  
**Effort:** 1 day  
**Depends On:** A2 (Module Registry)  
**Blocks:** Integration tests  

## Description

Add `Metadata` struct to `QRReader`. This is the second **Analytics** module in the pilot.

## Files to Modify
```
base/include/QRReader.h
base/src/QRReader.cpp
```

## Metadata to Add

```cpp
class QRReader : public Module {
public:
    struct Metadata {
        static constexpr std::string_view name = "QRReader";
        static constexpr ModuleCategory category = ModuleCategory::Analytics;
        static constexpr std::string_view version = "1.0";
        static constexpr std::string_view description = 
            "Detects and decodes QR codes and barcodes in video frames. "
            "Uses ZXing library for multi-format barcode support.";
        
        static constexpr std::array<std::string_view, 3> tags = {
            "detector", "qr", "barcode"
        };
        
        static constexpr std::array inputs = {
            PinDef{
                .name = "input",
                .frame_types = {"RawImagePlanar", "RawImagePacked"},
                .required = true,
                .description = "Video frames to scan for QR codes"
            }
        };
        
        static constexpr std::array outputs = {
            PinDef{
                .name = "output",
                .frame_types = {"RawImagePlanar", "RawImagePacked"},
                .required = true,
                .description = "Passthrough frames"
            },
            PinDef{
                .name = "qr_results",
                .frame_types = {"QRResultFrame"},
                .required = false,
                .description = "Decoded QR/barcode data"
            }
        };
        
        static constexpr std::array properties = {
            PropDef::DynamicBool("enabled", true, "Enable/disable scanning"),
            PropDef::Int("scan_interval", 1, 1, 100, 
                "Scan every N frames (1 = every frame)")
        };
    };
};

REGISTER_MODULE(QRReader, QRReaderProps)
```

## Acceptance Criteria
- [ ] Metadata struct exists
- [ ] QRResultFrame type referenced (may need to create in F-tasks)
- [ ] Scan interval is static (affects internal state)
- [ ] enabled is dynamic for runtime toggle

---

# M5: Add Metadata to FileWriterModule

**Sprint:** 2 (Week 3)  
**Priority:** P1 - High  
**Effort:** 1 day  
**Depends On:** A2 (Module Registry)  
**Blocks:** Integration tests  

## Description

Add `Metadata` struct to `FileWriterModule` (or `Mp4WriterSink`). This is the first **Sink** module in the pilot.

## Files to Modify
```
base/include/FileWriterModule.h (or Mp4WriterSink.h)
base/src/FileWriterModule.cpp
```

## Metadata to Add

```cpp
class FileWriterModule : public Module {
public:
    struct Metadata {
        static constexpr std::string_view name = "FileWriterModule";
        static constexpr ModuleCategory category = ModuleCategory::Sink;
        static constexpr std::string_view version = "1.0";
        static constexpr std::string_view description = 
            "Writes video frames to file. Supports MP4 container with "
            "H.264/H.265 encoding via hardware or software codecs.";
        
        static constexpr std::array<std::string_view, 3> tags = {
            "writer", "file", "mp4"
        };
        
        static constexpr std::array inputs = {
            PinDef{
                .name = "input",
                .frame_types = {"H264Frame", "H265Frame", "RawImagePlanar"},
                .required = true,
                .description = "Frames to write to file"
            }
        };
        
        static constexpr std::array outputs = {};  // Sink has no outputs
        
        static constexpr std::array properties = {
            PropDef::String("path", "", "Output file path", ".*\\.mp4$"),
            PropDef::Int("max_file_size_mb", 0, 0, 10000, 
                "Max file size in MB (0 = unlimited)"),
            PropDef::Bool("overwrite", false, "Overwrite existing file")
        };
    };
};

REGISTER_MODULE(FileWriterModule, FileWriterModuleProps)
```

## Acceptance Criteria
- [ ] Metadata struct exists
- [ ] No outputs (Sink module)
- [ ] Path has regex pattern for .mp4 extension
- [ ] Properties match actual module behavior

---

## Common Testing Strategy for All Modules

For each module M1-M5:

```cpp
// In module_metadata_tests.cpp

TEST(ModuleMetadata, FileReaderModuleRegistered) {
    auto& registry = ModuleRegistry::instance();
    ASSERT_TRUE(registry.hasModule("FileReaderModule"));
    
    auto* info = registry.getModule("FileReaderModule");
    ASSERT_NE(info, nullptr);
    
    EXPECT_EQ(info->category, ModuleCategory::Source);
    EXPECT_FALSE(info->tags.empty());
    EXPECT_TRUE(info->inputs.empty());  // Source has no inputs
    EXPECT_FALSE(info->outputs.empty());
}
```

---

## Definition of Done (for each module)
- [ ] Metadata struct added to header
- [ ] REGISTER_MODULE in .cpp file
- [ ] Unit test verifies registration
- [ ] Schema generator includes module
- [ ] Existing module tests still pass
- [ ] Code reviewed and merged
