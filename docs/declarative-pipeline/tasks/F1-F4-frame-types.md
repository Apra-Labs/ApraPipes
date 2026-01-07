# F1-F4: Frame Type Metadata

**Sprint:** 2 (Week 3)  
**Priority:** P1 - High  
**Effort:** 0.5 day each (2 days total)  
**Depends On:** A3 (FrameType Registry)  
**Blocks:** C4 (Validator Connection Checks)  

## Description

Add `Metadata` structs to core frame types and register them with `REGISTER_FRAME_TYPE`. These are needed for:
1. Validator to check pin compatibility
2. LLM context for understanding data flow
3. Documentation generation

---

# F1: RawImagePlanar Metadata

## File to Modify
```
base/include/RawImageMetadata.h
```

## Metadata to Add

```cpp
class RawImagePlanar {
public:
    struct Metadata {
        static constexpr std::string_view name = "RawImagePlanar";
        static constexpr std::string_view parent = "VideoFrame";
        static constexpr std::string_view description = 
            "Raw planar image with separate Y, U, V planes. "
            "Common formats: NV12, I420, YUV444.";
        
        static constexpr std::array<std::string_view, 4> tags = {
            "video", "raw", "planar", "yuv"
        };
        
        static constexpr std::array attributes = {
            AttrDef::Int("width", true, "Image width in pixels"),
            AttrDef::Int("height", true, "Image height in pixels"),
            AttrDef::Enum("format", {"NV12", "I420", "YUV444"}, true, 
                "Planar format")
        };
    };
};

REGISTER_FRAME_TYPE(RawImagePlanar)
```

---

# F2: H264Frame Metadata

## File to Modify
```
base/include/H264Metadata.h (or create core/frame_types/H264Frame.h)
```

## Metadata to Add

```cpp
class H264Frame {
public:
    struct Metadata {
        static constexpr std::string_view name = "H264Frame";
        static constexpr std::string_view parent = "EncodedVideoFrame";
        static constexpr std::string_view description = 
            "H.264/AVC encoded video frame containing NAL units. "
            "May be a keyframe (IDR) or inter-frame (P/B).";
        
        static constexpr std::array<std::string_view, 3> tags = {
            "video", "encoded", "h264"
        };
        
        static constexpr std::array attributes = {
            AttrDef::Int("width", true, "Frame width"),
            AttrDef::Int("height", true, "Frame height"),
            AttrDef::Bool("is_keyframe", true, "Is IDR frame"),
            AttrDef::Int("pts", false, "Presentation timestamp"),
            AttrDef::Int("dts", false, "Decode timestamp")
        };
    };
};

REGISTER_FRAME_TYPE(H264Frame)
```

---

# F3: DetectionResultFrame Metadata

## File to Create
```
base/include/core/frame_types/DetectionResultFrame.h
```

## Metadata to Add

```cpp
#pragma once
#include "core/Metadata.h"

namespace apra {

class DetectionResultFrame {
public:
    struct Metadata {
        static constexpr std::string_view name = "DetectionResultFrame";
        static constexpr std::string_view parent = "MetadataFrame";
        static constexpr std::string_view description = 
            "Object detection results containing bounding boxes, "
            "class labels, and confidence scores.";
        
        static constexpr std::array<std::string_view, 3> tags = {
            "metadata", "detection", "bounding_boxes"
        };
        
        static constexpr std::array attributes = {
            AttrDef::Int("num_detections", true, "Number of detections"),
            AttrDef::Int("source_width", true, "Original frame width"),
            AttrDef::Int("source_height", true, "Original frame height")
            // Actual bounding boxes are in the frame data, not metadata
        };
    };
};

REGISTER_FRAME_TYPE(DetectionResultFrame)

} // namespace apra
```

---

# F4: QRResultFrame Metadata

## File to Create
```
base/include/core/frame_types/QRResultFrame.h
```

## Metadata to Add

```cpp
#pragma once
#include "core/Metadata.h"

namespace apra {

class QRResultFrame {
public:
    struct Metadata {
        static constexpr std::string_view name = "QRResultFrame";
        static constexpr std::string_view parent = "MetadataFrame";
        static constexpr std::string_view description = 
            "QR code or barcode detection result containing "
            "decoded text and location in frame.";
        
        static constexpr std::array<std::string_view, 3> tags = {
            "metadata", "qr", "barcode"
        };
        
        static constexpr std::array attributes = {
            AttrDef::String("decoded_text", true, "Decoded content"),
            AttrDef::Enum("format", {"QR", "EAN13", "CODE128", "DATAMATRIX"}, true,
                "Barcode format detected")
        };
    };
};

REGISTER_FRAME_TYPE(QRResultFrame)

} // namespace apra
```

---

## Hierarchy Setup

Also need to register parent types if not already registered:

```cpp
// Base types (may already exist conceptually)
class Frame {
public:
    struct Metadata {
        static constexpr std::string_view name = "Frame";
        static constexpr std::string_view parent = "";  // Root
        static constexpr std::string_view description = "Base frame type";
        static constexpr std::array<std::string_view, 0> tags = {};
    };
};

class VideoFrame {
public:
    struct Metadata {
        static constexpr std::string_view name = "VideoFrame";
        static constexpr std::string_view parent = "Frame";
        static constexpr std::string_view description = "Video frame base type";
        static constexpr std::array<std::string_view, 1> tags = {"video"};
    };
};

class EncodedVideoFrame {
public:
    struct Metadata {
        static constexpr std::string_view name = "EncodedVideoFrame";
        static constexpr std::string_view parent = "VideoFrame";
        static constexpr std::string_view description = "Encoded video frame";
        static constexpr std::array<std::string_view, 2> tags = {"video", "encoded"};
    };
};

class MetadataFrame {
public:
    struct Metadata {
        static constexpr std::string_view name = "MetadataFrame";
        static constexpr std::string_view parent = "Frame";
        static constexpr std::string_view description = "Metadata-only frame";
        static constexpr std::array<std::string_view, 1> tags = {"metadata"};
    };
};
```

---

## Testing Strategy

```cpp
// In frame_type_registry_tests.cpp

TEST(FrameTypeRegistry, H264FrameHierarchy) {
    auto& registry = FrameTypeRegistry::instance();
    
    ASSERT_TRUE(registry.hasFrameType("H264Frame"));
    EXPECT_EQ(registry.getParent("H264Frame"), "EncodedVideoFrame");
    EXPECT_TRUE(registry.isSubtype("H264Frame", "VideoFrame"));
    EXPECT_TRUE(registry.isSubtype("H264Frame", "Frame"));
    EXPECT_FALSE(registry.isSubtype("H264Frame", "MetadataFrame"));
}

TEST(FrameTypeRegistry, TagQueries) {
    auto& registry = FrameTypeRegistry::instance();
    
    auto videoTypes = registry.getFrameTypesByTag("video");
    EXPECT_TRUE(std::find(videoTypes.begin(), videoTypes.end(), "H264Frame") 
        != videoTypes.end());
    EXPECT_TRUE(std::find(videoTypes.begin(), videoTypes.end(), "RawImagePlanar") 
        != videoTypes.end());
}

TEST(FrameTypeRegistry, PinCompatibility) {
    auto& registry = FrameTypeRegistry::instance();
    
    // H264Frame can connect to pin that accepts EncodedVideoFrame
    EXPECT_TRUE(registry.isCompatible("H264Frame", "EncodedVideoFrame"));
    
    // RawImagePlanar cannot connect to pin that expects H264Frame
    EXPECT_FALSE(registry.isCompatible("RawImagePlanar", "H264Frame"));
}
```

---

## Definition of Done (for each frame type)
- [ ] Metadata struct added
- [ ] REGISTER_FRAME_TYPE in appropriate file
- [ ] Parent type chain is complete
- [ ] Tags are correct
- [ ] Unit test verifies registration
- [ ] Hierarchy queries work
- [ ] Code reviewed and merged
