# A3: FrameType Registry

**Sprint:** 1 (Week 1-2)  
**Priority:** P1 - High  
**Effort:** 3 days  
**Depends On:** A1 (Core Metadata Types)  
**Blocks:** C4 (Validator Connection Checks), All frame type metadata tasks  

## Description

Create the registry for frame types with their hierarchy, attributes, and tags. This enables the validator and LLM to understand frame type compatibility (e.g., "H264Frame is-a EncodedVideoFrame is-a VideoFrame").

Runs **parallel to A2** after A1 completes.

## Acceptance Criteria

### Unit Tests
- [ ] `REGISTER_FRAME_TYPE(FrameClass)` macro compiles and registers
- [ ] `FrameTypeRegistry::instance()` returns singleton
- [ ] `getFrameType("H264Frame")` returns FrameTypeInfo or nullptr
- [ ] `getAllFrameTypes()` returns all registered type names
- [ ] `getFrameTypesByTag("video")` returns types with that tag
- [ ] `isSubtype("H264Frame", "VideoFrame")` returns true
- [ ] `isSubtype("H264Frame", "AudioFrame")` returns false
- [ ] `getSubtypes("VideoFrame")` returns all video frame types
- [ ] `getParent("H264Frame")` returns "EncodedVideoFrame"
- [ ] `isCompatible("H264Frame", "EncodedVideoFrame")` returns true (subtype check)
- [ ] `toJson()` exports hierarchy and tags

### Behavioral (Given/When/Then)

**Scenario: Register frame type with parent**
```
Given H264Frame with Metadata::parent = "EncodedVideoFrame"
When REGISTER_FRAME_TYPE(H264Frame) executes
Then getFrameType("H264Frame") is not null
And getParent("H264Frame") == "EncodedVideoFrame"
```

**Scenario: Frame type hierarchy query**
```
Given hierarchy: Frame -> VideoFrame -> EncodedVideoFrame -> H264Frame
When I call isSubtype("H264Frame", "Frame")
Then result is true (transitive)
```

**Scenario: Query by media tag**
```
Given H264Frame with tags = {"video", "encoded", "h264"}
And AudioPCMFrame with tags = {"audio", "raw", "pcm"}
When I call getFrameTypesByTag("video")
Then result contains "H264Frame"
And result does NOT contain "AudioPCMFrame"
```

**Scenario: Pin compatibility check**
```
Given a pin that accepts {"EncodedVideoFrame"}
And a module that outputs "H264Frame"
When validator checks compatibility
Then isCompatible("H264Frame", "EncodedVideoFrame") returns true
```

### Requirements
- Support multiple inheritance levels (Frame → VideoFrame → RawImage → RawImagePlanar)
- Thread-safe singleton
- Tags are separate from hierarchy (both used for queries)
- Must handle missing parent gracefully (log warning, treat as root)
- Export to JSON for LLM context

## Implementation Notes for Claude Code Agents

### File Locations
```
base/include/core/FrameTypeRegistry.h
base/src/core/FrameTypeRegistry.cpp
```

### Key Structures

```cpp
struct FrameTypeInfo {
    std::string name;
    std::string parent;         // Empty for root types
    std::string description;
    std::vector<std::string> tags;
    
    struct AttrInfo {
        std::string name;
        std::string type;       // "int", "float", "bool", "string", "enum"
        bool required;
        std::vector<std::string> enum_values;
        std::string description;
    };
    std::vector<AttrInfo> attributes;
};

class FrameTypeRegistry {
public:
    static FrameTypeRegistry& instance();
    
    // Registration
    void registerFrameType(FrameTypeInfo info);
    
    // Basic queries
    bool hasFrameType(const std::string& name) const;
    const FrameTypeInfo* getFrameType(const std::string& name) const;
    std::vector<std::string> getAllFrameTypes() const;
    
    // Tag queries
    std::vector<std::string> getFrameTypesByTag(const std::string& tag) const;
    
    // Hierarchy queries
    std::string getParent(const std::string& name) const;
    std::vector<std::string> getSubtypes(const std::string& name) const;
    std::vector<std::string> getAncestors(const std::string& name) const;
    
    // Compatibility (for validator)
    bool isSubtype(const std::string& child, const std::string& parent) const;
    bool isCompatible(const std::string& outputType, const std::string& inputType) const;
    
    // Export
    std::string toJson() const;
    std::string toMarkdown() const;  // Hierarchy diagram
    
private:
    FrameTypeRegistry() = default;
    std::map<std::string, FrameTypeInfo> types_;
    mutable std::mutex mutex_;
    
    // Cache for hierarchy queries
    mutable std::map<std::string, std::vector<std::string>> ancestorCache_;
};
```

### REGISTER_FRAME_TYPE Macro

```cpp
#define REGISTER_FRAME_TYPE(FrameClass) \
    namespace { \
        static bool _registered_ft_##FrameClass = []() { \
            FrameTypeInfo info; \
            info.name = std::string(FrameClass::Metadata::name); \
            info.parent = std::string(FrameClass::Metadata::parent); \
            info.description = std::string(FrameClass::Metadata::description); \
            for (const auto& tag : FrameClass::Metadata::tags) { \
                info.tags.push_back(std::string(tag)); \
            } \
            /* Copy attributes... */ \
            FrameTypeRegistry::instance().registerFrameType(std::move(info)); \
            return true; \
        }(); \
    }
```

### Hierarchy Implementation

```cpp
bool FrameTypeRegistry::isSubtype(const std::string& child, const std::string& parent) const {
    if (child == parent) return true;
    
    std::string current = child;
    while (!current.empty()) {
        auto* info = getFrameType(current);
        if (!info) break;
        
        current = info->parent;
        if (current == parent) return true;
    }
    return false;
}

// Compatibility: output is subtype of input (or exact match)
bool FrameTypeRegistry::isCompatible(const std::string& output, const std::string& input) const {
    return isSubtype(output, input);
}
```

### Expected Frame Type Hierarchy (for reference)
```
Frame (root)
├── VideoFrame
│   ├── RawImagePlanar      tags: [video, raw, planar, yuv]
│   ├── RawImagePacked      tags: [video, raw, packed, rgb]
│   └── EncodedVideoFrame
│       ├── H264Frame       tags: [video, encoded, h264]
│       ├── H265Frame       tags: [video, encoded, h265]
│       └── JPEGFrame       tags: [video, encoded, jpeg]
├── AudioFrame
│   ├── AudioPCMFrame       tags: [audio, raw, pcm]
│   └── EncodedAudioFrame
│       └── AACFrame        tags: [audio, encoded, aac]
├── MetadataFrame
│   ├── DetectionResultFrame  tags: [metadata, detection]
│   ├── MotionEventFrame      tags: [metadata, motion]
│   └── QRResultFrame         tags: [metadata, qr]
└── CommandFrame
    ├── EOSFrame            tags: [command, eos]
    └── FlushFrame          tags: [command, flush]
```

### Test File Location
```
base/test/frame_type_registry_tests.cpp
```

---

## Definition of Done
- [ ] REGISTER_FRAME_TYPE macro works
- [ ] Hierarchy queries work correctly (isSubtype, getAncestors)
- [ ] Tag queries return correct results
- [ ] isCompatible works for validator use case
- [ ] toJson() exports complete hierarchy
- [ ] Unit tests pass
- [ ] Code reviewed and merged
