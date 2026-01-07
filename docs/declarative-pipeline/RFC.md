# RFC: Declarative Pipeline Construction for ApraPipes

**Version:** 4.0  
**Status:** Draft  
**Author:** Akhil / Apra Labs  

### Changelog
| Version | Date | Changes |
|---------|------|---------|
| 4.0 | Dec 2024 | Added **Tags system**: Module tags (capability, platform, vendor) + FrameType tags (media, encoding, format). Tags on FrameTypes, NOT on PinDefs. |
| 3.0 | Dec 2024 | Initial RFC with C++ as single source of truth, Static/Dynamic properties, Controller modules |

---

## Elevator Pitch

**Today:** Building an ApraPipes pipeline requires writing C++ code—instantiating modules, setting properties, connecting pins, handling errors. This is powerful but creates friction: every new pipeline needs a developer, every change needs a recompile, and onboarding new users means teaching them C++.

**Tomorrow:** Describe your pipeline in a simple text file:

```toml
[modules.camera]
type = "RTSPSourceModule"
    [modules.camera.props]
    url = "rtsp://camera.local/stream"

[modules.motion]
type = "MotionDetectorModule"
    [modules.motion.props]
    sensitivity = 0.7

[[connections]]
from = "camera.output"
to = "motion.input"
```

Or better yet, just tell an LLM what you want:

> *"Monitor my RTSP camera for motion, save 10-second clips when detected, and let me adjust sensitivity via REST API"*

...and get a validated, runnable pipeline.

**The Vision:**
- **Non-developers** can create and modify pipelines
- **LLMs** can generate pipelines from natural language
- **Validation** catches errors before runtime
- **Runtime control** via Controller modules (REST API, schedulers, UI)
- **Zero drift** between code and documentation—C++ is the single source of truth

---

## Executive Summary

This RFC proposes a declarative pipeline definition system for ApraPipes with:

1. **C++ as Single Source of Truth** — All metadata (modules, pins, properties, frame types) defined in C++ headers
2. **Clean Architecture Separation** — Core (Factory + Validator) vs Frontends (TOML, YAML, etc.)
3. **Static vs Dynamic Properties** — Declarative distinction for thread-safe runtime modification
4. **Tags for Modules & FrameTypes** — Multi-dimensional categorization for LLM queries, validation, and docs
5. **Controller Modules** — First-class support for runtime pipeline control
6. **Automated Versioning** — Build system detects schema changes and increments versions
7. **LLM-Ready** — Generated schema enables AI-driven pipeline creation

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core C++ Infrastructure](#2-core-c-infrastructure)
3. [Property System (Static vs Dynamic)](#3-property-system-static-vs-dynamic)
4. [Controller Modules](#4-controller-modules)
5. [Pipeline Validator](#5-pipeline-validator)
6. [Module Factory](#6-module-factory)
7. [Frontend Parsers](#7-frontend-parsers)
8. [Build System & Versioning](#8-build-system--versioning)
9. [LLM Integration](#9-llm-integration)
10. [Implementation Phases](#10-implementation-phases)

---

## 1. Architecture Overview

### Design Principles

1. **C++ is the source of truth** — Metadata lives in headers, registries are build artifacts
2. **Frontends are interchangeable** — TOML, YAML, JSON, or future formats all produce the same IR
3. **Validation is comprehensive** — Offline tooling does deep validation; runtime is optimistic
4. **Properties are categorized** — Static (construction-time) vs Dynamic (runtime-safe)
5. **Controllers are first-class** — Modules that control other modules are part of the framework
6. **Tags enable discovery** — Modules have capability/platform tags; FrameTypes have media/format tags

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER LAYER                                      │
│                                                                              │
│   "Apply motion detection to RTSP stream, save snapshots"                   │
│                              │                                               │
│                              ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                         LLM                                           │  │
│   │  (Uses generated schema as context, outputs TOML)                     │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                    pipeline.toml                                      │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────────────┐
│                         FRONTEND LAYER                                       │
│                              │                                               │
│   ┌─────────┐  ┌─────────┐  ▼  ┌─────────┐  ┌───────────────────────────┐   │
│   │  TOML   │  │  YAML   │  │  │  JSON   │  │  Future: Visual Editor    │   │
│   │ Parser  │  │ Parser  │  │  │ Parser  │  │  gRPC, REST API, etc.     │   │
│   └────┬────┘  └────┬────┘  │  └────┬────┘  └─────────────┬─────────────┘   │
│        │            │       │       │                     │                  │
│        └────────────┴───────┴───────┴─────────────────────┘                  │
│                              │                                               │
│                              ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │              PipelineDescription (C++ Intermediate Repr)              │  │
│   │  - Module instances with properties                                   │  │
│   │  - Connection graph                                                   │  │
│   │  - Pipeline settings                                                  │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────────────┐
│                           CORE LAYER (C++)                                   │
│                              │                                               │
│                              ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                     PipelineValidator                                 │  │
│   │  - Module existence                                                   │  │
│   │  - Property validation (types, ranges, regex)                         │  │
│   │  - Pin compatibility (direct + conversions)                           │  │
│   │  - Graph validation (DAG, no dangling pins)                          │  │
│   │  - Schema version warnings                                            │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                      ModuleFactory                                    │  │
│   │  - Instantiate modules from registry                                  │  │
│   │  - Apply static properties                                            │  │
│   │  - Connect pins                                                       │  │
│   │  - Wire controller modules                                            │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                    Running Pipeline                                   │  │
│   │  - Controller modules can modify dynamic properties                   │  │
│   │  - Events flow through the graph                                      │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                        REGISTRIES                                     │  │
│   │  ModuleRegistry │ FrameTypeRegistry │ CompatibilityRegistry           │  │
│   │  (Populated at static init from Metadata structs)                     │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                    BUILD TIME │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BUILD ARTIFACTS                                      │
│                                                                              │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│   │ schema.json    │  │ modules.md     │  │ version.json   │                │
│   │ (LLM context)  │  │ (Documentation)│  │ (Auto-version) │                │
│   └────────────────┘  └────────────────┘  └────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core C++ Infrastructure

### Intermediate Representation

All frontends parse into this common C++ structure:

```cpp
// ============================================================
// File: core/PipelineDescription.h
// The intermediate representation that all parsers produce
// ============================================================

#pragma once
#include <string>
#include <vector>
#include <map>
#include <variant>
#include <optional>

namespace apra {

using PropertyValue = std::variant<
    int64_t,
    double,
    bool,
    std::string,
    std::vector<int64_t>,
    std::vector<double>,
    std::vector<std::string>
>;

struct ModuleInstance {
    std::string instance_id;      // User-defined unique ID
    std::string module_type;      // Registered module type name
    std::map<std::string, PropertyValue> properties;
};

struct Connection {
    std::string from_module;      // Source module instance_id
    std::string from_pin;         // Source output pin name
    std::string to_module;        // Destination module instance_id
    std::string to_pin;           // Destination input pin name
};

struct PipelineSettings {
    std::string name;
    std::string version = "1.0";
    std::string description;
    
    int queue_size = 10;
    std::string on_error = "restart_module";  // "stop_pipeline" | "skip_frame"
    bool auto_start = false;
};

struct PipelineDescription {
    PipelineSettings settings;
    std::vector<ModuleInstance> modules;
    std::vector<Connection> connections;
    
    // Source tracking for error messages
    std::string source_format;    // "toml", "yaml", "json", etc.
    std::string source_path;      // File path or "<inline>"
};

} // namespace apra
```

### Metadata Types

```cpp
// ============================================================
// File: core/Metadata.h
// Core metadata types - defined once, used everywhere
// ============================================================

#pragma once
#include <string_view>
#include <array>
#include <optional>
#include <initializer_list>

namespace apra {

// ============================================================
// Pin Definition
// ============================================================
struct PinDef {
    std::string_view name;
    std::initializer_list<std::string_view> frame_types;
    bool required = true;
    std::string_view description = "";
};

// ============================================================
// Property Definition with Static/Dynamic distinction
// ============================================================
struct PropDef {
    enum class Type { Int, Float, Bool, String, Enum };
    enum class Mutability { 
        Static,   // Set at construction, cannot change
        Dynamic   // Can be modified at runtime via Controller
    };
    
    std::string_view name;
    Type type;
    Mutability mutability = Mutability::Static;
    
    // Default value (as string for uniform handling)
    std::string_view default_value;
    
    // Validation constraints (evaluated by offline tooling)
    std::string_view min_value = "";      // For Int/Float
    std::string_view max_value = "";      // For Int/Float
    std::string_view regex_pattern = "";  // For String
    std::initializer_list<std::string_view> enum_values = {};
    
    // Documentation
    std::string_view description = "";
    std::string_view unit = "";           // e.g., "ms", "percent", "pixels"
    
    // ========================================================
    // Factory methods for clean declaration syntax
    // ========================================================
    
    static constexpr PropDef Int(
        std::string_view name,
        int default_val,
        int min_val,
        int max_val,
        std::string_view desc = "",
        Mutability mut = Mutability::Static
    );
    
    static constexpr PropDef Float(
        std::string_view name,
        double default_val,
        double min_val,
        double max_val,
        std::string_view desc = "",
        Mutability mut = Mutability::Static
    );
    
    static constexpr PropDef Bool(
        std::string_view name,
        bool default_val,
        std::string_view desc = "",
        Mutability mut = Mutability::Static
    );
    
    static constexpr PropDef String(
        std::string_view name,
        std::string_view default_val,
        std::string_view desc = "",
        std::string_view regex = "",
        Mutability mut = Mutability::Static
    );
    
    static constexpr PropDef Enum(
        std::string_view name,
        std::string_view default_val,
        std::initializer_list<std::string_view> values,
        std::string_view desc = "",
        Mutability mut = Mutability::Static
    );
    
    // Convenience: Dynamic variants
    static constexpr PropDef DynamicInt(
        std::string_view name, int default_val, int min_val, int max_val,
        std::string_view desc = ""
    ) {
        return Int(name, default_val, min_val, max_val, desc, Mutability::Dynamic);
    }
    
    static constexpr PropDef DynamicFloat(
        std::string_view name, double default_val, double min_val, double max_val,
        std::string_view desc = ""
    ) {
        return Float(name, default_val, min_val, max_val, desc, Mutability::Dynamic);
    }
    
    static constexpr PropDef DynamicBool(
        std::string_view name, bool default_val, std::string_view desc = ""
    ) {
        return Bool(name, default_val, desc, Mutability::Dynamic);
    }
};

// ============================================================
// Frame Type Attribute Definition
// ============================================================
struct AttrDef {
    enum class Type { Int, Int64, Float, Bool, String, Enum, IntArray };
    
    std::string_view name;
    Type type;
    bool required;
    std::initializer_list<std::string_view> enum_values = {};
    std::string_view description = "";
    
    static constexpr AttrDef Int(std::string_view name, bool req = true, 
                                  std::string_view desc = "");
    static constexpr AttrDef Float(std::string_view name, bool req = true,
                                    std::string_view desc = "");
    static constexpr AttrDef Bool(std::string_view name, bool req = true,
                                   std::string_view desc = "");
    static constexpr AttrDef String(std::string_view name, bool req = true,
                                     std::string_view desc = "");
    static constexpr AttrDef Enum(std::string_view name,
                                   std::initializer_list<std::string_view> values,
                                   bool req = true, std::string_view desc = "");
};

// ============================================================
// Module Category (Primary - single value)
// ============================================================
enum class ModuleCategory {
    Source,       // Produces frames (RTSP, file, camera)
    Sink,         // Consumes frames (file writer, display)
    Transform,    // Transforms frames (decoder, encoder, filter)
    Analytics,    // Analyzes frames (motion detection, object detection)
    Controller,   // Controls other modules (see Section 4)
    Utility       // Helper modules (queue, tee, mux)
};

// ============================================================
// Tags (Secondary - multiple values, flexible)
// ============================================================
// Tags provide multi-dimensional categorization for:
// - LLM queries: "find a decoder that outputs video"
// - Validation: "this module requires CUDA"
// - Documentation: auto-generate "Encoders" page
// - Future UI: filter/search modules
//
// Module Tag Dimensions:
//   Capability:  encoder, decoder, detector, tracker, filter, reader, writer
//   Platform:    cuda_required, cuda_optional, jetson_only, cpu_only, arm64_only
//   Vendor:      nvidia, intel, v4l2, opencv, ffmpeg
//   Codec:       h264, h265, jpeg, mp4, rtsp
//
// FrameType Tag Dimensions:
//   Media:       video, audio, image, text
//   Encoding:    raw, encoded, compressed
//   Format:      h264, h265, jpeg, pcm, nv12, rgb, yuv
//   Semantics:   metadata, detection, motion, transcript

} // namespace apra
```

---

## 3. Property System (Static vs Dynamic)

### Rationale

ApraPipes has an existing dynamic properties framework. Properties fall into two categories:

| Category | When Set | Thread Safety | Examples |
|----------|----------|---------------|----------|
| **Static** | Construction only | N/A | `device_id`, `url`, `codec` |
| **Dynamic** | Anytime via Controller | Must be thread-safe | `sensitivity`, `bitrate`, `enabled` |

The Metadata must declare which is which to:
1. Generate correct runtime APIs
2. Prevent race conditions
3. Document what can be changed live

### Module Example with Mixed Properties

```cpp
// ============================================================
// File: modules/MotionDetectorModule.h
// ============================================================

#pragma once
#include "core/Module.h"
#include "core/Metadata.h"

namespace apra {

struct MotionDetectorModuleProps : public ModuleProps {
    // Static properties (set at construction)
    int frame_buffer_size = 5;
    
    // Dynamic properties (can change at runtime)
    float sensitivity = 0.5f;
    float min_area_percent = 1.0f;
    bool enabled = true;
};

class MotionDetectorModule : public Module {
public:
    struct Metadata {
        static constexpr std::string_view name = "MotionDetectorModule";
        static constexpr ModuleCategory category = ModuleCategory::Analytics;
        static constexpr std::string_view description = 
            "Detects motion in video frames using background subtraction. "
            "Emits MotionEventFrame on motion start, during, and end.";
        
        static constexpr std::string_view version = "1.0";
        
        // Tags for LLM queries and filtering
        static constexpr std::array tags = {
            "detector",        // Capability
            "motion",          // What it detects
            "opencv",          // Implementation
            "cpu_only"         // Platform requirement
        };
        
        static constexpr std::array inputs = {
            PinDef{"input", {"RawImagePlanar", "RawImagePacked"}, true,
                   "Video frames to analyze"}
        };
        
        static constexpr std::array outputs = {
            PinDef{"motion_frames", {"RawImagePlanar", "RawImagePacked"}, true,
                   "Frames where motion was detected (passthrough with metadata)"},
            PinDef{"events", {"MotionEventFrame"}, true,
                   "Motion event notifications (START, ONGOING, END)"}
        };
        
        static constexpr std::array properties = {
            // STATIC: Cannot change after construction
            PropDef::Int("frame_buffer_size", 5, 2, 30,
                "Number of frames for background model"),
            
            // DYNAMIC: Can be adjusted at runtime via Controller
            PropDef::DynamicFloat("sensitivity", 0.5f, 0.0f, 1.0f,
                "Motion sensitivity (0=low, 1=high)"),
            
            PropDef::DynamicFloat("min_area_percent", 1.0f, 0.1f, 50.0f,
                "Minimum motion area as percentage of frame"),
            
            PropDef::DynamicBool("enabled", true,
                "Enable/disable motion detection (frames pass through when disabled)")
        };
    };
    
    // Dynamic property interface (generated or manual)
    bool setDynamicProperty(std::string_view name, const PropertyValue& value) override;
    PropertyValue getDynamicProperty(std::string_view name) const override;
    
    // ... rest of implementation
};

REGISTER_MODULE(MotionDetectorModule, MotionDetectorModuleProps)

} // namespace apra
```

### Generated Schema (Build Artifact)

The build system generates this from the Metadata:

```json
{
  "modules": {
    "MotionDetectorModule": {
      "category": "analytics",
      "tags": ["detector", "motion", "opencv", "cpu_only"],
      "description": "Detects motion in video frames...",
      "version": "1.0",
      "properties": {
        "frame_buffer_size": {
          "type": "int",
          "mutability": "static",
          "default": 5,
          "min": 2,
          "max": 30,
          "description": "Number of frames for background model"
        },
        "sensitivity": {
          "type": "float",
          "mutability": "dynamic",
          "default": 0.5,
          "min": 0.0,
          "max": 1.0,
          "description": "Motion sensitivity (0=low, 1=high)"
        }
      }
    }
  }
}
```

### FrameType Tags

Frame types carry tags that describe their media characteristics. This enables LLM queries like "find modules that output video" by checking output pin frame types for the `video` tag.

```cpp
// ============================================================
// Frame types with tags
// ============================================================

class H264Frame : public EncodedVideoFrame {
    struct Metadata {
        static constexpr std::string_view name = "H264Frame";
        static constexpr std::string_view parent = "EncodedVideoFrame";
        static constexpr std::string_view description = 
            "H.264/AVC encoded video frame (NAL units)";
        
        // Tags describe what this frame type represents
        static constexpr std::array tags = {
            "video",      // Media type
            "encoded",    // Encoding state
            "h264"        // Specific codec
        };
        
        static constexpr std::array attributes = {
            AttrDef::Int("width", true, "Frame width in pixels"),
            AttrDef::Int("height", true, "Frame height in pixels"),
            AttrDef::Bool("is_keyframe", true, "Whether this is an IDR frame")
        };
    };
};

class RawImagePlanar : public VideoFrame {
    struct Metadata {
        static constexpr std::string_view name = "RawImagePlanar";
        static constexpr std::string_view parent = "VideoFrame";
        static constexpr std::string_view description = 
            "Raw planar image (NV12, I420, YUV444)";
        
        static constexpr std::array tags = {
            "video",      // Media type
            "raw",        // Encoding state
            "planar",     // Memory layout
            "yuv"         // Color space family
        };
        
        static constexpr std::array attributes = {
            AttrDef::Int("width", true),
            AttrDef::Int("height", true),
            AttrDef::Enum("format", {"NV12", "I420", "YUV444"}, true)
        };
    };
};

class DetectionResultFrame : public MetadataFrame {
    struct Metadata {
        static constexpr std::string_view name = "DetectionResultFrame";
        static constexpr std::string_view parent = "MetadataFrame";
        static constexpr std::string_view description = 
            "Object detection results with bounding boxes and labels";
        
        static constexpr std::array tags = {
            "metadata",        // Media type (not video/audio)
            "detection",       // Semantic meaning
            "bounding_boxes"   // Data structure
        };
    };
};

class AudioPCMFrame : public AudioFrame {
    struct Metadata {
        static constexpr std::string_view name = "AudioPCMFrame";
        static constexpr std::string_view parent = "AudioFrame";
        
        static constexpr std::array tags = {
            "audio",      // Media type
            "raw",        // Encoding state
            "pcm"         // Format
        };
    };
};
```

### LLM Query Flow with Tags

```
User: "I need a module that decodes video"

LLM reasoning:
1. Find modules with tag "decoder"           → H264DecoderNvCodec, JPEGDecoder, ...
2. Check their output pins
3. Look up FrameType for each output pin
4. Filter where FrameType has tag "video"    → H264DecoderNvCodec ✓

Result: H264DecoderNvCodec
```

### Tag Taxonomy Reference

| Level | Dimension | Example Tags |
|-------|-----------|--------------|
| **Module** | Capability | `encoder`, `decoder`, `detector`, `tracker`, `filter`, `reader`, `writer`, `muxer`, `demuxer` |
| **Module** | Platform | `cuda_required`, `cuda_optional`, `jetson_only`, `cpu_only`, `arm64_only`, `x86_64_only` |
| **Module** | Vendor | `nvidia`, `intel`, `v4l2`, `opencv`, `ffmpeg` |
| **Module** | Codec | `h264`, `h265`, `jpeg`, `mp4`, `rtsp`, `webrtc` |
| **FrameType** | Media | `video`, `audio`, `image`, `text` |
| **FrameType** | Encoding | `raw`, `encoded`, `compressed` |
| **FrameType** | Format | `h264`, `h265`, `jpeg`, `pcm`, `nv12`, `rgb`, `yuv`, `bgr` |
| **FrameType** | Layout | `planar`, `packed`, `interleaved` |
| **FrameType** | Semantics | `metadata`, `detection`, `motion`, `transcript`, `command` |
```

---

## 4. Controller Modules

### Concept

Controller modules are special modules that:
1. Receive commands from external sources (REST API, WebSocket, UI, timers)
2. Modify dynamic properties of other modules in the pipeline
3. Can start/stop/pause other modules
4. Are declared in the pipeline definition like any other module

### Controller Module Interface

```cpp
// ============================================================
// File: core/ControllerModule.h
// ============================================================

#pragma once
#include "Module.h"
#include <functional>

namespace apra {

// Forward declaration
class Pipeline;

class ControllerModule : public Module {
public:
    // Controllers get a reference to the pipeline they control
    void setPipeline(Pipeline* pipeline) { pipeline_ = pipeline; }
    
protected:
    // API for subclasses to control other modules
    bool setModuleProperty(
        const std::string& module_id,
        const std::string& property,
        const PropertyValue& value
    );
    
    PropertyValue getModuleProperty(
        const std::string& module_id,
        const std::string& property
    ) const;
    
    bool enableModule(const std::string& module_id);
    bool disableModule(const std::string& module_id);
    
    // Send command to a module
    bool sendCommand(
        const std::string& module_id,
        const std::string& command,
        const std::map<std::string, PropertyValue>& params = {}
    );
    
private:
    Pipeline* pipeline_ = nullptr;
};

} // namespace apra
```

### Example: REST API Controller

```cpp
// ============================================================
// File: modules/RestApiController.h
// ============================================================

#pragma once
#include "core/ControllerModule.h"

namespace apra {

struct RestApiControllerProps : public ModuleProps {
    int port = 8080;
    std::string bind_address = "0.0.0.0";
    bool enable_cors = true;
};

class RestApiController : public ControllerModule {
public:
    struct Metadata {
        static constexpr std::string_view name = "RestApiController";
        static constexpr ModuleCategory category = ModuleCategory::Controller;
        static constexpr std::string_view description = 
            "Exposes pipeline control via REST API. "
            "Allows external systems to modify dynamic properties and send commands.";
        
        // Controllers typically have no data pins
        static constexpr std::array<PinDef, 0> inputs = {};
        static constexpr std::array<PinDef, 0> outputs = {};
        
        static constexpr std::array properties = {
            PropDef::Int("port", 8080, 1024, 65535,
                "HTTP port to listen on"),
            PropDef::String("bind_address", "0.0.0.0",
                "Network interface to bind to",
                R"(^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$)"),  // IPv4 regex
            PropDef::Bool("enable_cors", true,
                "Enable CORS headers for browser access")
        };
        
        // Controller-specific: which modules this controller can target
        // (empty = can target any module in the pipeline)
        static constexpr std::array<std::string_view, 0> target_modules = {};
    };
    
    // ... HTTP server implementation
};

REGISTER_MODULE(RestApiController, RestApiControllerProps)

} // namespace apra
```

### Example: Timer-Based Controller

```cpp
// ============================================================
// File: modules/SchedulerController.h
// ============================================================

namespace apra {

class SchedulerController : public ControllerModule {
public:
    struct Metadata {
        static constexpr std::string_view name = "SchedulerController";
        static constexpr ModuleCategory category = ModuleCategory::Controller;
        static constexpr std::string_view description = 
            "Executes scheduled actions on pipeline modules. "
            "Useful for time-based recording, periodic snapshots, etc.";
        
        static constexpr std::array<PinDef, 0> inputs = {};
        static constexpr std::array<PinDef, 0> outputs = {};
        
        static constexpr std::array properties = {
            PropDef::String("schedule", "",
                "Cron-like schedule expression",
                R"(^[\d\*\-\,\/\s]+$)"),  // Basic cron regex
            PropDef::String("action", "enable",
                "Action to perform: enable, disable, set_property, command"),
            PropDef::String("target_module", "",
                "Module instance ID to control"),
            PropDef::String("property_name", "",
                "Property to modify (for set_property action)"),
            PropDef::String("property_value", "",
                "Value to set (for set_property action)")
        };
    };
};

} // namespace apra
```

### Pipeline Definition with Controller

```toml
[pipeline]
name = "motion_recording_with_api"
description = "Motion detection pipeline with REST API control"

# Data modules
[modules.rtsp_source]
type = "RTSPSourceModule"
    [modules.rtsp_source.props]
    url = "rtsp://camera.local:554/stream"

[modules.decoder]
type = "H264DecoderNvCodec"
    [modules.decoder.props]
    device_id = 0

[modules.motion]
type = "MotionDetectorModule"
    [modules.motion.props]
    sensitivity = 0.5        # Can be changed via API
    min_area_percent = 2.0   # Can be changed via API
    enabled = true           # Can be toggled via API

[modules.recorder]
type = "FileWriterModule"
    [modules.recorder.props]
    path = "/recordings"

# Controller module - controls the pipeline via REST API
[modules.api]
type = "RestApiController"
    [modules.api.props]
    port = 8080
    bind_address = "0.0.0.0"

# Data connections
[[connections]]
from = "rtsp_source.output"
to = "decoder.input"

[[connections]]
from = "decoder.output"
to = "motion.input"

[[connections]]
from = "motion.motion_frames"
to = "recorder.input"

# Note: Controller modules don't need data connections
# They connect to the pipeline control plane automatically
```

### API Generated from Pipeline

The REST API controller would automatically expose:

```
GET  /api/modules                           # List all modules
GET  /api/modules/motion/properties         # Get motion detector properties
PUT  /api/modules/motion/properties/sensitivity  
     Body: { "value": 0.8 }                 # Set sensitivity (dynamic)

GET  /api/modules/motion/properties/frame_buffer_size
     Response: { "value": 5, "mutability": "static", "error": null }

PUT  /api/modules/motion/properties/frame_buffer_size
     Response: { "error": "Property 'frame_buffer_size' is static and cannot be modified at runtime" }
```

---

## 5. Pipeline Validator

### Validation Phases

```cpp
// ============================================================
// File: core/PipelineValidator.h
// ============================================================

#pragma once
#include "PipelineDescription.h"
#include "ModuleRegistry.h"
#include "FrameTypeRegistry.h"
#include <vector>
#include <string>

namespace apra {

struct ValidationIssue {
    enum class Level { 
        Error,    // Pipeline cannot be built
        Warning,  // Pipeline can be built but may have issues
        Info      // Informational (e.g., defaults applied)
    };
    
    Level level;
    std::string code;       // Machine-readable: "UNKNOWN_MODULE", "TYPE_MISMATCH"
    std::string location;   // "modules.decoder.props.device_id"
    std::string message;    // Human-readable description
    
    // For schema evolution warnings
    std::string suggestion = "";  // "Add property 'foo' with value 'bar'"
};

class PipelineValidator {
public:
    struct Result {
        std::vector<ValidationIssue> issues;
        
        bool hasErrors() const;
        bool hasWarnings() const;
        std::vector<ValidationIssue> errors() const;
        std::vector<ValidationIssue> warnings() const;
    };
    
    // Full validation (all phases)
    Result validate(const PipelineDescription& desc) const;
    
    // Individual phases (for tooling)
    Result validateModules(const PipelineDescription& desc) const;
    Result validateProperties(const PipelineDescription& desc) const;
    Result validateConnections(const PipelineDescription& desc) const;
    Result validateGraph(const PipelineDescription& desc) const;
    Result validateSchemaVersion(const PipelineDescription& desc) const;
    
private:
    // Property validation helpers
    bool validateIntRange(int64_t value, const PropDef& prop) const;
    bool validateFloatRange(double value, const PropDef& prop) const;
    bool validateRegex(const std::string& value, const PropDef& prop) const;
    bool validateEnum(const std::string& value, const PropDef& prop) const;
};

} // namespace apra
```

### Validation Rules

#### Phase 1: Module Validation
```
For each module instance:
  ✓ Module type exists in registry
  ✓ Module type matches category constraints (if any)
  ! Warn if module version differs from registry version
```

#### Phase 2: Property Validation
```
For each property in each module:
  ✓ Property name exists in module's Metadata
  ✓ Property type matches (int, float, bool, string, enum)
  ✓ Value is within declared range (min/max)
  ✓ String matches regex pattern (if declared)
  ✓ Enum value is in allowed set
  ! Warn if required property missing (will use default)
  ! Info: "Property 'X' not specified, using default 'Y'"
```

#### Phase 3: Connection Validation
```
For each connection:
  ✓ Source module exists
  ✓ Source pin exists on source module
  ✓ Destination module exists
  ✓ Destination pin exists on destination module
  ✓ Frame types are compatible:
      - Direct match, OR
      - Source type is subtype of input type, OR
      - Conversion path exists (if auto-convert enabled)
  ✓ No duplicate connections to same input pin
  ✓ Required input pins are connected
```

#### Phase 4: Graph Validation
```
✓ Pipeline has at least one source module (Category::Source)
✓ Graph is a DAG (no cycles)
! Warn if any module has unconnected optional outputs
! Warn if any non-sink module has no outgoing connections
```

#### Phase 5: Schema Version Validation
```
For each module instance:
  If module's schema version > version in pipeline file:
    ! Warn about new required properties
    ! Suggest adding properties with defaults
  If module's schema version < version in pipeline file:
    ✓ OK (forward compatibility)
```

### Error Codes

```cpp
namespace apra::ErrorCode {
    // Module errors
    constexpr auto UNKNOWN_MODULE = "E001";
    constexpr auto MODULE_VERSION_MISMATCH = "W001";
    
    // Property errors
    constexpr auto UNKNOWN_PROPERTY = "E010";
    constexpr auto PROPERTY_TYPE_MISMATCH = "E011";
    constexpr auto PROPERTY_OUT_OF_RANGE = "E012";
    constexpr auto PROPERTY_REGEX_MISMATCH = "E013";
    constexpr auto PROPERTY_INVALID_ENUM = "E014";
    constexpr auto PROPERTY_MISSING_REQUIRED = "W010";
    constexpr auto PROPERTY_USING_DEFAULT = "I010";
    
    // Connection errors
    constexpr auto UNKNOWN_SOURCE_MODULE = "E020";
    constexpr auto UNKNOWN_SOURCE_PIN = "E021";
    constexpr auto UNKNOWN_DEST_MODULE = "E022";
    constexpr auto UNKNOWN_DEST_PIN = "E023";
    constexpr auto FRAME_TYPE_INCOMPATIBLE = "E024";
    constexpr auto DUPLICATE_INPUT_CONNECTION = "E025";
    constexpr auto REQUIRED_PIN_UNCONNECTED = "E026";
    
    // Graph errors
    constexpr auto NO_SOURCE_MODULE = "E030";
    constexpr auto CYCLE_DETECTED = "E031";
    constexpr auto ORPHAN_MODULE = "W030";
    
    // Schema errors
    constexpr auto SCHEMA_NEWER_THAN_PIPELINE = "W040";
    constexpr auto NEW_REQUIRED_PROPERTY = "W041";
}
```

---

## 6. Module Factory

### Factory Implementation

```cpp
// ============================================================
// File: core/ModuleFactory.h
// ============================================================

#pragma once
#include "PipelineDescription.h"
#include "Pipeline.h"
#include "ModuleRegistry.h"
#include <memory>

namespace apra {

class ModuleFactory {
public:
    struct Options {
        bool auto_insert_converters = false;
        int max_conversion_cost = 100;
    };
    
    struct BuildResult {
        std::unique_ptr<Pipeline> pipeline;
        std::vector<ValidationIssue> issues;
        
        bool success() const { 
            return pipeline != nullptr; 
        }
    };
    
    explicit ModuleFactory(Options opts = {});
    
    // Build pipeline from validated description
    // (Caller should validate first, but factory will check for critical errors)
    BuildResult build(const PipelineDescription& desc);
    
private:
    Options options_;
    
    // Create module instance with properties applied
    std::unique_ptr<Module> createModule(
        const ModuleInstance& instance,
        std::vector<ValidationIssue>& issues
    );
    
    // Connect modules
    bool connectModules(
        Pipeline& pipeline,
        const std::vector<Connection>& connections,
        std::vector<ValidationIssue>& issues
    );
    
    // Wire controller modules to pipeline
    void wireControllers(Pipeline& pipeline);
};

} // namespace apra
```

### Usage Pattern

```cpp
// Complete workflow from TOML to running pipeline

#include "core/TomlParser.h"
#include "core/PipelineValidator.h"
#include "core/ModuleFactory.h"

int main() {
    // 1. Parse TOML to intermediate representation
    apra::TomlParser parser;
    auto parseResult = parser.parseFile("pipeline.toml");
    
    if (!parseResult.success) {
        std::cerr << "Parse error: " << parseResult.error << "\n";
        return 1;
    }
    
    apra::PipelineDescription desc = parseResult.description;
    
    // 2. Validate (comprehensive offline validation)
    apra::PipelineValidator validator;
    auto validationResult = validator.validate(desc);
    
    // Print all issues
    for (const auto& issue : validationResult.issues) {
        const char* prefix = 
            issue.level == apra::ValidationIssue::Level::Error ? "ERROR" :
            issue.level == apra::ValidationIssue::Level::Warning ? "WARN" : "INFO";
        
        std::cout << "[" << prefix << "] " << issue.code 
                  << " at " << issue.location << ": " 
                  << issue.message << "\n";
        
        if (!issue.suggestion.empty()) {
            std::cout << "  Suggestion: " << issue.suggestion << "\n";
        }
    }
    
    if (validationResult.hasErrors()) {
        std::cerr << "Validation failed with errors\n";
        return 1;
    }
    
    // 3. Build pipeline (optimistic - validation already done)
    apra::ModuleFactory factory;
    auto buildResult = factory.build(desc);
    
    if (!buildResult.success()) {
        std::cerr << "Build failed\n";
        return 1;
    }
    
    // 4. Run
    buildResult.pipeline->start();
    
    // Wait for shutdown signal...
    
    return 0;
}
```

---

## 7. Frontend Parsers

### Parser Interface

All parsers implement this interface and produce the same `PipelineDescription`:

```cpp
// ============================================================
// File: core/PipelineParser.h
// ============================================================

#pragma once
#include "PipelineDescription.h"
#include <string>

namespace apra {

struct ParseResult {
    bool success = false;
    PipelineDescription description;
    std::string error;           // Error message if !success
    int error_line = 0;          // Line number of error
    int error_column = 0;        // Column of error
};

class PipelineParser {
public:
    virtual ~PipelineParser() = default;
    
    virtual ParseResult parseFile(const std::string& filepath) = 0;
    virtual ParseResult parseString(const std::string& content) = 0;
    
    // What format does this parser handle?
    virtual std::string formatName() const = 0;
    virtual std::vector<std::string> fileExtensions() const = 0;
};

// Factory to get parser by format or file extension
class ParserFactory {
public:
    static std::unique_ptr<PipelineParser> createForFormat(const std::string& format);
    static std::unique_ptr<PipelineParser> createForFile(const std::string& filepath);
    
    static void registerParser(
        std::unique_ptr<PipelineParser> parser
    );
};

} // namespace apra
```

### TOML Parser (Primary)

```cpp
// ============================================================
// File: parsers/TomlParser.h
// ============================================================

#pragma once
#include "core/PipelineParser.h"

namespace apra {

class TomlParser : public PipelineParser {
public:
    ParseResult parseFile(const std::string& filepath) override;
    ParseResult parseString(const std::string& content) override;
    
    std::string formatName() const override { return "toml"; }
    std::vector<std::string> fileExtensions() const override { 
        return {".toml"}; 
    }
    
private:
    ParseResult parseToml(const std::string& content, const std::string& source);
};

} // namespace apra
```

### Adding a New Parser

Adding YAML support is just implementing the interface:

```cpp
// ============================================================
// File: parsers/YamlParser.cpp
// ============================================================

#include "YamlParser.h"
#include <yaml-cpp/yaml.h>

namespace apra {

ParseResult YamlParser::parseString(const std::string& content) {
    ParseResult result;
    result.description.source_format = "yaml";
    
    try {
        YAML::Node root = YAML::Load(content);
        
        // Parse pipeline section
        if (root["pipeline"]) {
            auto& settings = result.description.settings;
            settings.name = root["pipeline"]["name"].as<std::string>("");
            // ... etc
        }
        
        // Parse modules section
        if (root["modules"]) {
            for (const auto& kv : root["modules"]) {
                ModuleInstance instance;
                instance.instance_id = kv.first.as<std::string>();
                instance.module_type = kv.second["type"].as<std::string>();
                // ... parse properties
                result.description.modules.push_back(std::move(instance));
            }
        }
        
        // Parse connections
        // ... same pattern
        
        result.success = true;
    } catch (const YAML::Exception& e) {
        result.success = false;
        result.error = e.what();
        result.error_line = e.mark.line;
        result.error_column = e.mark.column;
    }
    
    return result;
}

// Register at static init
static bool _registered = []() {
    ParserFactory::registerParser(std::make_unique<YamlParser>());
    return true;
}();

} // namespace apra
```

---

## 8. Build System & Versioning

### Schema Generation

```cmake
# ============================================================
# CMakeLists.txt
# ============================================================

# Tool that dumps registry to JSON
add_executable(apra_schema_generator
    tools/schema_generator.cpp
)
target_link_libraries(apra_schema_generator aprapipes_all_modules)

# Generate schema at build time
set(SCHEMA_DIR ${CMAKE_BINARY_DIR}/schema)
set(SCHEMA_FILE ${SCHEMA_DIR}/aprapipes_schema.json)

add_custom_command(
    OUTPUT ${SCHEMA_FILE}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SCHEMA_DIR}
    COMMAND apra_schema_generator --output ${SCHEMA_FILE}
    DEPENDS apra_schema_generator
    COMMENT "Generating ApraPipes schema from C++ metadata"
)

add_custom_target(generate_schema ALL DEPENDS ${SCHEMA_FILE})
```

### Automated Version Detection

```cpp
// ============================================================
// File: tools/version_checker.cpp
// Run during CI to detect schema changes
// ============================================================

#include "core/ModuleRegistry.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <old_schema.json> <new_schema.json>\n";
        return 1;
    }
    
    // Load schemas
    json old_schema, new_schema;
    std::ifstream(argv[1]) >> old_schema;
    std::ifstream(argv[2]) >> new_schema;
    
    // Compare modules
    for (const auto& [name, new_module] : new_schema["modules"].items()) {
        if (!old_schema["modules"].contains(name)) {
            std::cout << "NEW MODULE: " << name << "\n";
            continue;
        }
        
        const auto& old_module = old_schema["modules"][name];
        
        // Check for new properties
        for (const auto& [prop_name, prop_def] : new_module["properties"].items()) {
            if (!old_module["properties"].contains(prop_name)) {
                std::cout << "NEW PROPERTY: " << name << "." << prop_name;
                if (prop_def.contains("default")) {
                    std::cout << " (default: " << prop_def["default"] << ")";
                }
                std::cout << "\n";
            }
        }
        
        // Check for removed properties
        for (const auto& [prop_name, prop_def] : old_module["properties"].items()) {
            if (!new_module["properties"].contains(prop_name)) {
                std::cout << "REMOVED PROPERTY: " << name << "." << prop_name << "\n";
            }
        }
        
        // Check for type changes
        // ... etc
    }
    
    return 0;
}
```

### CI Integration

```yaml
# .github/workflows/schema-check.yml

name: Schema Version Check

on: [push, pull_request]

jobs:
  check-schema:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Need history for comparison
      
      - name: Build schema generator
        run: |
          cmake -B build
          cmake --build build --target apra_schema_generator
      
      - name: Generate current schema
        run: ./build/apra_schema_generator --output current_schema.json
      
      - name: Get previous schema
        run: |
          git show HEAD~1:schema/aprapipes_schema.json > previous_schema.json || echo '{}' > previous_schema.json
      
      - name: Compare schemas
        run: |
          ./build/apra_version_checker previous_schema.json current_schema.json > schema_changes.txt
          if [ -s schema_changes.txt ]; then
            echo "## Schema Changes Detected" >> $GITHUB_STEP_SUMMARY
            cat schema_changes.txt >> $GITHUB_STEP_SUMMARY
          fi
```

---

## 9. LLM Integration

### Context Generation

```cpp
// ============================================================
// File: tools/llm_context_generator.cpp
// ============================================================

#include "core/ModuleRegistry.h"
#include "core/FrameTypeRegistry.h"
#include <sstream>

std::string generateLLMContext() {
    std::ostringstream ctx;
    
    ctx << R"(# ApraPipes Pipeline Generation Guide

You are generating TOML pipeline definitions for ApraPipes, a video processing framework.

## Rules
1. Use ONLY module types listed below
2. Use exact property names and valid values
3. All connections must have compatible frame types
4. Controller modules don't need data connections

## Output Format
Generate valid TOML. Wrap in ```toml code blocks.
Explain your design choices after the TOML.

)";

    // Add module reference
    ctx << "## Available Modules\n\n";
    
    for (const auto& name : ModuleRegistry::instance().getAllModules()) {
        const auto* info = ModuleRegistry::instance().getModule(name);
        
        ctx << "### " << info->name << "\n";
        ctx << "Category: " << categoryToString(info->category) << "\n";
        ctx << "Description: " << info->description << "\n\n";
        
        if (!info->inputs.empty()) {
            ctx << "**Inputs:**\n";
            for (const auto& pin : info->inputs) {
                ctx << "- `" << pin.name << "`: accepts ";
                for (size_t i = 0; i < pin.frame_types.size(); ++i) {
                    if (i > 0) ctx << ", ";
                    ctx << pin.frame_types[i];
                }
                ctx << "\n";
            }
        }
        
        if (!info->outputs.empty()) {
            ctx << "**Outputs:**\n";
            for (const auto& pin : info->outputs) {
                ctx << "- `" << pin.name << "`: produces ";
                for (size_t i = 0; i < pin.frame_types.size(); ++i) {
                    if (i > 0) ctx << ", ";
                    ctx << pin.frame_types[i];
                }
                ctx << "\n";
            }
        }
        
        ctx << "**Properties:**\n";
        for (const auto& prop : info->properties) {
            ctx << "- `" << prop.name << "` (" << prop.type << ")";
            ctx << " [" << (prop.mutability == "dynamic" ? "dynamic" : "static") << "]";
            ctx << " default=" << prop.default_value;
            if (!prop.min_value.empty()) {
                ctx << " range=[" << prop.min_value << "," << prop.max_value << "]";
            }
            ctx << "\n";
        }
        
        ctx << "\n---\n\n";
    }
    
    // Add frame type hierarchy
    ctx << "## Frame Type Compatibility\n\n";
    ctx << FrameTypeRegistry::instance().toMarkdown();
    
    // Add example
    ctx << R"(
## Example Pipeline

```toml
[pipeline]
name = "example"

[modules.source]
type = "RTSPSourceModule"
    [modules.source.props]
    url = "rtsp://camera/stream"

[modules.decoder]
type = "H264DecoderNvCodec"
    [modules.decoder.props]
    device_id = 0

[[connections]]
from = "source.output"
to = "decoder.input"
```

)";
    
    return ctx.str();
}
```

### LLM Workflow

```cpp
// ============================================================
// File: core/LLMPipelineGenerator.h
// ============================================================

#pragma once
#include "PipelineDescription.h"
#include "PipelineValidator.h"
#include <functional>
#include <string>

namespace apra {

class LLMPipelineGenerator {
public:
    // User provides their LLM call implementation
    using LLMCallFn = std::function<std::string(
        const std::string& system_prompt,
        const std::string& user_message
    )>;
    
    struct Options {
        int max_iterations = 3;
        bool verbose = false;
    };
    
    struct Result {
        bool success = false;
        std::string toml_output;
        std::string llm_explanation;
        std::vector<ValidationIssue> validation_issues;
        int iterations_used = 0;
    };
    
    LLMPipelineGenerator(LLMCallFn llm_fn, Options opts = {});
    
    Result generate(const std::string& user_request);
    
private:
    LLMCallFn llm_fn_;
    Options options_;
    std::string system_prompt_;
    PipelineValidator validator_;
    
    std::string extractTOML(const std::string& llm_response) const;
    std::string formatValidationFeedback(
        const std::vector<ValidationIssue>& issues
    ) const;
};

} // namespace apra
```

### Usage

```cpp
#include "core/LLMPipelineGenerator.h"
#include <anthropic/client.h>  // Your Claude SDK

int main() {
    // Wrap your LLM client
    anthropic::Client claude(getenv("ANTHROPIC_API_KEY"));
    
    auto llm_fn = [&](const std::string& system, const std::string& user) {
        return claude.messages({
            {"role", "user", "content", user}
        }, {
            {"system", system},
            {"model", "claude-sonnet-4-20250514"},
            {"max_tokens", 4096}
        });
    };
    
    apra::LLMPipelineGenerator generator(llm_fn, {.max_iterations = 3});
    
    auto result = generator.generate(
        "Create a pipeline that monitors an RTSP camera for motion, "
        "records 10-second clips when motion is detected, and exposes "
        "a REST API to adjust motion sensitivity."
    );
    
    if (result.success) {
        std::cout << "Generated pipeline:\n" << result.toml_output << "\n\n";
        std::cout << "Explanation:\n" << result.llm_explanation << "\n";
        
        // The TOML is validated - can proceed to build
    } else {
        std::cout << "Generation failed after " << result.iterations_used 
                  << " iterations\n";
        for (const auto& issue : result.validation_issues) {
            std::cout << "  " << issue.message << "\n";
        }
    }
}
```

---

## 10. Implementation Phases

### Phase 1: Core Infrastructure (MVP) — 6 weeks

**Goal:** Validate end-to-end flow with minimal feature set

- [ ] `Metadata.h` with PinDef, PropDef (including Static/Dynamic)
- [ ] `ModuleRegistry` with registration and queries
- [ ] `FrameTypeRegistry` with hierarchy  
- [ ] `REGISTER_MODULE` macro
- [ ] Add Metadata to 5 core modules:
  - `RTSPSourceModule` (Source)
  - `H264DecoderNvCodec` (Transform)
  - `MotionDetectorModule` (Analytics)
  - `FileWriterModule` (Sink)
  - `RestApiController` (Controller)
- [ ] `PipelineDescription` intermediate representation
- [ ] `TomlParser` (using toml++ or toml11)
- [ ] `PipelineValidator` (Phase 1-3 validation)
- [ ] `ModuleFactory` (basic construction)
- [ ] CLI: `aprapipes validate pipeline.toml`
- [ ] CLI: `aprapipes run pipeline.toml`
- [ ] CMake schema generation

### Phase 2: Complete Validation — 3 weeks

- [ ] Full property validation (ranges, regex, enums)
- [ ] Frame type compatibility checking
- [ ] `CompatibilityRegistry` for conversions
- [ ] Graph validation (DAG, cycles, orphans)
- [ ] Schema version checking with warnings
- [ ] CLI: `aprapipes list-modules`
- [ ] CLI: `aprapipes describe <module>`

### Phase 3: All Modules — 4 weeks

- [ ] Add Metadata to all existing modules
- [ ] Controller module base class and wiring
- [ ] Dynamic property runtime modification
- [ ] Documentation generation (Markdown)
- [ ] JSON Schema generation for IDE support

### Phase 4: LLM Integration — 2 weeks

- [ ] LLM context generator
- [ ] `LLMPipelineGenerator` with iterative refinement
- [ ] CLI: `aprapipes generate "<description>"`
- [ ] Example integration with Claude API

### Phase 5: Tooling & Polish — 2 weeks

- [ ] VS Code extension (syntax highlighting, validation)
- [ ] YAML parser (optional secondary format)
- [ ] Pipeline visualization (DOT/Mermaid export)
- [ ] Performance optimization
- [ ] Integration tests

---

## Appendix A: Complete Module Example

```cpp
// ============================================================
// File: modules/MotionDetectorModule.h
// Complete example with all metadata patterns
// ============================================================

#pragma once
#include "core/Module.h"
#include "core/Metadata.h"

namespace apra {

struct MotionDetectorModuleProps : public ModuleProps {
    // Static properties
    int frame_buffer_size = 5;
    std::string algorithm = "mog2";
    
    // Dynamic properties
    float sensitivity = 0.5f;
    float min_area_percent = 1.0f;
    int cooldown_ms = 2000;
    bool enabled = true;
};

class MotionDetectorModule : public Module {
public:
    // ========================================================
    // METADATA - Single Source of Truth
    // ========================================================
    struct Metadata {
        // Identity
        static constexpr std::string_view name = "MotionDetectorModule";
        static constexpr ModuleCategory category = ModuleCategory::Analytics;
        static constexpr std::string_view version = "1.2";
        static constexpr std::string_view description = 
            "Detects motion in video frames using background subtraction. "
            "Outputs frames where motion was detected and emits motion events. "
            "Supports MOG2 and KNN algorithms.";
        
        // Tags for filtering, LLM queries, and documentation
        static constexpr std::array tags = {
            "detector",        // Capability: what it does
            "motion",          // Specifics: what it detects
            "opencv",          // Vendor: implementation
            "cpu_only"         // Platform: no GPU required
        };
        
        // Pins
        static constexpr std::array inputs = {
            PinDef{
                .name = "input",
                .frame_types = {"RawImagePlanar", "RawImagePacked"},
                .required = true,
                .description = "Video frames to analyze for motion"
            }
        };
        
        static constexpr std::array outputs = {
            PinDef{
                .name = "motion_frames",
                .frame_types = {"RawImagePlanar", "RawImagePacked"},
                .required = true,
                .description = "Frames where motion was detected (passthrough)"
            },
            PinDef{
                .name = "all_frames",
                .frame_types = {"RawImagePlanar", "RawImagePacked"},
                .required = false,
                .description = "All frames with motion metadata attached"
            },
            PinDef{
                .name = "events",
                .frame_types = {"MotionEventFrame"},
                .required = true,
                .description = "Motion events (START, ONGOING, END)"
            }
        };
        
        // Properties
        static constexpr std::array properties = {
            // === STATIC PROPERTIES ===
            PropDef::Int("frame_buffer_size", 5, 2, 30,
                "Number of frames for background model history"),
            
            PropDef::Enum("algorithm", "mog2", {"mog2", "knn"},
                "Background subtraction algorithm"),
            
            // === DYNAMIC PROPERTIES ===
            PropDef::DynamicFloat("sensitivity", 0.5f, 0.0f, 1.0f,
                "Motion detection sensitivity (0=low, 1=high)"),
            
            PropDef::DynamicFloat("min_area_percent", 1.0f, 0.1f, 50.0f,
                "Minimum motion area as percentage of frame"),
            
            PropDef::DynamicInt("cooldown_ms", 2000, 0, 60000,
                "Minimum time between motion END and next START event"),
            
            PropDef::DynamicBool("enabled", true,
                "Enable/disable detection (frames pass through when disabled)")
        };
    };
    
    // ========================================================
    // Implementation
    // ========================================================
    
    explicit MotionDetectorModule(MotionDetectorModuleProps props);
    ~MotionDetectorModule() override;
    
    bool init() override;
    bool process(Frame* frame) override;
    bool term() override;
    
    // Dynamic property interface
    bool setDynamicProperty(std::string_view name, const PropertyValue& value) override {
        std::lock_guard<std::mutex> lock(props_mutex_);
        
        if (name == "sensitivity") {
            props_.sensitivity = std::get<double>(value);
            return true;
        }
        if (name == "min_area_percent") {
            props_.min_area_percent = std::get<double>(value);
            return true;
        }
        if (name == "cooldown_ms") {
            props_.cooldown_ms = std::get<int64_t>(value);
            return true;
        }
        if (name == "enabled") {
            props_.enabled = std::get<bool>(value);
            return true;
        }
        
        return false;  // Unknown property
    }
    
    PropertyValue getDynamicProperty(std::string_view name) const override {
        std::lock_guard<std::mutex> lock(props_mutex_);
        
        if (name == "sensitivity") return props_.sensitivity;
        if (name == "min_area_percent") return props_.min_area_percent;
        if (name == "cooldown_ms") return static_cast<int64_t>(props_.cooldown_ms);
        if (name == "enabled") return props_.enabled;
        
        throw std::runtime_error("Unknown property: " + std::string(name));
    }
    
private:
    MotionDetectorModuleProps props_;
    mutable std::mutex props_mutex_;  // Protects dynamic properties
    
    // Implementation details...
};

} // namespace apra

// Registration in .cpp file:
// REGISTER_MODULE(MotionDetectorModule, MotionDetectorModuleProps)
```

---

## Appendix B: Pipeline TOML Reference

```toml
# ============================================================
# Complete Pipeline Definition Reference
# ============================================================

# Pipeline metadata (required)
[pipeline]
name = "my_pipeline"                    # Required: unique identifier
version = "1.0"                         # Optional: pipeline version
description = "Description here"        # Optional: human-readable description

# Pipeline settings (optional)
[pipeline.settings]
queue_size = 10                         # Default queue depth between modules
on_error = "restart_module"             # "stop_pipeline" | "skip_frame" | "restart_module"
auto_start = false                      # Start pipeline immediately after construction

# Module instances
# Format: [modules.<instance_id>]
[modules.my_source]
type = "RTSPSourceModule"               # Must match registered module name
    [modules.my_source.props]           # Properties for this instance
    url = "rtsp://camera/stream"
    transport = "tcp"

[modules.decoder]
type = "H264DecoderNvCodec"
    [modules.decoder.props]
    device_id = 0

[modules.motion]
type = "MotionDetectorModule"
    [modules.motion.props]
    sensitivity = 0.7
    enabled = true

[modules.writer]
type = "FileWriterModule"
    [modules.writer.props]
    path = "/recordings"

# Controller (no data connections needed)
[modules.api]
type = "RestApiController"
    [modules.api.props]
    port = 8080

# Connections (array of tables)
[[connections]]
from = "my_source.output"               # <instance_id>.<pin_name>
to = "decoder.input"

[[connections]]
from = "decoder.output"
to = "motion.input"

[[connections]]
from = "motion.motion_frames"
to = "writer.input"
```

---

## References

- [Issue #426: Intelligent Pin Connection](https://github.com/Apra-Labs/ApraPipes/issues/426)
- [TOML Specification](https://toml.io/)
- [C++20 Designated Initializers](https://en.cppreference.com/w/cpp/language/aggregate_initialization)
