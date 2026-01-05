# Developer Guide: Module Registration for Declarative Pipelines

> Complete guide for ApraPipes developers on registering modules for use in declarative (JSON-based) pipelines.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Registration Patterns](#registration-patterns)
4. [Module Categories](#module-categories)
5. [Input and Output Pins](#input-and-output-pins)
6. [Property Definitions](#property-definitions)
7. [Frame Types](#frame-types)
8. [Self-Managed Pins](#self-managed-pins)
9. [Adding applyProperties to Modules](#adding-applyproperties-to-modules)
10. [Testing Your Registration](#testing-your-registration)
11. [Common Scenarios](#common-scenarios)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)

---

## Overview

The declarative pipeline system allows users to define video processing pipelines using JSON configuration files instead of writing C++ code. For this to work, each module must be **registered** with metadata describing:

- **Category**: Source, Sink, Transform, Analytics, or Utility
- **Input/Output pins**: What frame types the module accepts and produces
- **Properties**: Configuration options available in JSON

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     JSON Pipeline File                          │
│  "modules": {                                                   │
│    "reader": {                                                  │
│      "type": "FileReaderModule",                                │
│      "props": { "strFullFileNameWithPattern": "./video.mp4" }   │
│    }                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      JsonParser                                 │
│  Parses JSON → PipelineDescription                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ModuleRegistry                               │
│  Looks up module metadata by type name                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ "FileReaderModule" → ModuleInfo (category, pins, props) │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ModuleFactory                                │
│  Creates module instances using registered factory function     │
│  Applies JSON properties via applyProperties()                  │
│  Connects modules using setNext()                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Minimal Registration (in ModuleRegistrations.cpp)

Add your module to `base/src/declarative/ModuleRegistrations.cpp`:

```cpp
#include "YourModule.h"  // Add include at top

// In ensureBuiltinModulesRegistered(), add:
if (!registry.hasModule("YourModule")) {
    registerModule<YourModule, YourModuleProps>()
        .category(ModuleCategory::Transform)
        .description("Brief description of what your module does")
        .input("input", "RawImage")
        .output("output", "RawImage");
}
```

That's it! Your module is now usable in JSON pipelines:

```json
{
  "modules": {
    "mymodule": {
      "type": "YourModule"
    }
  }
}
```

---

## Registration Patterns

### Pattern 1: Simple Transform Module

For modules that take one input and produce one output of the same type:

```cpp
registerModule<RotateCV, RotateCVProps>()
    .category(ModuleCategory::Transform)
    .description("Rotates images by a specified angle using OpenCV")
    .tags("transform", "rotate", "image", "opencv")
    .input("input", "RawImage")
    .output("output", "RawImage")
    .floatProp("angle", "Rotation angle in degrees", true, 0.0, -360.0, 360.0);
```

### Pattern 2: Source Module (No Input)

For modules that generate frames:

```cpp
registerModule<FileReaderModule, FileReaderModuleProps>()
    .category(ModuleCategory::Source)
    .description("Reads frames from files matching a pattern")
    .tags("source", "file", "reader")
    .output("output", "Frame")  // Generic output
    .stringProp("strFullFileNameWithPattern", "File path pattern", true)
    .boolProp("readLoop", "Loop back to start when reaching end", false, true)
    .enumProp("outputFrameType", "Output frame type", false, "Frame",
        "Frame", "EncodedImage", "RawImage", "RawImagePlanar");
```

### Pattern 3: Sink Module (No Output)

For modules that consume frames:

```cpp
registerModule<StatSink, StatSinkProps>()
    .category(ModuleCategory::Sink)
    .description("Statistics sink for measuring pipeline performance")
    .tags("sink", "stats", "performance", "debug")
    .input("input", "Frame");  // Accepts any frame type
```

### Pattern 4: Multi-Input Module

For modules that merge multiple streams:

```cpp
registerModule<Merge, MergeProps>()
    .category(ModuleCategory::Utility)
    .description("Merges frames from multiple input pins")
    .tags("utility", "merge", "sync")
    .input("input_1", "Frame")
    .input("input_2", "Frame")
    .output("output", "Frame")
    .selfManagedOutputPins();  // Module creates pins dynamically
```

### Pattern 5: Multi-Output Module

For modules that split streams:

```cpp
registerModule<Split, SplitProps>()
    .category(ModuleCategory::Utility)
    .description("Splits input frames across multiple output pins")
    .tags("utility", "split", "routing")
    .input("input", "Frame")
    .output("output_1", "Frame")
    .output("output_2", "Frame")
    .selfManagedOutputPins();  // Module creates pins in addInputPin()
```

### Pattern 6: Codec/Decoder Module

For modules that transform between frame types:

```cpp
registerModule<ImageDecoderCV, ImageDecoderCVProps>()
    .category(ModuleCategory::Transform)
    .description("Decodes encoded images (JPEG, PNG, BMP) to raw image format")
    .tags("decoder", "image", "opencv")
    .input("input", "EncodedImage")
    .output("output", "RawImage");
```

### Pattern 7: Analytics Module

For modules that produce detection/analysis results:

```cpp
registerModule<FaceDetectorXform, FaceDetectorXformProps>()
    .category(ModuleCategory::Analytics)
    .description("Detects faces in image frames using deep learning models")
    .tags("analytics", "face", "detection", "transform")
    .input("input", "RawImage")
    .output("output", "Frame")  // Outputs FaceDetectsInfo
    .selfManagedOutputPins();
```

---

## Module Categories

| Category | Description | Example Modules |
|----------|-------------|-----------------|
| `Source` | Generates frames (no input) | FileReaderModule, WebcamSource, RTSPClientSrc |
| `Sink` | Consumes frames (no output) | FileWriterModule, StatSink, Mp4WriterSink |
| `Transform` | Processes frames (input → output) | ImageDecoderCV, RotateCV, ColorConversion |
| `Analytics` | Produces detection/analysis data | FaceDetectorXform, QRReader |
| `Utility` | Flow control, routing | Split, Merge, ValveModule |

Usage:
```cpp
.category(ModuleCategory::Transform)
```

---

## Input and Output Pins

### Single Frame Type

```cpp
.input("input", "RawImage")
.output("output", "RawImage")
```

### Multiple Frame Types (Module Accepts Either)

```cpp
.input("input", "RawImage", "RawImagePlanar")  // Accepts both types
```

### Optional Input

```cpp
.optionalInput("mask", "RawImage")  // Optional mask input
```

### Frame Type Hierarchy

Frame types form a hierarchy for compatibility checking:

```
Frame (root)
├── RawImage
│   └── RawImagePlanar
├── EncodedImage
│   ├── H264Data
│   ├── HEVCData
│   └── BMPImage
├── Audio
├── Array
├── ControlFrame
│   ├── ChangeDetection
│   ├── PropsChange
│   └── Command
└── AnalyticsFrame
    ├── FaceDetectsInfo
    └── DefectsInfo
```

**Compatibility Rules:**
- A module expecting `RawImage` will also accept `RawImagePlanar` (subtype)
- A module expecting `Frame` accepts anything
- A module expecting `RawImage` will NOT accept `EncodedImage` (different branch)

---

## Property Definitions

### String Property

```cpp
.stringProp("name", "description", required, "default_value")

// Example:
.stringProp("strFullFileNameWithPattern", "File path pattern", true)  // Required, no default
.stringProp("outputFormat", "Output format", false, "jpeg")  // Optional with default
```

### Integer Property

```cpp
.intProp("name", "description", required, default, min, max)

// Example:
.intProp("width", "Frame width in pixels", true, 0, 1, 4096)  // Required, range 1-4096
.intProp("fps", "Frames per second", false, 30, 1, 120)  // Optional, default 30
```

### Float Property

```cpp
.floatProp("name", "description", required, default, min, max)

// Example:
.floatProp("angle", "Rotation angle in degrees", true, 0.0, -360.0, 360.0)
.floatProp("scale", "Scale factor", false, 1.0, 0.1, 10.0)
```

### Boolean Property

```cpp
.boolProp("name", "description", required, default)

// Example:
.boolProp("readLoop", "Loop playback", false, true)  // Optional, default true
.boolProp("enableMetadata", "Enable metadata track", false, false)
```

### Enum Property

```cpp
.enumProp("name", "description", required, "default", "value1", "value2", ...)

// Example:
.enumProp("conversionType", "Color space conversion", true, "RGB_TO_MONO",
    "RGB_TO_MONO", "BGR_TO_MONO", "BGR_TO_RGB", "RGB_TO_BGR",
    "RGB_TO_YUV420PLANAR", "YUV420PLANAR_TO_RGB")
```

### Dynamic Property (Runtime-Modifiable)

```cpp
.dynamicProp("name", "type", "description", required, "default")

// Example:
.dynamicProp("contrast", "float", "Contrast multiplier", false, "1.0")
.dynamicProp("text", "string", "Text to overlay", false, "")
```

---

## Frame Types

### Registering a New Frame Type

If your module uses a custom frame type, add it to `FrameTypeRegistrations.cpp`:

```cpp
// In registerBuiltinFrameTypes()
registerType("MyCustomData", "Frame",
             "Description of my custom frame type",
             {"custom", "mymodule"});
```

For hierarchical types:
```cpp
// MySpecificData inherits from AnalyticsFrame
registerType("MySpecificData", "AnalyticsFrame",
             "Specific detection results from MyModule",
             {"analytics", "detection", "mymodule"});
```

### Frame Type Best Practices

1. **Use existing types when possible** - Check `FrameTypeRegistrations.cpp` first
2. **Choose the right parent** - Your type should inherit from the most specific applicable parent
3. **Add descriptive tags** - Helps with filtering and documentation

---

## Self-Managed Pins

Some modules create output pins dynamically in their `addInputPin()` method. For these modules, you must call `.selfManagedOutputPins()`:

```cpp
registerModule<ImageEncoderCV, ImageEncoderCVProps>()
    .category(ModuleCategory::Transform)
    .description("Encodes raw images to JPEG/PNG format")
    .input("input", "RawImage")
    .output("output", "EncodedImage")
    .selfManagedOutputPins();  // REQUIRED - module creates pin in addInputPin()
```

**When to use:**
- Module's `addInputPin()` calls `addOutputPin()` internally
- Module creates multiple output pins based on input type
- Module's output pin count depends on runtime configuration

**Common modules requiring this:**
- FaceDetectorXform, QRReader (analytics modules)
- ImageEncoderCV, ImageResizeCV, RotateCV
- Split, Merge
- ColorConversion, VirtualPTZ

---

## Adding applyProperties to Modules

For full JSON property support, modules should implement `applyProperties()` in their Props class. This is done in the module's header file.

### Example: Adding applyProperties to YourModuleProps

In `YourModule.h`:

```cpp
class YourModuleProps : public ModuleProps {
public:
    std::string inputPath;
    int width = 640;
    int height = 480;
    float scale = 1.0;
    bool enableFeature = false;

    YourModuleProps() {}

    // Add this method for JSON property binding
    void applyProperties(const std::map<std::string, apra::ScalarPropertyValue>& props) {
        for (const auto& [key, value] : props) {
            if (key == "inputPath") {
                if (auto* v = std::get_if<std::string>(&value)) inputPath = *v;
            } else if (key == "width") {
                if (auto* v = std::get_if<int64_t>(&value)) width = static_cast<int>(*v);
            } else if (key == "height") {
                if (auto* v = std::get_if<int64_t>(&value)) height = static_cast<int>(*v);
            } else if (key == "scale") {
                if (auto* v = std::get_if<double>(&value)) scale = static_cast<float>(*v);
            } else if (key == "enableFeature") {
                if (auto* v = std::get_if<bool>(&value)) enableFeature = *v;
            }
        }
    }
};
```

### With Enum Properties

```cpp
enum class ConversionType { RGB_TO_MONO, BGR_TO_RGB, /* ... */ };

class ColorConversionProps : public ModuleProps {
public:
    ConversionType conversionType = ConversionType::RGB_TO_MONO;

    void applyProperties(const std::map<std::string, apra::ScalarPropertyValue>& props) {
        for (const auto& [key, value] : props) {
            if (key == "conversionType") {
                if (auto* v = std::get_if<std::string>(&value)) {
                    if (*v == "RGB_TO_MONO") conversionType = ConversionType::RGB_TO_MONO;
                    else if (*v == "BGR_TO_RGB") conversionType = ConversionType::BGR_TO_RGB;
                    // ... more mappings
                }
            }
        }
    }
};
```

**Note:** Even without `applyProperties()`, modules can still be registered. They'll just use default-constructed Props.

---

## Testing Your Registration

### 1. Build and Run Unit Tests

```bash
cmake --build build --target aprapipesut -j8
./build/aprapipesut --run_test="ModuleRegistrationTests/*"
```

### 2. Check Registration Coverage

```bash
./build/aprapipesut --run_test="ModuleRegistrationTests/AllModules_AreDiscovered"
```

This test compares registered modules against discovered Module subclasses.

### 3. Validate with CLI

```bash
# List registered modules
./build/aprapipes_cli list-modules

# Get module details
./build/aprapipes_cli describe YourModule

# Validate a JSON pipeline using your module
./build/aprapipes_cli validate your_pipeline.json
```

### 4. Create a Test Pipeline

Create `docs/declarative-pipeline/examples/working/test_yourmodule.json`:

```json
{
  "pipeline": {
    "name": "test_yourmodule"
  },
  "modules": {
    "source": {
      "type": "TestSignalGenerator",
      "props": {
        "width": 640,
        "height": 480
      }
    },
    "yourmodule": {
      "type": "YourModule",
      "props": {
      }
    },
    "sink": {
      "type": "StatSink"
    }
  },
  "connections": [
    { "from": "source", "to": "yourmodule" },
    { "from": "yourmodule", "to": "sink" }
  ]
}
```

Test it:
```bash
./build/aprapipes_cli validate docs/declarative-pipeline/examples/working/test_yourmodule.json
./build/aprapipes_cli run docs/declarative-pipeline/examples/working/test_yourmodule.json
```

---

## Common Scenarios

### Scenario 1: Creating a New Module

1. Create your module class extending `Module`
2. Create your Props class extending `ModuleProps`
3. Add `applyProperties()` to Props (optional but recommended)
4. Register in `ModuleRegistrations.cpp`
5. Test with a JSON pipeline

### Scenario 2: Modifying an Existing Module

If you add new properties to an existing module:

1. Update `applyProperties()` in the Props class header
2. Update registration in `ModuleRegistrations.cpp` to include new property
3. Update documentation and examples

### Scenario 3: Module with Dynamic Output Type

Some modules (like FileReaderModule) can output different frame types:

```cpp
registerModule<FileReaderModule, FileReaderModuleProps>()
    .output("output", "Frame")  // Generic - actual type set by property
    .enumProp("outputFrameType", "Output frame type", false, "Frame",
        "Frame", "EncodedImage", "RawImage", "RawImagePlanar");
```

### Scenario 4: Module that Bridges Frame Types

For modules like ColorConversion that convert between types:

```cpp
registerModule<ColorConversion, ColorConversionProps>()
    .input("input", "RawImage", "RawImagePlanar")  // Accepts both
    .output("output", "RawImage", "RawImagePlanar")  // Can produce either
    .enumProp("conversionType", "Conversion type", true, "RGB_TO_MONO",
        "RGB_TO_MONO", "YUV420PLANAR_TO_RGB", /* ... */);
```

### Scenario 5: Adding a New Frame Type

1. Add to `FrameTypeRegistrations.cpp`:
   ```cpp
   registerType("MyNewType", "Frame", "Description", {"tag1", "tag2"});
   ```

2. Use in module registration:
   ```cpp
   .output("output", "MyNewType")
   ```

---

## Best Practices

### 1. Match Property Names to Member Variables

```cpp
// In Props class:
std::string strFullFileNameWithPattern;

// In registration:
.stringProp("strFullFileNameWithPattern", "File path pattern", true)
```

### 2. Provide Accurate Frame Types

Don't just use "Frame" for everything. Be specific:
- `RawImage` for uncompressed pixel data
- `EncodedImage` for JPEG/PNG/compressed images
- `H264Data` for H.264 encoded video

### 3. Mark Required Properties Correctly

```cpp
.stringProp("videoPath", "Path to video file", true)  // Required
.intProp("bufferSize", "Buffer size", false, 10)      // Optional with default
```

### 4. Use selfManagedOutputPins When Needed

If validation fails with "duplicate output pin" errors, your module likely creates pins in `addInputPin()`. Add `.selfManagedOutputPins()`.

### 5. Add Meaningful Tags

Tags help with filtering and documentation:
```cpp
.tags("transform", "image", "opencv", "resize")
```

### 6. Keep Descriptions Concise

```cpp
// Good:
.description("Rotates images by a specified angle using OpenCV")

// Too verbose:
.description("This module takes an input image and rotates it by the specified angle in degrees using the OpenCV library's rotation functions...")
```

---

## Troubleshooting

### "Module not found: YourModule"

1. Check spelling matches class name exactly
2. Verify registration is in `ModuleRegistrations.cpp`
3. Ensure `#include "YourModule.h"` is at top of file

### "Frame type mismatch" Validation Error

1. Check input pin frame types match upstream module's output
2. Consider using a bridge module (e.g., ColorConversion, ImageDecoderCV)
3. Verify frame type hierarchy in `FrameTypeRegistrations.cpp`

### "Duplicate output pin" Error

Add `.selfManagedOutputPins()` to registration if your module creates output pins in `addInputPin()`.

### "Unknown property: xyz"

1. Add property to registration with `.stringProp()`, `.intProp()`, etc.
2. If property should be ignored, it's not in registration metadata

### Properties Not Applied

1. Verify `applyProperties()` exists in Props class
2. Check property name matches exactly
3. Verify type conversion (int64_t for integers, double for floats)

### CI Build Failures

After adding registration:
1. Include all necessary headers in `ModuleRegistrations.cpp`
2. Handle platform-specific modules with `#ifdef` guards:
   ```cpp
   #ifdef ENABLE_CUDA
   // CUDA-only module registrations
   #endif
   ```

---

## Reference: Complete Registration Example

```cpp
// In ModuleRegistrations.cpp:

#include "MyAwesomeModule.h"

// In ensureBuiltinModulesRegistered():
if (!registry.hasModule("MyAwesomeModule")) {
    registerModule<MyAwesomeModule, MyAwesomeModuleProps>()
        .category(ModuleCategory::Transform)
        .description("Performs awesome transformations on images")
        .tags("transform", "image", "awesome")
        .input("input", "RawImage")
        .output("output", "RawImage")
        .stringProp("mode", "Processing mode", false, "auto")
        .intProp("intensity", "Processing intensity", false, 50, 0, 100)
        .floatProp("threshold", "Detection threshold", false, 0.5, 0.0, 1.0)
        .boolProp("enableDebug", "Enable debug output", false, false)
        .enumProp("quality", "Output quality", false, "medium",
            "low", "medium", "high", "ultra")
        .selfManagedOutputPins();  // Only if module creates pins dynamically
}
```

```cpp
// In MyAwesomeModule.h:

class MyAwesomeModuleProps : public ModuleProps {
public:
    std::string mode = "auto";
    int intensity = 50;
    float threshold = 0.5f;
    bool enableDebug = false;
    std::string quality = "medium";

    void applyProperties(const std::map<std::string, apra::ScalarPropertyValue>& props) {
        for (const auto& [key, value] : props) {
            if (key == "mode") {
                if (auto* v = std::get_if<std::string>(&value)) mode = *v;
            } else if (key == "intensity") {
                if (auto* v = std::get_if<int64_t>(&value)) intensity = static_cast<int>(*v);
            } else if (key == "threshold") {
                if (auto* v = std::get_if<double>(&value)) threshold = static_cast<float>(*v);
            } else if (key == "enableDebug") {
                if (auto* v = std::get_if<bool>(&value)) enableDebug = *v;
            } else if (key == "quality") {
                if (auto* v = std::get_if<std::string>(&value)) quality = *v;
            }
        }
    }
};
```

---

## Next Steps

- See [Pipeline Author Guide](./PIPELINE_AUTHOR_GUIDE.md) for JSON pipeline creation
- Run `./build/aprapipes_cli list-modules` to see all registered modules
- Check `docs/declarative-pipeline/examples/` for example pipelines
