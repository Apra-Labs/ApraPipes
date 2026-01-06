# Developer Guide: Module Registration for Declarative Pipelines

> Complete guide for ApraPipes developers on registering modules for use in declarative (JSON/JavaScript) pipelines.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Registration Patterns](#registration-patterns)
4. [Property Definitions](#property-definitions)
5. [Dynamic Properties](#dynamic-properties)
6. [Frame Types](#frame-types)
7. [Self-Managed Pins](#self-managed-pins)
8. [Testing Your Registration](#testing-your-registration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The declarative pipeline system allows users to define video processing pipelines using:
- **JSON configuration files** (via CLI)
- **JavaScript/Node.js** (via native addon)

For this to work, each module must be **registered** with metadata describing its inputs, outputs, and properties.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     JSON Config / JS Object                      │
│  {                                                               │
│    "modules": {                                                  │
│      "ptz": { "type": "VirtualPTZ", "props": { "roiX": 0.5 } }  │
│    }                                                             │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ModuleRegistry                                │
│  Looks up module metadata by type name                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ "VirtualPTZ" → ModuleInfo (category, pins, properties)  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ModuleFactory                                 │
│  Creates module instances, applies properties, connects pins     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Step 1: Register Your Module

Add your module to `base/src/declarative/ModuleRegistrations.cpp`:

```cpp
#include "YourModule.h"  // Add include at top

// In ensureBuiltinModulesRegistered(), add:
if (!registry.hasModule("YourModule")) {
    registerModule<YourModule, YourModuleProps>()
        .category(ModuleCategory::Transform)
        .description("Brief description of what your module does")
        .input("input", "RawImage")
        .output("output", "RawImage")
        .intProp("width", "Frame width", true, 640, 1, 4096);
}
```

### Step 2: Add Property Binding to Props Class

In your module's header file, add the `applyProperties` static method:

```cpp
#include "declarative/PropertyMacros.h"

class YourModuleProps : public ModuleProps {
public:
    int width = 640;
    int height = 480;

    // Property binding for declarative pipeline
    template<typename PropsT>
    static void applyProperties(
        PropsT& props,
        const std::map<std::string, apra::ScalarPropertyValue>& values,
        std::vector<std::string>& missingRequired
    ) {
        apra::applyProp(props.width, "width", values, true, missingRequired);
        apra::applyProp(props.height, "height", values, true, missingRequired);
    }
};
```

### Step 3: Test It

Your module is now usable in JSON pipelines:

```json
{
  "modules": {
    "mymodule": {
      "type": "YourModule",
      "props": { "width": 1920, "height": 1080 }
    }
  }
}
```

And in JavaScript:

```javascript
const ap = require('./aprapipes.node');
const pipeline = ap.createPipeline({
    modules: {
        mymodule: { type: "YourModule", props: { width: 1920, height: 1080 } }
    },
    connections: []
});
```

---

## Registration Patterns

### Pattern 1: Source Module (No Input)

```cpp
registerModule<TestSignalGenerator, TestSignalGeneratorProps>()
    .category(ModuleCategory::Source)
    .description("Generates test signal frames")
    .tags("source", "test", "generator")
    .output("output", "RawImagePlanar")
    .intProp("width", "Frame width in pixels", true, 0, 1, 4096)
    .intProp("height", "Frame height in pixels", true, 0, 1, 4096)
    .stringProp("pattern", "Pattern type: GRADIENT, CHECKERBOARD, COLOR_BARS, GRID", false, "GRADIENT");
```

### Pattern 2: Transform Module

```cpp
registerModule<VirtualPTZ, VirtualPTZProps>()
    .category(ModuleCategory::Transform)
    .description("Virtual Pan-Tilt-Zoom using ROI cropping")
    .tags("transform", "ptz", "crop", "zoom")
    .input("input", "RawImage")
    .output("output", "RawImage")
    .floatProp("roiX", "ROI X position (0-1)", false, 0.0, 0.0, 1.0)
    .floatProp("roiY", "ROI Y position (0-1)", false, 0.0, 0.0, 1.0)
    .floatProp("roiWidth", "ROI width (0-1)", false, 1.0, 0.0, 1.0)
    .floatProp("roiHeight", "ROI height (0-1)", false, 1.0, 0.0, 1.0)
    .selfManagedOutputPins();
```

### Pattern 3: Sink Module (No Output)

```cpp
registerModule<FileWriterModule, FileWriterModuleProps>()
    .category(ModuleCategory::Sink)
    .description("Writes frames to files")
    .tags("sink", "file", "writer")
    .input("input", "Frame")
    .stringProp("strFullFileNameWithPattern", "Output file pattern (e.g., output_????.jpg)", true);
```

### Pattern 4: Codec Module

```cpp
registerModule<ImageEncoderCV, ImageEncoderCVProps>()
    .category(ModuleCategory::Transform)
    .description("Encodes raw images to JPEG/PNG")
    .tags("encoder", "image", "opencv", "jpeg", "png")
    .input("input", "RawImage")
    .output("output", "EncodedImage")
    .selfManagedOutputPins();
```

---

## Property Definitions

### Property Methods

| Method | Parameters | Example |
|--------|------------|---------|
| `.stringProp()` | name, desc, required, default | `.stringProp("path", "File path", true)` |
| `.intProp()` | name, desc, required, default, min, max | `.intProp("width", "Width", true, 640, 1, 4096)` |
| `.floatProp()` | name, desc, required, default, min, max | `.floatProp("scale", "Scale", false, 1.0, 0.1, 10.0)` |
| `.boolProp()` | name, desc, required, default | `.boolProp("loop", "Loop playback", false, true)` |
| `.enumProp()` | name, desc, required, default, values... | `.enumProp("mode", "Mode", false, "auto", "auto", "manual")` |

### applyProperties Pattern

Use the `apra::applyProp` helper from `PropertyMacros.h`:

```cpp
#include "declarative/PropertyMacros.h"

class MyModuleProps : public ModuleProps {
public:
    std::string path;
    int width = 640;
    float scale = 1.0f;
    bool enabled = true;

    template<typename PropsT>
    static void applyProperties(
        PropsT& props,
        const std::map<std::string, apra::ScalarPropertyValue>& values,
        std::vector<std::string>& missingRequired
    ) {
        // Required properties (4th param = true)
        apra::applyProp(props.path, "path", values, true, missingRequired);
        apra::applyProp(props.width, "width", values, true, missingRequired);

        // Optional properties (4th param = false)
        apra::applyProp(props.scale, "scale", values, false, missingRequired);
        apra::applyProp(props.enabled, "enabled", values, false, missingRequired);
    }
};
```

### Enum Properties

For enum properties, handle the string-to-enum conversion:

```cpp
class ColorConversionProps : public ModuleProps {
public:
    ColorConversionType conversionType = ColorConversionType::RGB_TO_BGR;

    template<typename PropsT>
    static void applyProperties(
        PropsT& props,
        const std::map<std::string, apra::ScalarPropertyValue>& values,
        std::vector<std::string>& missingRequired
    ) {
        auto it = values.find("conversionType");
        if (it != values.end()) {
            if (auto* v = std::get_if<std::string>(&it->second)) {
                if (*v == "RGB_TO_BGR") props.conversionType = ColorConversionType::RGB_TO_BGR;
                else if (*v == "YUV420PLANAR_TO_RGB") props.conversionType = ColorConversionType::YUV420PLANAR_TO_RGB;
                // ... more mappings
            }
        } else {
            missingRequired.push_back("conversionType");
        }
    }
};
```

---

## Dynamic Properties

Dynamic properties allow JavaScript to read and modify module properties at runtime while the pipeline is running.

### Step 1: Add Dynamic Property Methods to Props Class

```cpp
class VirtualPTZProps : public ModuleProps {
public:
    float roiX = 0.0f;
    float roiY = 0.0f;
    float roiWidth = 1.0f;
    float roiHeight = 1.0f;

    // Static method returning list of dynamic property names
    static std::vector<std::string> dynamicPropertyNames() {
        return {"roiX", "roiY", "roiWidth", "roiHeight"};
    }

    // Get property value
    apra::ScalarPropertyValue getProperty(const std::string& name) const {
        if (name == "roiX") return static_cast<double>(roiX);
        if (name == "roiY") return static_cast<double>(roiY);
        if (name == "roiWidth") return static_cast<double>(roiWidth);
        if (name == "roiHeight") return static_cast<double>(roiHeight);
        throw std::runtime_error("Unknown property: " + name);
    }

    // Set property value
    bool setProperty(const std::string& name, const apra::ScalarPropertyValue& value) {
        if (name == "roiX") {
            if (auto* v = std::get_if<double>(&value)) { roiX = static_cast<float>(*v); return true; }
        } else if (name == "roiY") {
            if (auto* v = std::get_if<double>(&value)) { roiY = static_cast<float>(*v); return true; }
        } else if (name == "roiWidth") {
            if (auto* v = std::get_if<double>(&value)) { roiWidth = static_cast<float>(*v); return true; }
        } else if (name == "roiHeight") {
            if (auto* v = std::get_if<double>(&value)) { roiHeight = static_cast<float>(*v); return true; }
        }
        return false;
    }

    // Required: applyProperties for initial config
    template<typename PropsT>
    static void applyProperties(PropsT& props, const std::map<std::string, apra::ScalarPropertyValue>& values, std::vector<std::string>& missingRequired) {
        apra::applyProp(props.roiX, "roiX", values, false, missingRequired);
        apra::applyProp(props.roiY, "roiY", values, false, missingRequired);
        apra::applyProp(props.roiWidth, "roiWidth", values, false, missingRequired);
        apra::applyProp(props.roiHeight, "roiHeight", values, false, missingRequired);
    }
};
```

### Step 2: Implement setProps() in Module

The module must apply property changes to its internal state:

```cpp
void VirtualPTZ::setProps(VirtualPTZProps& props) {
    // If pipeline not running, update directly
    if (!canQueueProps()) {
        mDetail->mProps = props;
        mDetail->setProps(props);
        return;
    }
    // If running, queue for thread-safe update
    Module::addPropsToQueue(props);
}

VirtualPTZProps VirtualPTZ::getProps() {
    fillProps(mDetail->mProps);
    return mDetail->mProps;
}
```

### Step 3: JavaScript Usage

```javascript
const ptz = pipeline.getModule('ptz');

if (ptz.hasDynamicProperties()) {
    // Get available properties
    console.log(ptz.getDynamicPropertyNames());  // ["roiX", "roiY", "roiWidth", "roiHeight"]

    // Read current value
    const x = ptz.getProperty('roiX');

    // Update at runtime (while pipeline is running)
    ptz.setProperty('roiX', 0.5);
    ptz.setProperty('roiWidth', 0.5);
}
```

---

## Frame Types

### Common Frame Types

| Type | Description | Used By |
|------|-------------|---------|
| `Frame` | Base type (accepts anything) | Generic sinks |
| `RawImage` | Uncompressed RGB/BGR pixels | Most transforms |
| `RawImagePlanar` | YUV planar format | TestSignalGenerator, video sources |
| `EncodedImage` | JPEG/PNG encoded data | Encoders, file writers |
| `H264Data` | H.264 video frames | Video decoders/encoders |

### Type Hierarchy

```
Frame (root)
├── RawImage
│   └── RawImagePlanar
├── EncodedImage
│   ├── H264Data
│   └── BMPImage
└── AnalyticsFrame
    ├── FaceDetectsInfo
    └── DefectsInfo
```

### Compatibility Rules

- **Exact match**: Always works
- **Subtype → Parent**: Works (RawImagePlanar connects to RawImage input)
- **Different branches**: Fails (RawImage cannot connect to EncodedImage)

---

## Self-Managed Pins

Add `.selfManagedOutputPins()` when your module creates output pins dynamically in `addInputPin()`:

```cpp
registerModule<ImageEncoderCV, ImageEncoderCVProps>()
    .input("input", "RawImage")
    .output("output", "EncodedImage")
    .selfManagedOutputPins();  // Module creates pin in addInputPin()
```

**Common modules requiring this:**
- Transform modules that copy input metadata to output
- Analytics modules that create output pins based on input type
- Split/Merge utilities

---

## Testing Your Registration

### 1. Build and Run Tests

```bash
cmake --build build --parallel
./build/aprapipesut --run_test="ModuleRegistrationTests/*"
```

### 2. Validate with CLI

```bash
./build/aprapipes_cli list-modules
./build/aprapipes_cli describe YourModule
```

### 3. Test with Node.js

```javascript
const ap = require('./aprapipes.node');

const pipeline = ap.createPipeline({
    modules: {
        test: { type: "YourModule", props: { width: 640 } }
    },
    connections: []
});

const mod = pipeline.getModule('test');
console.log(mod.type);  // "YourModule"
console.log(mod.hasDynamicProperties());
```

---

## Best Practices

1. **Match property names exactly** - JSON property names must match the member variable names in your Props class

2. **Use selfManagedOutputPins when needed** - If validation fails with "duplicate output pin", your module creates pins in addInputPin()

3. **Add meaningful descriptions** - Property descriptions show up in CLI and documentation

4. **Support dynamic properties for real-time control** - Any property users might want to change at runtime should be dynamic

5. **Test with actual pipelines** - Create a test JSON file and run it through the CLI

---

## Troubleshooting

### "Module not found: YourModule"

- Check spelling matches class name exactly
- Verify include is at top of ModuleRegistrations.cpp
- Ensure registration is inside `ensureBuiltinModulesRegistered()`

### "Unknown property: xyz"

- Add property with `.stringProp()`, `.intProp()`, etc.
- Property name must match exactly (case-sensitive)

### "Frame type mismatch"

- Check input/output frame types are compatible
- Use ColorConversion for RawImagePlanar → RawImage
- Use ImageDecoderCV for EncodedImage → RawImage

### "Duplicate output pin"

- Add `.selfManagedOutputPins()` to registration

### Properties not applying

- Verify `applyProperties()` exists in Props class
- Check property names match registration
- Use `apra::applyProp()` helper for type-safe binding

---

## Reference: Module Categories

| Category | Description | Examples |
|----------|-------------|----------|
| `Source` | Generates frames (no input) | TestSignalGenerator, FileReaderModule |
| `Sink` | Consumes frames (no output) | FileWriterModule, StatSink |
| `Transform` | Processes frames | VirtualPTZ, ColorConversion, ImageEncoderCV |
| `Analytics` | Detection/analysis | FaceDetectorXform, QRReader |
| `Utility` | Flow control | Split, Merge, ValveModule |

---

## See Also

- [Node.js API Reference](../node-api.md) - JavaScript API for pipelines
- [Examples](../../examples/node/) - Working Node.js examples
- [PIPELINE_AUTHOR_GUIDE.md](./PIPELINE_AUTHOR_GUIDE.md) - JSON pipeline authoring
