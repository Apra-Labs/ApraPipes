# D2: Property Binding System

**Sprint:** 2
**Priority:** P0 - Critical Path
**Effort:** 5 days
**Depends On:** D1 (Module Factory), A1 (Metadata)
**Blocks:** All module registrations

## Problem Statement

Currently, the `REGISTER_MODULE` macro creates a factory function that:
1. Creates a default-constructed `PropsClass`
2. **Does NOT apply** the properties from the TOML/map (there's a TODO)
3. Returns a module with only default values

Additionally, the current static registration approach has issues with static libraries:
- When `libaprapipes.a` is linked, unused translation units are not included
- Modules registered via static init in their `.cpp` files may never run
- This causes modules to silently not appear in the registry

```cpp
// Current broken implementation in REGISTER_MODULE:
info.factory = [](const std::map<std::string, ScalarPropertyValue>& props) {
    PropsClass moduleProps;
    /* TODO: Apply props to moduleProps based on Metadata */  // <-- NOT IMPLEMENTED!
    return std::make_unique<ModuleClass>(moduleProps);
};
```

This means **all TOML properties are ignored** and modules always get default values.

## Survey Results: Property Types Across 60+ Modules

Analysis of 30 ModuleProps classes revealed:

### Type Coverage Required

| Category | Types Found | Examples |
|----------|-------------|----------|
| **Primitives** | int, uint32_t, uint64_t, float, double, bool | deviceId, width, fps |
| **Strings** | std::string | paths, URLs, text content |
| **Containers** | std::vector<int>, std::vector<std::string> | ROI coords, file lists |
| **Enums** | 9+ custom enums | Interpolation, CodecProfile, ConversionType |
| **Special** | cudaStream_t, cv::Ptr, boost::shared_ptr | GPU resources (not TOML-settable) |

### Static vs Dynamic Split (~35% dynamic, ~65% static)

**Dynamic (can change after init):**
- ML params: `scaleFactor`, `confidenceThreshold`
- Image processing: `brightness`, `contrast`, `kernelSize`
- Overlay: `offsetX`, `offsetY`, `alpha`, text properties
- PTZ: `roiX`, `roiY`, `roiWidth`, `roiHeight`

**Static (construction-only):**
- Paths: `strFullFileNameWithPattern`, `videoPath`, `baseFolder`
- Network: `rtspURL`, `userName`, `password`
- Hardware: `cameraId`, `deviceId`, `audioInputDeviceIndex`
- Codec: `vProfile`, `gopLength`, `frameRate`
- Dimensions: `width`, `height` (for output modules)
- GPU: `stream`, `cuContext` (CUDA resources)

### Enum Types Found

1. `AffineTransformProps::Interpolation` (7 values)
2. `AffineTransformProps::TransformType` (2 values)
3. `ColorConversionProps::ConversionType` (11 values)
4. `H264EncoderNVCodecProps::H264CodecProfile` (3 values)
5. `MotionVectorExtractorProps::MVExtractMethod` (2 values)
6. `FacialLandmarkCVProps::FaceDetectionModelType` (2 values)
7. `ImageMetadata::ImageType` (multiple values)

## Design Goals

1. **DRY (Don't Repeat Yourself)**: Property defined ONCE, generates:
   - Member variable declaration
   - PropDef metadata entry
   - Property binding for set/get operations
   - Static/Dynamic marker

2. **Runtime Access**: Support `getProperty(name)` and `setProperty(name, value)` at runtime.

3. **Static vs Dynamic**: Compile-time distinction enforced:
   - Static: set via TOML, immutable after init()
   - Dynamic: can be changed at runtime via Controller modules

4. **Extensible Validators**: Easy to add new property types and validation rules.

5. **Compile-Time Safety**: Type mismatches caught at compile time.

6. **Automatic Type Conversion**: Handle TOML int64_t → int, double → float, etc.

## Proposed Solution: Single-Definition X-Macro Pattern

### The DRY Principle

Define each property **ONCE** using a macro that generates:
1. The member variable
2. The PropDef metadata entry
3. The binding for property application (applyProperties)
4. Runtime access functions (getProperty/setProperty)
5. Static/Dynamic marker for lifecycle enforcement

### PROPS_DEF Macro Pattern (Updated)

```cpp
// ============================================================
// STEP 1: Define properties ONCE using X-Macro pattern
// ============================================================

// Format: P(type, name, lifecycle, requirement, default, description)
//   lifecycle: Static | Dynamic
//   requirement: Required | Optional

#define FILE_READER_PROPS(P) \
    P(std::string, path,      Static,  Required, "",    "File path pattern") \
    P(bool,        loop,      Static,  Optional, false, "Loop when reaching end") \
    P(int,         maxFrames, Static,  Optional, -1,    "Max frames (-1=all)") \
    P(int,         startFrame,Static,  Optional, 0,     "Starting frame number")

// Example with dynamic properties (ML module)
#define FACE_DETECTOR_PROPS(P) \
    P(std::string, modelPath,   Static,  Required, "",    "Path to model file") \
    P(int,         deviceId,    Static,  Optional, 0,     "CUDA device ID") \
    P(float,       scaleFactor, Dynamic, Optional, 1.1f,  "Scale factor for detection") \
    P(float,       confidence,  Dynamic, Optional, 0.5f,  "Min confidence threshold")

// ============================================================
// STEP 2: Use the macro to generate everything
// ============================================================

class FileReaderModuleProps : public ModuleProps {
public:
    // Generate member declarations
    DECLARE_PROPS(FILE_READER_PROPS)

    // Default constructor auto-generated
    FileReaderModuleProps() : ModuleProps() {}
};
```

### What DECLARE_PROPS Generates

```cpp
// Expands to:

class FileReaderModuleProps : public ModuleProps {
public:
    // 1. Member variables with defaults
    std::string path = "";
    bool loop = false;
    int maxFrames = -1;
    int startFrame = 0;

    // 2. PropDef metadata array (with lifecycle info)
    static constexpr auto properties = std::array{
        apra::PropDef::RequiredString("path", "File path pattern", apra::PropLifecycle::Static),
        apra::PropDef::Bool("loop", false, "Loop when reaching end", apra::PropLifecycle::Static),
        apra::PropDef::Int("maxFrames", -1, "Max frames (-1=all)", apra::PropLifecycle::Static),
        apra::PropDef::Int("startFrame", 0, "Starting frame number", apra::PropLifecycle::Static)
    };

    // 3. Property application function (from TOML at construction)
    static void applyProperties(
        FileReaderModuleProps& props,
        const std::map<std::string, apra::ScalarPropertyValue>& values,
        std::vector<std::string>& missingRequired
    ) {
        apra::applyProp(props.path, "path", values, true, missingRequired);
        apra::applyProp(props.loop, "loop", values, false, missingRequired);
        apra::applyProp(props.maxFrames, "maxFrames", values, false, missingRequired);
        apra::applyProp(props.startFrame, "startFrame", values, false, missingRequired);
    }

    // 4. Runtime property getter (returns variant)
    apra::ScalarPropertyValue getProperty(const std::string& name) const {
        if (name == "path") return path;
        if (name == "loop") return loop;
        if (name == "maxFrames") return static_cast<int64_t>(maxFrames);
        if (name == "startFrame") return static_cast<int64_t>(startFrame);
        throw std::runtime_error("Unknown property: " + name);
    }

    // 5. Runtime property setter (for Dynamic properties only)
    bool setProperty(const std::string& name, const apra::ScalarPropertyValue& value) {
        // All properties are Static in this class - reject runtime changes
        throw std::runtime_error("Property '" + name + "' is static (cannot change after init)");
    }

    // 6. Get list of dynamic property names (for Controller modules)
    static std::vector<std::string> dynamicPropertyNames() {
        return {};  // No dynamic properties in FileReaderModule
    }

    // Default constructor
    FileReaderModuleProps() : ModuleProps() {}
};
```

### Example with Dynamic Properties

```cpp
// FaceDetectorXformProps with dynamic properties:

class FaceDetectorXformProps : public ModuleProps {
public:
    // Members
    std::string modelPath = "";
    int deviceId = 0;
    float scaleFactor = 1.1f;  // Dynamic - can change at runtime
    float confidence = 0.5f;    // Dynamic - can change at runtime

    // ... properties array with lifecycle info ...

    // Runtime setter handles dynamic properties
    bool setProperty(const std::string& name, const apra::ScalarPropertyValue& value) {
        if (name == "scaleFactor") {
            scaleFactor = std::get<double>(value);
            return true;
        }
        if (name == "confidence") {
            confidence = std::get<double>(value);
            return true;
        }
        // Static properties cannot be changed
        if (name == "modelPath" || name == "deviceId") {
            throw std::runtime_error("Property '" + name + "' is static");
        }
        throw std::runtime_error("Unknown property: " + name);
    }

    static std::vector<std::string> dynamicPropertyNames() {
        return {"scaleFactor", "confidence"};
    }
};
```

### Extended Property Types

```cpp
// Format variations for different property types:

#define MY_MODULE_PROPS(P) \
    /* Required string */ \
    P(std::string, url, Required, "", "RTSP URL") \
    \
    /* Optional with default */ \
    P(bool, enabled, Optional, true, "Enable processing") \
    \
    /* Int with range */ \
    P_RANGE(int, deviceId, Optional, 0, 0, 7, "CUDA device ID") \
    \
    /* Float with range */ \
    P_RANGE(float, sensitivity, Optional, 0.5f, 0.0f, 1.0f, "Detection sensitivity") \
    \
    /* Enum */ \
    P_ENUM(std::string, algorithm, Optional, "mog2", (mog2, knn, gsoc), "Algorithm") \
    \
    /* String with regex */ \
    P_REGEX(std::string, ipAddr, Required, "", R"(\d+\.\d+\.\d+\.\d+)", "IP address")
```

### Complete Example

```cpp
// ============================================================
// FileReaderModule.h
// ============================================================

#include "declarative/PropertyMacros.h"

// Property definitions (single source of truth)
#define FILE_READER_PROPS(P) \
    P(std::string, path,       Required, "",    "File path pattern") \
    P(bool,        loop,       Optional, false, "Loop when reaching end") \
    P(int,         maxFrames,  Optional, -1,    "Max frames to read (-1=all)") \
    P(int,         startFrame, Optional, 0,     "Starting frame number")

class FileReaderModuleProps : public ModuleProps {
public:
    DECLARE_PROPS(FILE_READER_PROPS)

    // Default constructor
    FileReaderModuleProps() : ModuleProps() {}

    // Optional: convenience constructor for C++ usage
    FileReaderModuleProps(const std::string& _path, bool _loop = false)
        : ModuleProps(), path(_path), loop(_loop) {}
};

class FileReaderModule : public Module {
public:
    struct Metadata {
        static constexpr std::string_view name = "FileReaderModule";
        static constexpr apra::ModuleCategory category = apra::ModuleCategory::Source;
        static constexpr std::string_view version = "1.0";
        static constexpr std::string_view description = "Reads frames from files";

        static constexpr std::array<std::string_view, 2> tags = {"reader", "file"};
        static constexpr std::array<apra::PinDef, 0> inputs = {};
        static constexpr std::array<apra::PinDef, 1> outputs = {
            apra::PinDef::output("output", FrameMetadata::ENCODED_IMAGE)
        };

        // Reference the Props properties (DRY!)
        static constexpr auto& properties = FileReaderModuleProps::properties;
    };

    explicit FileReaderModule(FileReaderModuleProps props);
    // ... implementation
};

// In FileReaderModule.cpp:
REGISTER_MODULE(FileReaderModule, FileReaderModuleProps)
```

## Implementation

### Header: `declarative/PropertyMacros.h`

```cpp
#pragma once
#include "declarative/Metadata.h"
#include <map>
#include <string>
#include <variant>
#include <vector>
#include <type_traits>
#include <array>
#include <algorithm>
#include <stdexcept>

namespace apra {

// ============================================================
// Property Lifecycle enum (in Metadata.h, shown here for context)
// ============================================================
// enum class PropLifecycle { Static, Dynamic };

// ============================================================
// Type-safe property application with automatic conversion
// ============================================================
template<typename T>
void applyProp(
    T& member,
    const char* propName,
    const std::map<std::string, ScalarPropertyValue>& values,
    bool isRequired,
    std::vector<std::string>& missingRequired
) {
    auto it = values.find(propName);
    if (it == values.end()) {
        if (isRequired) {
            missingRequired.push_back(propName);
        }
        return;  // Keep default value
    }

    std::visit([&member](auto&& val) {
        using V = std::decay_t<decltype(val)>;

        if constexpr (std::is_same_v<T, std::string> && std::is_same_v<V, std::string>) {
            member = val;
        }
        else if constexpr (std::is_same_v<T, bool> && std::is_same_v<V, bool>) {
            member = val;
        }
        else if constexpr (std::is_integral_v<T> && std::is_same_v<V, int64_t>) {
            member = static_cast<T>(val);
        }
        else if constexpr (std::is_floating_point_v<T> && std::is_same_v<V, double>) {
            member = static_cast<T>(val);
        }
        else if constexpr (std::is_floating_point_v<T> && std::is_same_v<V, int64_t>) {
            member = static_cast<T>(val);
        }
    }, it->second);
}

// ============================================================
// Helper to convert member to ScalarPropertyValue
// ============================================================
template<typename T>
ScalarPropertyValue toPropertyValue(const T& member) {
    if constexpr (std::is_same_v<T, std::string>) {
        return member;
    }
    else if constexpr (std::is_same_v<T, bool>) {
        return member;
    }
    else if constexpr (std::is_integral_v<T>) {
        return static_cast<int64_t>(member);
    }
    else if constexpr (std::is_floating_point_v<T>) {
        return static_cast<double>(member);
    }
    else {
        static_assert(sizeof(T) == 0, "Unsupported property type");
    }
}

// ============================================================
// Helper to apply value from variant to member
// ============================================================
template<typename T>
bool applyFromVariant(T& member, const ScalarPropertyValue& value) {
    return std::visit([&member](auto&& val) -> bool {
        using V = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, std::string> && std::is_same_v<V, std::string>) {
            member = val; return true;
        }
        else if constexpr (std::is_same_v<T, bool> && std::is_same_v<V, bool>) {
            member = val; return true;
        }
        else if constexpr (std::is_integral_v<T> && std::is_same_v<V, int64_t>) {
            member = static_cast<T>(val); return true;
        }
        else if constexpr (std::is_floating_point_v<T> &&
                          (std::is_same_v<V, double> || std::is_same_v<V, int64_t>)) {
            member = static_cast<T>(val); return true;
        }
        return false;
    }, value);
}

} // namespace apra

// ============================================================
// X-Macro Helpers
// ============================================================

// Mutability markers for X-Macro (matches PropDef::Mutability in Metadata.h)
#define PROP_MUTABILITY_Static  apra::PropDef::Mutability::Static
#define PROP_MUTABILITY_Dynamic apra::PropDef::Mutability::Dynamic

// Requirement markers
#define PROP_IS_REQ_Required true
#define PROP_IS_REQ_Optional false

// ============================================================
// Member Declaration Generator
// P(type, name, lifecycle, requirement, default, description)
// ============================================================
#define PROP_DECL_MEMBER(type, name, lifecycle, req, def, desc) type name = def;

// ============================================================
// PropDef Array Generator
// ============================================================
#define PROP_MAKE_DEF(type, name, lifecycle, req, def, desc) \
    apra::PropDef::create<type>(#name, def, desc, PROP_MUTABILITY_##lifecycle, !PROP_IS_REQ_##req),

// ============================================================
// Apply Properties Generator
// ============================================================
#define PROP_APPLY(type, name, lifecycle, req, def, desc) \
    apra::applyProp(props.name, #name, values, PROP_IS_REQ_##req, missingRequired);

// ============================================================
// Get Property Generator
// ============================================================
#define PROP_GET(type, name, lifecycle, req, def, desc) \
    if (propName == #name) return apra::toPropertyValue(name);

// ============================================================
// Set Property Generator (for Dynamic properties only)
// ============================================================
#define PROP_SET_STATIC(type, name, lifecycle, req, def, desc) \
    if (propName == #name) { \
        throw std::runtime_error("Cannot modify static property '" #name "' after init()"); \
    }
#define PROP_SET_DYNAMIC(type, name, lifecycle, req, def, desc) \
    if (propName == #name) { \
        return apra::applyFromVariant(name, value); \
    }
#define PROP_SET(type, name, lifecycle, req, def, desc) \
    PROP_SET_##lifecycle(type, name, lifecycle, req, def, desc)

// ============================================================
// Dynamic Property Names Generator
// ============================================================
#define PROP_DYN_NAME_Static(name)
#define PROP_DYN_NAME_Dynamic(name) names.push_back(#name);
#define PROP_DYN_NAME(type, name, lifecycle, req, def, desc) \
    PROP_DYN_NAME_##lifecycle(name)

// ============================================================
// DECLARE_PROPS - Main macro that generates everything
// ============================================================
#define DECLARE_PROPS(PROPS_MACRO) \
    /* 1. Declare member variables with defaults */ \
    PROPS_MACRO(PROP_DECL_MEMBER) \
    \
    /* 2. Static PropDef array for introspection */ \
    static inline const std::vector<apra::PropDef>& getPropertyDefs() { \
        static std::vector<apra::PropDef> defs = { \
            PROPS_MACRO(PROP_MAKE_DEF) \
        }; \
        return defs; \
    } \
    \
    /* 3. Apply properties from TOML/map at construction */ \
    template<typename PropsT> \
    static void applyProperties( \
        PropsT& props, \
        const std::map<std::string, apra::ScalarPropertyValue>& values, \
        std::vector<std::string>& missingRequired \
    ) { \
        PROPS_MACRO(PROP_APPLY) \
    } \
    \
    /* 4. Get property by name (runtime introspection) */ \
    apra::ScalarPropertyValue getProperty(const std::string& propName) const { \
        PROPS_MACRO(PROP_GET) \
        throw std::runtime_error("Unknown property: " + propName); \
    } \
    \
    /* 5. Set property by name (only Dynamic properties allowed) */ \
    bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) { \
        PROPS_MACRO(PROP_SET) \
        throw std::runtime_error("Unknown property: " + propName); \
    } \
    \
    /* 6. Get list of dynamic (runtime-modifiable) property names */ \
    static std::vector<std::string> dynamicPropertyNames() { \
        std::vector<std::string> names; \
        PROPS_MACRO(PROP_DYN_NAME) \
        return names; \
    }
```

### Updated REGISTER_MODULE Macro

```cpp
#define REGISTER_MODULE(ModuleClass, PropsClass) \
    static_assert(std::is_default_constructible<PropsClass>::value, \
        "REGISTER_MODULE requires " #PropsClass " to have a default constructor."); \
    namespace { \
        static bool _registered_##ModuleClass = []() { \
            apra::ModuleInfo info; \
            info.name = std::string(ModuleClass::Metadata::name); \
            info.category = ModuleClass::Metadata::category; \
            info.version = std::string(ModuleClass::Metadata::version); \
            info.description = std::string(ModuleClass::Metadata::description); \
            \
            for (const auto& tag : ModuleClass::Metadata::tags) \
                info.tags.push_back(std::string(tag)); \
            for (const auto& pin : ModuleClass::Metadata::inputs) \
                info.inputs.push_back(apra::detail::toPinInfo(pin)); \
            for (const auto& pin : ModuleClass::Metadata::outputs) \
                info.outputs.push_back(apra::detail::toPinInfo(pin)); \
            for (const auto& prop : ModuleClass::Metadata::properties) \
                info.properties.push_back(apra::detail::toPropInfo(prop)); \
            \
            /* Factory function - creates module with props applied */ \
            info.factory = [](const std::map<std::string, apra::ScalarPropertyValue>& props) \
                -> std::unique_ptr<Module> { \
                PropsClass moduleProps; \
                std::vector<std::string> missingRequired; \
                \
                /* Apply properties using generated applyProperties */ \
                PropsClass::applyProperties(moduleProps, props, missingRequired); \
                \
                /* Check required properties */ \
                if (!missingRequired.empty()) { \
                    std::string msg = "Missing required properties: "; \
                    for (size_t i = 0; i < missingRequired.size(); ++i) { \
                        if (i > 0) msg += ", "; \
                        msg += missingRequired[i]; \
                    } \
                    throw std::runtime_error(msg); \
                } \
                \
                return std::make_unique<ModuleClass>(moduleProps); \
            }; \
            \
            apra::ModuleRegistry::instance().registerModule(std::move(info)); \
            return true; \
        }(); \
    }
```

## Migration Path for Existing Modules

For existing modules that already have member variables with different names, there are two approaches:

### Option A: Refactor to use X-Macro (Recommended for new code)

```cpp
// Replace old Props class:
// OLD:
class FileReaderModuleProps : public ModuleProps {
    std::string strFullFileNameWithPattern;  // Old name
    bool readLoop;                           // Old name
};

// NEW: Use X-Macro
#define FILE_READER_PROPS(P) \
    P(std::string, path, Required, "", "File path") \
    P(bool, loop, Optional, false, "Loop playback")

class FileReaderModuleProps : public ModuleProps {
public:
    DECLARE_PROPS(FILE_READER_PROPS)
    FileReaderModuleProps() : ModuleProps() {}
};
```

### Option B: Legacy Binding (For gradual migration)

Keep existing member names, add explicit binding:

```cpp
class FileReaderModuleProps : public ModuleProps {
public:
    // Keep existing member names
    std::string strFullFileNameWithPattern = "";
    bool readLoop = false;

    // Add metadata with TOML-friendly names
    static constexpr auto properties = std::array{
        apra::PropDef::RequiredString("path", "File path"),
        apra::PropDef::Bool("loop", false, "Loop playback")
    };

    // Explicit binding: TOML name -> member
    static void applyProperties(
        FileReaderModuleProps& props,
        const std::map<std::string, apra::ScalarPropertyValue>& values,
        std::vector<std::string>& missingRequired
    ) {
        apra::applyProp(props.strFullFileNameWithPattern, "path", values, true, missingRequired);
        apra::applyProp(props.readLoop, "loop", values, false, missingRequired);
    }

    FileReaderModuleProps() : ModuleProps() {}
};
```

## Validation Flow

```
TOML file
    ↓
TomlParser.parse() → PipelineDescription
    ↓
PipelineValidator.validate() → Check required props, types, ranges
    ↓
ModuleFactory.build()
    ↓
ModuleRegistry.createModule(type, props)
    ↓
factory lambda:
    1. PropsClass moduleProps;           // default constructed
    2. PropsClass::applyProperties(...)  // apply TOML values
    3. Check missingRequired             // fail if required missing
    4. return make_unique<Module>(moduleProps)
```

## Error Messages

```
[ERROR] E012 @ modules.source.props.path: Required property 'path' not provided for module 'FileReaderModule'
[ERROR] E011 @ modules.decoder.props.device_id: Property type mismatch: expected int, got string
[WARN]  W002 @ modules.encoder.props.bitrate: Property 'bitrate' not recognized, ignoring
```

## Summary: Developer Effort Comparison

### OLD Way (Current - Broken)

Developer must:
1. Define member variable with default
2. Define PropDef metadata separately (duplicate name, type, default)
3. Write BIND_PROPS mapping (third mention of property)
4. Hope they stay in sync (easy to drift)

```cpp
// 4 places to maintain for each property!
std::string path = "";                                    // 1. Member
PropDef::RequiredString("path", "..."),                   // 2. Metadata (duplicates name)
BIND(path),                                               // 3. Binding (duplicates name again)
// Plus default constructor initialization                 // 4. Constructor
```

### NEW Way (DRY - Proposed)

Developer defines each property ONCE:

```cpp
// ONE definition per property!
#define FILE_READER_PROPS(P) \
    P(std::string, path, Required, "", "File path")       // Everything in one place

class FileReaderModuleProps : public ModuleProps {
public:
    DECLARE_PROPS(FILE_READER_PROPS)                      // Generates all 3 automatically
    FileReaderModuleProps() : ModuleProps() {}
};
```

## Acceptance Criteria

### Core Property Binding
- [ ] X-Macro pattern works with DECLARE_PROPS
- [ ] Properties of all types work: string, bool, int, float
- [ ] Required properties cause build failure when missing
- [ ] Optional properties use defaults when not specified
- [ ] Type conversion works (int64 → int, double → float)
- [ ] Legacy binding (Option B) works for gradual migration

### Runtime Property Access
- [ ] `getProperty(name)` returns current value as variant
- [ ] `setProperty(name, value)` updates dynamic properties
- [ ] `setProperty` throws for static properties after init
- [ ] `dynamicPropertyNames()` returns list of modifiable properties

### Static vs Dynamic Lifecycle
- [x] `PropDef::Mutability::Static` and `PropDef::Mutability::Dynamic` (already in Metadata.h)
- [x] PropDef stores mutability information (already implemented)
- [ ] X-Macro `P(type, name, lifecycle, req, default, desc)` syntax
- [ ] Schema generator exports lifecycle info for documentation

### Extensibility
- [ ] ValidatorRegistry supports custom validators
- [ ] New property types can be added without core changes
- [ ] `P_INTERNAL` variant for non-TOML properties (e.g., cudaStream_t)

### Testing
- [ ] Unit tests cover all property types
- [ ] Unit tests for getProperty/setProperty
- [ ] Unit tests for static vs dynamic enforcement
- [ ] At least 2 modules migrated as examples

## Definition of Done

- [ ] `declarative/PropertyMacros.h` implemented with:
  - DECLARE_PROPS macro
  - applyProp template functions
  - getProperty/setProperty generators
  - Static/Dynamic lifecycle support
- [ ] `declarative/PropertyValidators.h` implemented with:
  - PropertyValidator base class
  - RangeValidator, RegexValidator, EnumValidator
  - ValidatorRegistry singleton
- [x] `declarative/Metadata.h` already has:
  - PropDef::Mutability enum (Static, Dynamic)
  - PropDef mutability field
  - DynamicInt/DynamicFloat/etc. convenience methods
- [ ] REGISTER_MODULE updated to call applyProperties
- [ ] Unit tests for property binding
- [ ] Integration test: TOML → Parse → Build → Properties applied correctly
- [ ] 2 pilot modules converted (FileWriterModule, one other)
- [ ] Documentation updated

## Extensibility Framework

The property system is designed to be easily extensible without modifying core code.

### Adding New Property Types

To add a new property type (e.g., `std::vector<int>`):

```cpp
// 1. Add type conversion in applyProp (PropertyMacros.h)
template<>
void applyProp<std::vector<int>>(
    std::vector<int>& member,
    const char* propName,
    const std::map<std::string, ScalarPropertyValue>& values,
    bool isRequired,
    std::vector<std::string>& missingRequired
) {
    // Handle array conversion from TOML
    // PropertyValue (not ScalarPropertyValue) supports arrays
}

// 2. Add PropDef factory method (Metadata.h)
namespace apra {
    struct PropDef {
        // ... existing methods ...

        static constexpr PropDef IntArray(
            std::string_view name,
            std::string_view description,
            PropLifecycle lifecycle = PropLifecycle::Static
        ) {
            return PropDef{name, "int[]", description, {}, {}, {}, lifecycle, false};
        }
    };
}

// 3. Add X-Macro variant (PropertyMacros.h)
#define P_ARRAY(elem_type, name, lifecycle, req, desc) \
    // ... generates std::vector<elem_type> member ...
```

### Adding New Validators

Validators are registered with the system and can be applied to any property:

```cpp
// 1. Define validator in PropertyValidators.h
namespace apra::validators {

// Base interface for validators
struct PropertyValidator {
    virtual ~PropertyValidator() = default;
    virtual bool validate(const ScalarPropertyValue& value, std::string& error) const = 0;
    virtual std::string describe() const = 0;
};

// Range validator (generic)
template<typename T>
class RangeValidator : public PropertyValidator {
    T min_, max_;
public:
    RangeValidator(T min, T max) : min_(min), max_(max) {}

    bool validate(const ScalarPropertyValue& value, std::string& error) const override {
        auto v = std::get<T>(value);
        if (v < min_ || v > max_) {
            error = "Value " + std::to_string(v) + " out of range ["
                  + std::to_string(min_) + ", " + std::to_string(max_) + "]";
            return false;
        }
        return true;
    }

    std::string describe() const override {
        return "range[" + std::to_string(min_) + "," + std::to_string(max_) + "]";
    }
};

// Regex validator for strings
class RegexValidator : public PropertyValidator {
    std::string pattern_;
    std::regex regex_;
public:
    explicit RegexValidator(std::string pattern)
        : pattern_(std::move(pattern)), regex_(pattern_) {}

    bool validate(const ScalarPropertyValue& value, std::string& error) const override {
        auto& str = std::get<std::string>(value);
        if (!std::regex_match(str, regex_)) {
            error = "String '" + str + "' does not match pattern: " + pattern_;
            return false;
        }
        return true;
    }

    std::string describe() const override { return "regex(" + pattern_ + ")"; }
};

// Enum validator
class EnumValidator : public PropertyValidator {
    std::vector<std::string> allowed_;
public:
    EnumValidator(std::initializer_list<std::string> values) : allowed_(values) {}

    bool validate(const ScalarPropertyValue& value, std::string& error) const override {
        auto& str = std::get<std::string>(value);
        if (std::find(allowed_.begin(), allowed_.end(), str) == allowed_.end()) {
            error = "Invalid value '" + str + "'. Allowed: ";
            for (size_t i = 0; i < allowed_.size(); ++i) {
                if (i > 0) error += ", ";
                error += allowed_[i];
            }
            return false;
        }
        return true;
    }

    std::string describe() const override {
        std::string s = "enum{";
        for (size_t i = 0; i < allowed_.size(); ++i) {
            if (i > 0) s += ",";
            s += allowed_[i];
        }
        return s + "}";
    }
};

} // namespace apra::validators

// 2. Register custom validator
namespace apra {
    class ValidatorRegistry {
        std::map<std::string, std::shared_ptr<validators::PropertyValidator>> validators_;
    public:
        static ValidatorRegistry& instance();

        void registerValidator(const std::string& name,
                               std::shared_ptr<validators::PropertyValidator> v) {
            validators_[name] = std::move(v);
        }

        std::shared_ptr<validators::PropertyValidator> get(const std::string& name) const {
            auto it = validators_.find(name);
            return it != validators_.end() ? it->second : nullptr;
        }
    };
}

// 3. Use in X-Macro with custom validators
#define MY_PROPS(P) \
    P_VALIDATED(int, port, Static, Required, 8080, "Port number", "range:1024:65535") \
    P_VALIDATED(std::string, email, Static, Required, "", "Email", "regex:.*@.*\\..*")
```

### Adding Platform-Specific Properties

For CUDA/Jetson-only properties that cannot be set via TOML:

```cpp
// Use P_INTERNAL for non-TOML properties (set programmatically)
#define CUDA_MODULE_PROPS(P) \
    P(int, deviceId, Static, Optional, 0, "CUDA device ID") \
    P_INTERNAL(cudaStream_t, stream, "CUDA stream - set via setStream()")

// P_INTERNAL generates:
// - Member variable (no default)
// - NO PropDef entry (not in TOML schema)
// - NO apply from TOML
// - getProperty/setProperty work normally

// Usage in module:
class CudaModuleProps : public ModuleProps {
public:
    DECLARE_PROPS(CUDA_MODULE_PROPS)

    // Additional setter for internal property
    void setStream(cudaStream_t s) { stream = s; }
};
```

## Runtime Property Access

### Use Cases

1. **Controller Modules**: Change parameters during pipeline execution
2. **Debugging**: Inspect current property values
3. **Serialization**: Save/restore pipeline state
4. **UI Integration**: Build property editors

### Interface

```cpp
// In Module base class (extends existing API)
class Module {
public:
    // ... existing methods ...

    // Runtime property access (delegates to Props)
    virtual ScalarPropertyValue getProperty(const std::string& name) const {
        throw std::runtime_error("Property access not implemented");
    }

    virtual bool setProperty(const std::string& name, const ScalarPropertyValue& value) {
        throw std::runtime_error("Property modification not implemented");
    }

    virtual std::vector<std::string> getDynamicPropertyNames() const {
        return {};
    }

    // Check if property is dynamic
    virtual bool isPropertyDynamic(const std::string& name) const {
        auto names = getDynamicPropertyNames();
        return std::find(names.begin(), names.end(), name) != names.end();
    }
};

// Module implementation forwards to props
class FaceDetectorXform : public Module {
    FaceDetectorXformProps props_;
public:
    ScalarPropertyValue getProperty(const std::string& name) const override {
        return props_.getProperty(name);
    }

    bool setProperty(const std::string& name, const ScalarPropertyValue& value) override {
        if (!isPropertyDynamic(name)) {
            throw std::runtime_error("Cannot modify static property: " + name);
        }
        return props_.setProperty(name, value);
    }

    std::vector<std::string> getDynamicPropertyNames() const override {
        return FaceDetectorXformProps::dynamicPropertyNames();
    }
};
```

### Controller Module Example

```cpp
// Example: ParameterController module that adjusts detection threshold
class ParameterController : public Module {
    Module* target_;
    std::string propertyName_;
public:
    void step() override {
        // Read current value
        auto current = target_->getProperty(propertyName_);

        // Compute new value (e.g., from feedback loop)
        auto newValue = computeNewValue(current);

        // Update target module's dynamic property
        target_->setProperty(propertyName_, newValue);
    }
};
```

## Static vs Dynamic Properties - Lifecycle Enforcement

### Already Implemented in Metadata.h

```cpp
// PropDef already supports this via Mutability enum:
struct PropDef {
    enum class Mutability {
        Static,   // Set at construction, cannot change
        Dynamic   // Can be modified at runtime via Controller
    };

    Mutability mutability = Mutability::Static;

    // Factory methods already support it:
    static constexpr PropDef Int(..., Mutability mut = Mutability::Static);
    static constexpr PropDef DynamicInt(...);  // Convenience for Dynamic
};
```

### Runtime Enforcement

```cpp
// Generated setProperty checks lifecycle
bool setProperty(const std::string& name, const ScalarPropertyValue& value) {
    // P_STATIC properties:
    if (name == "modelPath") {
        throw std::runtime_error("Cannot modify static property 'modelPath' after init()");
    }

    // P_DYNAMIC properties:
    if (name == "scaleFactor") {
        scaleFactor = std::get<double>(value);
        onPropertyChanged("scaleFactor");  // Optional: notify module
        return true;
    }

    throw std::runtime_error("Unknown property: " + name);
}
```

### Property Change Notifications

Modules can override to react to dynamic property changes:

```cpp
class FaceDetectorXform : public Module {
protected:
    virtual void onPropertyChanged(const std::string& name) {
        if (name == "scaleFactor" || name == "confidence") {
            // Reconfigure detector with new parameters
            reconfigureDetector();
        }
    }
};
```

## Module Registration Design (Revised)

### Problem with Static Registration in Static Libraries

When `libaprapipes.a` is linked into a customer application:

```
Customer's main.cpp
       ↓ links
libaprapipes.a
  ├── FileReaderModule.o    ← Has REGISTER_MODULE static init
  ├── H264Decoder.o         ← Has REGISTER_MODULE static init
  ├── TomlParser.o          ← Customer uses this
  └── ...

Linker: "Customer only calls TomlParser, I'll only include TomlParser.o"
Result: FileReaderModule.o not included → not registered!
```

### Solution: Lazy Registration + Central Registration File

**1. Central `ModuleRegistrations.cpp` with fluent builder:**

```cpp
// base/src/declarative/ModuleRegistrations.cpp

#include "declarative/ModuleRegistrations.h"
#include "declarative/ModuleRegistry.h"

// Include all module headers
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "H264Decoder.h"
// ... all 60+ modules

namespace apra {

void ensureBuiltinModulesRegistered() {
    static std::once_flag flag;
    std::call_once(flag, []() {

        registerModule<FileReaderModule, FileReaderModuleProps>()
            .category(ModuleCategory::Source)
            .description("Reads frames from image/video files")
            .tags("reader", "file", "source")
            .output("output", "EncodedImage");

        registerModule<FileWriterModule, FileWriterModuleProps>()
            .category(ModuleCategory::Sink)
            .description("Writes frames to files")
            .tags("writer", "file", "sink")
            .input("input", "RawImage", "EncodedImage");

        registerModule<H264Decoder, H264DecoderProps>()
            .category(ModuleCategory::Transform)
            .description("Decodes H264 video frames")
            .tags("decoder", "h264", "video")
            .input("encoded", "H264Frame")
            .output("decoded", "RawImagePlanar");

        // ... all other modules
    });
}

} // namespace apra
```

**2. Trigger registration on first declarative pipeline use:**

```cpp
// TomlParser.cpp
PipelineDescription TomlParser::parse(const std::string& toml) {
    ensureBuiltinModulesRegistered();  // Called on first use
    // ... parsing logic
}

// ModuleFactory.cpp
BuildResult ModuleFactory::build(const PipelineDescription& desc) {
    ensureBuiltinModulesRegistered();  // Also here for safety
    // ... build logic
}
```

**3. Fluent builder implementation:**

```cpp
// declarative/ModuleRegistrationBuilder.h

template<typename ModuleClass, typename PropsClass>
class ModuleRegistrationBuilder {
    ModuleInfo info_;
public:
    ModuleRegistrationBuilder() {
        info_.name = typeid(ModuleClass).name();  // Demangled later
        // Or use __PRETTY_FUNCTION__ parsing
    }

    ModuleRegistrationBuilder& category(ModuleCategory cat) {
        info_.category = cat;
        return *this;
    }

    ModuleRegistrationBuilder& description(std::string_view desc) {
        info_.description = std::string(desc);
        return *this;
    }

    ModuleRegistrationBuilder& tags(auto... t) {
        (info_.tags.push_back(std::string(t)), ...);
        return *this;
    }

    ModuleRegistrationBuilder& input(std::string_view name, auto... frameTypes) {
        PinInfo pin{std::string(name), {}, true};
        (pin.frameTypes.push_back(std::string(frameTypes)), ...);
        info_.inputs.push_back(std::move(pin));
        return *this;
    }

    ModuleRegistrationBuilder& output(std::string_view name, auto... frameTypes) {
        PinInfo pin{std::string(name), {}, true};
        (pin.frameTypes.push_back(std::string(frameTypes)), ...);
        info_.outputs.push_back(std::move(pin));
        return *this;
    }

    ~ModuleRegistrationBuilder() {
        // Factory function with property binding
        info_.factory = [](const std::map<std::string, ScalarPropertyValue>& props) {
            PropsClass moduleProps;
            std::vector<std::string> missing;
            PropsClass::applyProperties(moduleProps, props, missing);
            if (!missing.empty()) {
                throw std::runtime_error("Missing required properties: " + join(missing));
            }
            return std::make_unique<ModuleClass>(moduleProps);
        };

        ModuleRegistry::instance().registerModule(std::move(info_));
    }
};

template<typename ModuleClass, typename PropsClass>
ModuleRegistrationBuilder<ModuleClass, PropsClass> registerModule() {
    return ModuleRegistrationBuilder<ModuleClass, PropsClass>();
}
```

### Detecting Unregistered Modules

**CMake scanner generates list of all Module subclasses:**

```cmake
# cmake/ScanModuleClasses.cmake
file(GLOB_RECURSE HEADERS "${CMAKE_SOURCE_DIR}/base/include/*.h")

set(MODULE_CLASSES "")
foreach(header ${HEADERS})
    file(READ ${header} content)
    string(REGEX MATCHALL "class ([A-Za-z0-9_]+) *: *public Module" matches "${content}")
    foreach(match ${matches})
        string(REGEX REPLACE "class ([A-Za-z0-9_]+).*" "\\1" classname "${match}")
        list(APPEND MODULE_CLASSES "\"${classname}\"")
    endforeach()
endforeach()

list(REMOVE_DUPLICATES MODULE_CLASSES)
list(JOIN MODULE_CLASSES ",\n    " MODULE_CLASSES_STR)
file(WRITE "${CMAKE_BINARY_DIR}/generated/module_subclasses.inc" "    ${MODULE_CLASSES_STR}")
```

**Unit test catches missing registrations:**

```cpp
// test/declarative/module_registration_tests.cpp

BOOST_AUTO_TEST_CASE(AllConcreteModulesRegistered) {
    // Generated by CMake - all classes inheriting from Module
    std::set<std::string> allModuleSubclasses = {
        #include "generated/module_subclasses.inc"
    };

    // Known abstract/excluded classes
    std::set<std::string> excluded = {
        "AbsControlModule",   // Abstract base class
        "Module",             // Base class itself
    };

    // Trigger registration
    ensureBuiltinModulesRegistered();
    auto& registry = ModuleRegistry::instance();

    // Check each module
    std::vector<std::string> missing;
    for (const auto& moduleName : allModuleSubclasses) {
        if (excluded.count(moduleName)) continue;
        if (!registry.hasModule(moduleName)) {
            missing.push_back(moduleName);
        }
    }

    if (!missing.empty()) {
        std::stringstream msg;
        msg << "\n\nUNREGISTERED MODULES DETECTED:\n\n";
        for (const auto& m : missing) {
            msg << "  - " << m << "\n";
        }
        msg << "\nTo fix, either:\n";
        msg << "  1. Add to base/src/declarative/ModuleRegistrations.cpp (if concrete)\n";
        msg << "  2. Add to 'excluded' set in this test (if abstract base)\n\n";
        BOOST_FAIL(msg.str());
    }
}
```

### Benefits of This Design

| Aspect | Benefit |
|--------|---------|
| No linker issues | Registration file is always linked (used by TomlParser) |
| No SIOF | Registration happens at defined time (first parse/build) |
| DRY | Module name derived from class, not typed twice |
| Readable | Fluent builder syntax is self-documenting |
| Fail-safe | Unit test catches any forgotten registrations |
| Thread-safe | `std::call_once` guarantees single initialization |

## Open Questions (Updated)

1. ~~**Enum handling**~~: Resolved - use `P_ENUM` variant or `EnumValidator`

2. ~~**Range validation**~~: Resolved - validators run in PipelineValidator phase

3. ~~**Dynamic properties**~~: Resolved - use `P(type, name, Dynamic, ...)` syntax

4. ~~**Static library linking**~~: Resolved - central registration file + lazy init

5. ~~**Forgotten registrations**~~: Resolved - CMake scanner + unit test detection

6. **Thread safety**: Should setProperty be thread-safe for dynamic properties?
   - Proposal: Module implementation responsibility (use mutex if needed)

7. **Change batching**: Should multiple property changes be atomic?
   - Proposal: Add `setProperties(map<string, value>)` for batch updates

## Implementation Tasks

### Phase 1: Property Binding Infrastructure

| Task | File | Description |
|------|------|-------------|
| D2.1 | `declarative/PropertyMacros.h` | Create DECLARE_PROPS macro with X-Macro pattern |
| D2.2 | `declarative/PropertyMacros.h` | Add applyProp, toPropertyValue, applyFromVariant helpers |
| D2.3 | `declarative/PropertyMacros.h` | Generate getProperty/setProperty/dynamicPropertyNames |
| D2.4 | `test/declarative/property_macros_tests.cpp` | Unit tests for property binding |

### Phase 2: Module Registration Builder

| Task | File | Description |
|------|------|-------------|
| D2.5 | `declarative/ModuleRegistrationBuilder.h` | Fluent builder with category/description/tags/input/output |
| D2.6 | `declarative/ModuleRegistrations.h` | Header with `ensureBuiltinModulesRegistered()` declaration |
| D2.7 | `declarative/ModuleRegistrations.cpp` | Central registration file (start with 5 pilot modules) |
| D2.8 | `declarative/TomlParser.cpp` | Add `ensureBuiltinModulesRegistered()` call |
| D2.9 | `declarative/ModuleFactory.cpp` | Add `ensureBuiltinModulesRegistered()` call |

### Phase 3: Missing Registration Detection

| Task | File | Description |
|------|------|-------------|
| D2.10 | `cmake/ScanModuleClasses.cmake` | CMake script to scan for Module subclasses |
| D2.11 | `CMakeLists.txt` | Add custom target to generate `module_subclasses.inc` |
| D2.12 | `test/declarative/module_registration_tests.cpp` | Test that fails if modules are unregistered |

### Phase 4: Migrate Pilot Modules

| Task | File | Description |
|------|------|-------------|
| D2.13 | `FileWriterModule.h` | Add DECLARE_PROPS to FileWriterModuleProps |
| D2.14 | `FaceDetectorXform.h` | Add DECLARE_PROPS with Dynamic properties |
| D2.15 | Remove old `REGISTER_MODULE` | Delete static registration from pilot module .cpp files |

### Phase 5: Complete Module Registration

| Task | File | Description |
|------|------|-------------|
| D2.16 | `ModuleRegistrations.cpp` | Register all 60+ modules |
| D2.17 | Update exclusion list | Add abstract base classes to test exclusions |

## File Summary

```
base/include/declarative/
  PropertyMacros.h              # NEW - DECLARE_PROPS macro
  ModuleRegistrationBuilder.h   # NEW - Fluent builder
  ModuleRegistrations.h         # NEW - ensureBuiltinModulesRegistered()

base/src/declarative/
  ModuleRegistrations.cpp       # NEW - Central registration file
  TomlParser.cpp                # MODIFY - Add registration call
  ModuleFactory.cpp             # MODIFY - Add registration call

base/test/declarative/
  property_macros_tests.cpp     # NEW - Property binding tests
  module_registration_tests.cpp # NEW - Missing registration detection

cmake/
  ScanModuleClasses.cmake       # NEW - Header scanner script

base/CMakeLists.txt             # MODIFY - Add scanner target
```

## Acceptance Criteria (Updated)

### Core Property Binding
- [ ] X-Macro pattern works with DECLARE_PROPS
- [ ] Properties of all types work: string, bool, int, float
- [ ] Required properties throw when missing
- [ ] Optional properties use defaults when not specified
- [ ] Type conversion works (int64 → int, double → float)

### Runtime Property Access
- [ ] `getProperty(name)` returns current value as variant
- [ ] `setProperty(name, value)` updates dynamic properties
- [ ] `setProperty` throws for static properties
- [ ] `dynamicPropertyNames()` returns list of modifiable properties

### Module Registration
- [ ] Fluent builder syntax works: `registerModule<T, Props>().category().description()...`
- [ ] `ensureBuiltinModulesRegistered()` registers all modules on first call
- [ ] Thread-safe via `std::call_once`
- [ ] No linker issues with static library

### Missing Registration Detection
- [ ] CMake generates `module_subclasses.inc` at configure time
- [ ] Unit test fails with helpful message if module not registered
- [ ] Test lists exact fix (add to registrations or exclusions)

### Testing
- [ ] Unit tests for property macros
- [ ] Unit tests for registration builder
- [ ] Integration test: TOML → Parse → Build → Properties applied correctly
- [ ] 2 pilot modules converted (FileWriterModule, FaceDetectorXform)

## Definition of Done

- [ ] All Phase 1-3 tasks complete
- [ ] At least 5 pilot modules registered
- [ ] All unit tests passing
- [ ] CMake scanner working
- [ ] Missing registration test working
- [ ] Documentation updated
