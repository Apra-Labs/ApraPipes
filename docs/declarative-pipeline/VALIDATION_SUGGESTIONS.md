# Validation Suggestion System Design

## Problem Statement

When a user connects two modules with incompatible frame types, the validator currently just reports an error:

```
[ERROR] E304 @ connections: Incompatible frame types: RawImagePlanar → RawImage
```

This leaves users without guidance on how to fix the issue. The validator should suggest inserting conversion modules.

## Solution: Type Bridge Suggestions

### Concept

When detecting a type mismatch, the validator:
1. Queries the registry for modules that can **accept** the source output type
2. Filters to modules that can **produce** the destination input type
3. Suggests inserting these "bridge" modules

### Example

```json
{
  "connections": [
    { "from": "generator", "to": "virtualPTZ" }
  ]
}
```
Where generator outputs RawImagePlanar and virtualPTZ expects RawImage.

**Current output:**
```
[ERROR] E304 @ connections: Incompatible frame types: generator(RawImagePlanar) → virtualPTZ(RawImage)
```

**Proposed output:**
```
[ERROR] E304 @ connections: Incompatible frame types: generator(RawImagePlanar) → virtualPTZ(RawImage)
[SUGGESTION] Insert ColorConversion with conversionType="YUV420PLANAR_TO_RGB" between these modules:

  "modules": {
    "planar_to_rgb": {
      "type": "ColorConversion",
      "props": { "conversionType": "YUV420PLANAR_TO_RGB" }
    }
  },
  "connections": [
    { "from": "generator", "to": "planar_to_rgb" },
    { "from": "planar_to_rgb", "to": "virtualPTZ" }
  ]
```

## Implementation Design

### 1. Add Bridge Module Query to ModuleRegistry

```cpp
// ModuleRegistry.h
struct BridgeModule {
    std::string name;
    std::string inputType;
    std::string outputType;
    std::map<std::string, std::string> requiredProps;  // e.g., conversionType=YUV420PLANAR_TO_RGB
};

std::vector<BridgeModule> findBridgeModules(
    const std::string& sourceType,
    const std::string& destType
) const;
```

### 2. Extend ValidationIssue with Suggestions

```cpp
// PipelineValidator.h
struct ValidationIssue {
    // ... existing fields ...

    struct Suggestion {
        std::string description;
        std::string jsonSnippet;       // Ready-to-copy JSON
    };
    std::vector<Suggestion> suggestions;
};
```

### 3. Add Suggestion Generation to validateConnections()

```cpp
// PipelineValidator.cpp - in validateConnections()
if (!FrameTypeRegistry::instance().isCompatible(srcType, dstType)) {
    // Existing error
    result.issues.push_back(ValidationIssue::error(...));

    // NEW: Find bridge modules
    auto bridges = ModuleRegistry::instance().findBridgeModules(srcType, dstType);
    if (!bridges.empty()) {
        auto& issue = result.issues.back();
        for (const auto& bridge : bridges) {
            Suggestion s;
            s.description = "Insert " + bridge.name + " to convert " +
                            srcType + " → " + dstType;
            s.jsonSnippet = generateJsonSnippet(bridge, srcModule, dstModule);
            issue.suggestions.push_back(s);
        }
    }
}
```

### 4. Known Type Conversions Registry

Create a data structure mapping known type conversions:

```cpp
// FrameTypeConversions.h
struct TypeConversion {
    std::string fromType;
    std::string toType;
    std::string moduleName;
    std::map<std::string, std::string> props;
};

const std::vector<TypeConversion> KNOWN_CONVERSIONS = {
    // Planar to Packed conversions
    {"RawImagePlanar", "RawImage", "ColorConversion", {{"conversionType", "YUV420PLANAR_TO_RGB"}}},

    // Packed to Planar conversions
    {"RawImage", "RawImagePlanar", "ColorConversion", {{"conversionType", "RGB_TO_YUV420PLANAR"}}},

    // Decode conversions
    {"EncodedImage", "RawImage", "ImageDecoderCV", {}},
    {"H264Data", "RawImagePlanar", "H264Decoder", {}},

    // Encode conversions
    {"RawImage", "EncodedImage", "ImageEncoderCV", {}},
};
```

### 5. CLI Output Format

```
$ aprapipes_cli validate pipeline.json

[ERROR] E304 @ connections[0]: Incompatible frame types
  Source: generator (TestSignalGenerator) outputs RawImagePlanar
  Dest:   ptz (VirtualPTZ) expects RawImage

  SUGGESTION: Insert ColorConversion module
  ─────────────────────────────────────────
  Add this to your JSON:

  "modules": {
    "convert_planar_to_rgb": {
      "type": "ColorConversion",
      "props": { "conversionType": "YUV420PLANAR_TO_RGB" }
    }
  }

  And update connections:

  "connections": [
    { "from": "generator", "to": "convert_planar_to_rgb" },
    { "from": "convert_planar_to_rgb", "to": "ptz" }
  ]

✗ Validation failed with 1 error, 1 suggestion
```

## Implementation Phases

### Phase 1: Basic Suggestion Infrastructure
- [ ] Add `Suggestion` struct to ValidationIssue
- [ ] Add suggestion rendering to CLI output
- [ ] Add `--suggestions` flag (enabled by default)

### Phase 2: Known Conversions
- [ ] Create KNOWN_CONVERSIONS constant
- [ ] Implement lookup in validateConnections()
- [ ] Generate JSON snippets

### Phase 3: Dynamic Bridge Discovery
- [ ] Implement findBridgeModules() in ModuleRegistry
- [ ] Scan registered modules for compatible bridges
- [ ] Handle multi-hop conversions (A → B → C)

### Phase 4: JSON Output
- [ ] Include suggestions in `--json` output
- [ ] Machine-parseable format for IDE integration

## Benefits

1. **User-friendly**: Tells users exactly what to do
2. **Educational**: Teaches pipeline structure
3. **Reduces friction**: Copy-paste solution
4. **IDE-ready**: JSON format enables auto-fix features
