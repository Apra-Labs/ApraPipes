# D3: Multi-Pin Connection Support

> **Status:** Design Phase
> **Dependencies:** D1 (Module Factory), D2 (Property Binding)
> **Priority:** High (required for complex pipelines)

---

## 1. Problem Statement

The current declarative pipeline implementation only supports simple single-input, single-output modules. Real-world pipelines require:

1. **Multi-output modules** (e.g., H264Encoder outputs video + motion vectors)
2. **Multi-input modules** (e.g., OverlayNPPI overlays image on background)
3. **Selective pin connections** (connect specific outputs to specific inputs)
4. **Optional inputs** (some inputs are not required)

### Current Limitations

```toml
# Current: works for simple 1:1 connections
[[connections]]
from = "source.output"
to = "decoder.input"

# Problem: How do we connect encoder's motion_vectors output separately?
# Problem: How do we connect overlay's second input (foreground)?
```

---

## 2. ApraPipes Pin Model Analysis

### 2.1 Output Pins

**Created explicitly** by modules in their constructor or `setMetadata()`:

```cpp
// Single output (most modules)
mOutputPinId = addOutputPin(metadata);

// Multiple outputs (H264EncoderV4L2)
h264FrameOutputPinId = addOutputPin(mOutputMetadata);
motionVectorFramePinId = addOutputPin(motionVectorOutputMetadata);
```

**Pin IDs are auto-generated** as `{ModuleId}_pin_{N}` (e.g., `H264EncoderV4L2_1_pin_1`).

### 2.2 Input Pins

**Created implicitly** when `setNext()` is called:

```cpp
// Connects ALL output pins of source to dest
source->setNext(dest);

// Connects SPECIFIC pins only
vector<string> pins = {"encoder_1_pin_1"};
source->setNext(dest, pins);
```

### 2.3 Multi-Input Module Examples

| Module | Input Count | Input Types | Notes |
|--------|-------------|-------------|-------|
| FramesMuxer | 2+ | Any | Synchronizes multiple streams |
| HistogramOverlay | 2 | RAW_IMAGE + ARRAY | Image + histogram data |
| OverlayNPPI | 2 | RAW_IMAGE + OVERLAY_INFO | Background + overlay info |

### 2.4 Multi-Output Module Examples

| Module | Output Count | Output Types | Notes |
|--------|--------------|--------------|-------|
| H264EncoderV4L2 | 2 | H264_DATA + OVERLAY_INFO | Video + motion vectors |
| Mp4ReaderSource | 3 | ENCODED_IMAGE, H264_DATA, MP4_VIDEO_METADATA | Multiple media types |
| Demuxer (future) | N | Various | Dynamic output count |

---

## 3. Design Options

### Option A: Implicit Pin Resolution (Current Approach)

**How it works:**
- Use first declared output pin for single-output modules
- Require explicit pin names only for multi-output modules

**TOML syntax:**
```toml
# Single output - pin name optional (uses first output)
[[connections]]
from = "source.output"
to = "decoder.input"

# Multi-output - pin name required
[[connections]]
from = "encoder.h264"
to = "writer.input"

[[connections]]
from = "encoder.motion_vectors"
to = "analyzer.input"
```

**Pros:**
- Simple for common cases
- Backwards compatible

**Cons:**
- Inconsistent syntax (sometimes required, sometimes not)
- Mapping from TOML names to internal IDs needs maintenance

---

### Option B: Explicit Pin Mapping in Registry

**How it works:**
- Registry declares pin names that map to internal pin order
- Pin names in TOML match registry declarations
- Factory creates output pins with predictable names

**Registry declaration:**
```cpp
registerModule<H264EncoderV4L2, H264EncoderV4L2Props>()
    .category(ModuleCategory::Transform)
    .input("raw_frame", "RawImage", "RawImagePlanar")  // Input 0
    .output("h264", "H264Frame")                        // Output 0
    .output("motion_vectors", "OverlayInfo");           // Output 1
```

**TOML syntax:**
```toml
[[connections]]
from = "encoder.h264"           # Maps to output 0
to = "writer.input"

[[connections]]
from = "encoder.motion_vectors" # Maps to output 1
to = "analyzer.input"
```

**Internal mapping:**
```cpp
// Registry stores: {"h264" → 0, "motion_vectors" → 1}
// Factory creates pins in order, stores mapping
// Connection uses: pinNameToId["h264"] → "H264EncoderV4L2_1_pin_1"
```

**Pros:**
- Consistent naming in TOML
- Type-safe (pin types declared in registry)
- Self-documenting (registry shows all pins)

**Cons:**
- Requires updating all module registrations
- Need to keep registry and module impl in sync

---

### Option C: Named Pin Creation (Invasive)

**How it works:**
- Modify modules to use explicit pin names
- `addOutputPin(metadata, "h264")` instead of auto-generated

**Pros:**
- Direct name-to-ID mapping
- No translation layer

**Cons:**
- Requires modifying existing modules
- Breaking change to module API

---

### Option D: Hybrid Approach (Recommended)

**How it works:**
1. Registry declares logical pin names and order
2. Factory creates pins in registry-declared order
3. Factory maintains `pinName → pinId` mapping for each instance
4. Connection resolution uses mapping

**Benefits:**
- No changes to module implementations
- Clear TOML syntax
- Type information available for validation

---

## 4. Recommended Design: Option D (Hybrid)

### 4.1 Registry Enhancement

Current:
```cpp
registerModule<H264EncoderV4L2, H264EncoderV4L2Props>()
    .output("h264", "H264Frame")
    .output("motion_vectors", "OverlayInfo");
```

The `.output()` calls already exist and declare:
- Pin name (for TOML)
- Frame type(s) (for validation and metadata creation)

### 4.2 Factory Changes

**During module creation:**
```cpp
// Store mapping for each module instance
struct ModuleContext {
    boost::shared_ptr<Module> module;
    std::map<std::string, std::string> outputPinMap;  // "h264" → "H264EncoderV4L2_1_pin_1"
    std::map<std::string, std::string> inputPinMap;   // "input" → actual input pin ID
};

std::map<std::string, ModuleContext> moduleContextMap;
```

**When creating output pins:**
```cpp
void setupOutputPins(Module* module, const ModuleInfo& info, ModuleContext& ctx) {
    for (size_t i = 0; i < info.outputs.size(); ++i) {
        const auto& outputPin = info.outputs[i];

        // Create metadata from frame type
        auto metadata = createMetadata(outputPin.frame_types[0]);

        // Add pin and capture the generated ID
        std::string generatedPinId = module->addOutputPin(metadata);

        // Store mapping: TOML name → internal ID
        ctx.outputPinMap[outputPin.name] = generatedPinId;
    }
}
```

### 4.3 Connection Resolution

**TOML syntax:**
```toml
[[connections]]
from = "encoder.h264"         # instance.pinName
to = "writer.input"
```

**Resolution logic:**
```cpp
bool connectModules(const Connection& conn) {
    // Parse "encoder.h264" → instance="encoder", pin="h264"
    auto [srcInstance, srcPin] = parseConnectionPoint(conn.from);
    auto [dstInstance, dstPin] = parseConnectionPoint(conn.to);

    // Get module contexts
    auto& srcCtx = moduleContextMap[srcInstance];
    auto& dstCtx = moduleContextMap[dstInstance];

    // Resolve pin names to internal IDs
    std::string srcPinId = srcCtx.outputPinMap[srcPin];

    // Connect using pin-specific setNext
    vector<string> pinIds = {srcPinId};
    return srcCtx.module->setNext(dstCtx.module, pinIds);
}
```

### 4.4 Multi-Input Handling

For modules with multiple inputs (e.g., OverlayNPPI):

**TOML:**
```toml
[modules.overlay]
type = "OverlayNPPI"

[[connections]]
from = "background.output"
to = "overlay.background"       # First input

[[connections]]
from = "detector.overlay_info"
to = "overlay.foreground"       # Second input
```

**Input pins are created by `setNext()`** - the order of connection calls determines input order. We can either:

1. **Order-based:** Process connections in TOML order (first connection = input 0)
2. **Name-based:** Store expected input names in registry, match during connection

**Recommendation:** Order-based for simplicity, with validation against registry.

### 4.5 Optional Input Handling

Registry already supports `.optionalInput()`:

```cpp
registerModule<SomeModule, SomeModuleProps>()
    .input("primary", "RawImage")
    .optionalInput("secondary", "RawImage");  // Not required
```

**Validation logic:**
```cpp
void validateConnections() {
    for (const auto& [name, ctx] : moduleContextMap) {
        const ModuleInfo* info = registry.getModule(ctx.moduleType);

        for (const auto& input : info->inputs) {
            if (input.required) {
                // Check if this input has a connection
                bool hasConnection = hasConnectionTo(name, input.name);
                if (!hasConnection) {
                    issues.push_back(BuildIssue::error(
                        MISSING_REQUIRED_CONNECTION,
                        "modules." + name,
                        "Required input '" + input.name + "' not connected"
                    ));
                }
            }
        }
    }
}
```

---

## 5. TOML Syntax Examples

### 5.1 Simple Pipeline (Current - No Change)

```toml
[modules.source]
type = "FileReaderModule"

[modules.writer]
type = "FileWriterModule"

[[connections]]
from = "source.output"
to = "writer.input"
```

### 5.2 Multi-Output: Video Encoder

```toml
[modules.capture]
type = "V4L2CameraSource"

[modules.encoder]
type = "H264EncoderV4L2"

[modules.writer]
type = "FileWriterModule"

[modules.motion_analyzer]
type = "MotionAnalyzer"

# Video frames → encoder
[[connections]]
from = "capture.output"
to = "encoder.input"

# Encoded video → file writer
[[connections]]
from = "encoder.h264"
to = "writer.input"

# Motion vectors → analyzer
[[connections]]
from = "encoder.motion_vectors"
to = "motion_analyzer.input"
```

### 5.3 Multi-Input: Overlay

```toml
[modules.camera]
type = "V4L2CameraSource"

[modules.face_detector]
type = "FaceDetectorXform"

[modules.overlay]
type = "OverlayNPPI"

[modules.display]
type = "ImageViewerModule"

# Camera → face detector
[[connections]]
from = "camera.output"
to = "face_detector.input"

# Camera output also goes to overlay background
[[connections]]
from = "camera.output"
to = "overlay.background"

# Face detector output goes to overlay foreground
[[connections]]
from = "face_detector.overlay_info"
to = "overlay.foreground"

# Overlay → display
[[connections]]
from = "overlay.output"
to = "display.input"
```

### 5.4 Muxer (N inputs)

```toml
[modules.camera1]
type = "V4L2CameraSource"

[modules.camera2]
type = "V4L2CameraSource"

[modules.camera3]
type = "V4L2CameraSource"

[modules.muxer]
type = "FramesMuxer"

[modules.writer]
type = "FileWriterModule"

# Multiple cameras → muxer (order determines input index)
[[connections]]
from = "camera1.output"
to = "muxer.input"

[[connections]]
from = "camera2.output"
to = "muxer.input"

[[connections]]
from = "camera3.output"
to = "muxer.input"

[[connections]]
from = "muxer.output"
to = "writer.input"
```

---

## 6. Frame Type Metadata

### 6.1 Problem

Currently we create generic `FrameMetadata(FrameType)`. For proper operation, some modules need specific metadata:

```cpp
// Generic (current)
auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::RAW_IMAGE));

// Specific (needed)
auto metadata = framemetadata_sp(new RawImageMetadata(1920, 1080, ImageMetadata::RGB, CV_8UC3));
```

### 6.2 Options

**Option A: Infer from file/property**
- FileReaderModule: detect from file
- CameraSource: detect from device

**Option B: Explicit in TOML**
```toml
[modules.source.outputs.output]
frame_type = "RawImage"
width = 1920
height = 1080
image_type = "RGB"
```

**Option C: Module-specific initialization**
- Let `init()` create proper metadata based on actual input

**Recommendation:** Start with generic metadata (current approach), let modules refine in `init()`. Add explicit TOML metadata as optional enhancement later.

---

## 7. Implementation Plan

### Phase 1: Core Multi-Pin Support

1. **Add pin mapping to ModuleFactory**
   - Create `ModuleContext` structure
   - Store `outputPinMap` during pin creation
   - Modify `connectModules()` to use pin names

2. **Update output pin creation**
   - Create pins in registry-declared order
   - Map TOML names to generated IDs

3. **Update connection resolution**
   - Parse `instance.pin` from connection points
   - Look up actual pin IDs from context map
   - Use `setNext(module, pinIdArr)` for specific pins

### Phase 2: Multi-Input Support

1. **Track input connections**
   - Store which inputs have been connected
   - Process connections in TOML order for input ordering

2. **Validate required inputs**
   - After all connections, check required inputs are satisfied

### Phase 3: Enhanced Validation

1. **Frame type compatibility**
   - Check output type matches input type
   - Warn on type mismatches

2. **Connection graph validation**
   - Detect cycles
   - Detect orphaned modules
   - Detect unconnected required inputs

---

## 8. Acceptance Criteria

- [ ] Multi-output modules work (H264EncoderV4L2)
- [ ] Multi-input modules work (OverlayNPPI, FramesMuxer)
- [ ] Pin names in TOML resolve correctly
- [ ] Required input validation works
- [ ] Optional inputs don't cause errors when unconnected
- [ ] Backwards compatible with existing simple pipelines
- [ ] Unit tests for all scenarios

---

## 9. Files to Modify

| File | Changes |
|------|---------|
| `ModuleFactory.h` | Add ModuleContext, pin mapping |
| `ModuleFactory.cpp` | Update connection resolution |
| `ModuleRegistry.h` | Already has pin info (no changes) |
| `PipelineValidator.cpp` | Add input connection validation |
| Unit tests | Add multi-pin test cases |

---

## 10. Open Questions

1. **Pin name conflicts:** What if two modules declare same pin name?
   - Answer: Pin names are scoped to module instance, so "encoder.output" and "decoder.output" are distinct.

2. **Dynamic pin count:** Some modules create pins based on input (e.g., Demuxer)?
   - Answer: Phase 2 consideration. May need runtime registration.

3. **Pin reordering:** If registry declares pins in different order than module creates?
   - Answer: Registry order is authoritative. Module must create in same order.
