# CUDA Memory Type Auto-Bridging Design

> Design document for automatic memory type handling in the declarative pipeline framework.

**Status**: Design Phase
**Author**: Claude Code + Akhil
**Date**: 2026-01-12
**Branch**: `feat-declarative-pipeline-v2`

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Existing Infrastructure](#existing-infrastructure)
3. [Solution Overview](#solution-overview)
4. [Per-PIN Memory Type Registration](#per-pin-memory-type-registration)
5. [Functional Tags for Module Equivalence](#functional-tags-for-module-equivalence)
6. [Expert Validator Behavior](#expert-validator-behavior)
7. [Auto-Bridge Insertion](#auto-bridge-insertion)
8. [Edge Cases](#edge-cases)
9. [Implementation Plan](#implementation-plan)
10. [Testing Strategy](#testing-strategy)
11. [Future: LLM Pipeline Building](#future-llm-pipeline-building)

---

## Problem Statement

### Current Failure

When a user creates a CUDA pipeline declaratively:

```json
{
  "modules": {
    "color": { "type": "ColorConversion" },
    "blur": { "type": "GaussianBlur", "props": { "kernelSize": 15 } },
    "encoder": { "type": "ImageEncoderCV" }
  },
  "connections": [
    { "from": "color", "to": "blur" },
    { "from": "blur", "to": "encoder" }
  ]
}
```

The pipeline fails at init time:

```
GaussianBlur: input memType is expected to be CUDA_DEVICE. Actual<1>
```

### Root Cause

1. CUDA modules validate input memory type at `init()` time
2. At init time, upstream module metadata hasn't propagated memory type
3. GaussianBlur sees default HOST (1) instead of CUDA_DEVICE (3)
4. Validation fails before any frames flow

### Why Imperative Code Works

In imperative C++, the programmer explicitly manages memory transfers:

```cpp
auto cudaMemCopy = boost::shared_ptr<Module>(
    new CudaMemCopy(CudaMemCopyProps(cudaMemcpy::HostToDevice), cudaStream));
auto gaussianBlur = boost::shared_ptr<Module>(
    new GaussianBlur(props, cudaStream));
cudaMemCopy->setNext(gaussianBlur);
```

The declarative framework should handle this automatically.

---

## Existing Infrastructure

### MemType Enum

Located in `FrameMetadata.h:57`:

```cpp
enum MemType
{
    HOST = 1,          // Regular CPU memory
    HOST_PINNED = 2,   // Page-locked CPU memory (faster CUDA transfers)
    CUDA_DEVICE = 3,   // CUDA GPU memory
    DMABUF = 4         // DMA buffer (Linux/Jetson)
};
```

**This existing enum should be used directly by the declarative framework.**

### Future MemType Extensions

When adding new GPU backends:

```cpp
enum MemType
{
    HOST = 1,
    HOST_PINNED = 2,
    CUDA_DEVICE = 3,
    DMABUF = 4,
    // Future backends:
    METAL = 5,         // Apple Metal (MTLBuffer)
    VULKAN = 6,        // Vulkan (VkDeviceMemory)
    IMSDK = 7,         // Intel Media SDK (mfxFrameSurface1)
    OPENCL = 8         // OpenCL (cl_mem)
};
```

### Shared cudastream

`ModuleFactory` already creates a shared `cudastream_sp` for all CUDA modules. This infrastructure remains unchanged.

---

## Solution Overview

### Philosophy

**Old approach**: User must manually place CudaMemCopy modules
**New approach**: Framework automatically bridges memory type gaps

CudaMemCopy becomes an **internal mechanism**, not a user-facing module.

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DECLARATIVE PIPELINE BUILD                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. PARSE JSON                                                       │
│     └── Extract modules and connections                              │
│                                                                      │
│  2. CREATE MODULES                                                   │
│     └── Instantiate all user-defined modules                         │
│                                                                      │
│  3. MEMORY TYPE ANALYSIS  ◄── NEW                                    │
│     ├── Query each module's PIN memtypes from registry               │
│     ├── Build memtype flow graph                                     │
│     └── Identify mismatches                                          │
│                                                                      │
│  4. AUTO-BRIDGE INSERTION  ◄── NEW                                   │
│     ├── For each HOST→CUDA_DEVICE mismatch: insert CudaMemCopy(H2D)  │
│     ├── For each CUDA_DEVICE→HOST mismatch: insert CudaMemCopy(D2H)  │
│     ├── Use shared cudastream for all CUDA modules                   │
│     └── Warn about suboptimal patterns                               │
│                                                                      │
│  5. CONNECT MODULES                                                  │
│     └── setNext() with proper connections (including bridges)        │
│                                                                      │
│  6. INIT PIPELINE                                                    │
│     └── module->init() now succeeds (memtypes are correct)           │
│                                                                      │
│  7. REPORT TO USER                                                   │
│     ├── Show optimized pipeline                                      │
│     ├── List auto-inserted modules                                   │
│     └── Suggest optimizations                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Per-PIN Memory Type Registration

### Model

Every module PIN has a memory type:
- **Input PIN**: expects exactly one memtype
- **Output PIN**: produces exactly one memtype
- **Default**: HOST (when not specified)

Since no PIN accepts multiple memory types, this is a simple 1:1 mapping.

### Registration API Extension

```cpp
// In Metadata.h - extend PinDef
struct PinDef {
    std::string name;
    std::vector<FrameMetadata::FrameType> types;
    FrameMetadata::MemType memType = FrameMetadata::HOST;  // NEW
    bool required = true;
};
```

### Registration Examples

```cpp
// CPU-only module (default HOST, can omit memType)
ModuleRegistrationBuilder<ColorConversion>("ColorConversion")
    .inputPin("input", FrameType::RAW_IMAGE)      // implicitly HOST
    .outputPin("output", FrameType::RAW_IMAGE)    // implicitly HOST
    ...

// CUDA module - must declare CUDA_DEVICE
ModuleRegistrationBuilder<GaussianBlur>("GaussianBlur")
    .inputPin("input", FrameType::RAW_IMAGE, FrameMetadata::CUDA_DEVICE)
    .outputPin("output", FrameType::RAW_IMAGE, FrameMetadata::CUDA_DEVICE)
    ...

// Memory transfer module - output depends on props
ModuleRegistrationBuilder<CudaMemCopy>("CudaMemCopy")
    .inputPin("input", FrameType::RAW_IMAGE)  // accepts HOST or CUDA_DEVICE
    .dynamicOutputMemType([](const auto& props) {
        return props.kind == "HostToDevice"
            ? FrameMetadata::CUDA_DEVICE
            : FrameMetadata::HOST;
    })
    ...
```

### Module Classification by MemType

| Category | Input | Output | Examples |
|----------|-------|--------|----------|
| CPU-only | HOST | HOST | ColorConversion, ImageResizeCV, ImageEncoderCV |
| CUDA-only | CUDA_DEVICE | CUDA_DEVICE | GaussianBlur, ResizeNPPI, EffectsNPPI |
| Uploader | HOST | CUDA_DEVICE | CudaMemCopy(H2D) |
| Downloader | CUDA_DEVICE | HOST | CudaMemCopy(D2H) |
| DMA | DMABUF | DMABUF | NvArgusCamera, JPEGEncoderL4TM |

---

## Functional Tags for Module Equivalence

### Problem

When a CPU module breaks a GPU chain, the validator should suggest a GPU alternative. But how does it know that `ResizeNPPI` is equivalent to `ImageResizeCV`?

### Solution: Functional Tags

```cpp
ModuleRegistrationBuilder<ImageResizeCV>("ImageResizeCV")
    .tag("function:resize")
    .tag("backend:cpu")
    ...

ModuleRegistrationBuilder<ResizeNPPI>("ResizeNPPI")
    .tag("function:resize")
    .tag("backend:cuda")
    ...
```

### Tag Taxonomy

```cpp
// Functional tags - WHAT it does
.tag("function:resize")
.tag("function:blur")
.tag("function:encode")
.tag("function:decode")
.tag("function:color-convert")
.tag("function:overlay")
.tag("function:detect-faces")
.tag("function:rotate")

// Backend tags - WHERE it runs
.tag("backend:cpu")
.tag("backend:cuda")
.tag("backend:v4l2")
.tag("backend:jetson")
.tag("backend:nvjpeg")

// Media tags - WHAT media type
.tag("media:image")
.tag("media:video")
.tag("media:audio")

// Format tags - specific formats
.tag("format:jpeg")
.tag("format:h264")
.tag("format:h265")
.tag("format:mp4")
.tag("format:raw")
```

### Validator Query Logic

```cpp
// When CPU module breaks GPU chain:
std::string function = extractTag(module, "function:");  // e.g., "resize"
auto alternatives = registry.query({
    {"function:" + function},
    {"backend:cuda"}
});
if (!alternatives.empty()) {
    suggestAlternative(module, alternatives[0]);
}
```

### Complete Module Tag Examples

```cpp
// Image resize - CPU
ModuleRegistrationBuilder<ImageResizeCV>("ImageResizeCV")
    .tag("function:resize")
    .tag("backend:cpu")
    .tag("media:image")
    .inputPin("input", FrameType::RAW_IMAGE, FrameMetadata::HOST)
    .outputPin("output", FrameType::RAW_IMAGE, FrameMetadata::HOST)
    ...

// Image resize - CUDA
ModuleRegistrationBuilder<ResizeNPPI>("ResizeNPPI")
    .tag("function:resize")
    .tag("backend:cuda")
    .tag("media:image")
    .inputPin("input", FrameType::RAW_IMAGE, FrameMetadata::CUDA_DEVICE)
    .outputPin("output", FrameType::RAW_IMAGE, FrameMetadata::CUDA_DEVICE)
    ...

// JPEG encode - CPU
ModuleRegistrationBuilder<ImageEncoderCV>("ImageEncoderCV")
    .tag("function:encode")
    .tag("backend:cpu")
    .tag("format:jpeg")
    .tag("format:png")
    .inputPin("input", FrameType::RAW_IMAGE, FrameMetadata::HOST)
    .outputPin("output", FrameType::ENCODED_IMAGE, FrameMetadata::HOST)
    ...

// JPEG encode - CUDA
ModuleRegistrationBuilder<JPEGEncoderNVJPEG>("JPEGEncoderNVJPEG")
    .tag("function:encode")
    .tag("backend:cuda")
    .tag("backend:nvjpeg")
    .tag("format:jpeg")
    .inputPin("input", FrameType::RAW_IMAGE, FrameMetadata::CUDA_DEVICE)
    .outputPin("output", FrameType::ENCODED_IMAGE, FrameMetadata::HOST)  // Note: outputs to HOST
    ...
```

---

## Expert Validator Behavior

### Example 1: Simple CUDA Pipeline

**User Input:**
```json
{
  "modules": {
    "generator": { "type": "TestSignalGenerator" },
    "blur": { "type": "GaussianBlur" },
    "encoder": { "type": "ImageEncoderCV" }
  }
}
```

**Analysis:**
```
generator    HOST         → blur     CUDA_DEVICE   ✗ MISMATCH
blur         CUDA_DEVICE  → encoder  HOST          ✗ MISMATCH
```

**Validator Output:**
```
[INFO] Building pipeline...
[INFO] Memory type mismatch: generator (HOST) → blur (CUDA_DEVICE)
[INFO] Auto-inserting CudaMemCopy (HostToDevice) between 'generator' and 'blur'
[INFO] Memory type mismatch: blur (CUDA_DEVICE) → encoder (HOST)
[INFO] Auto-inserting CudaMemCopy (DeviceToHost) between 'blur' and 'encoder'
[INFO]
[INFO] Final pipeline:
       generator → [H2D] → blur → [D2H] → encoder
                   ~~~~          ~~~~
                   (auto)        (auto)
```

### Example 2: Consecutive CUDA Modules

**User Input:**
```json
{
  "modules": {
    "color": { "type": "ColorConversion" },
    "blur": { "type": "GaussianBlur" },
    "resize": { "type": "ResizeNPPI" },
    "effects": { "type": "EffectsNPPI" },
    "encoder": { "type": "ImageEncoderCV" }
  }
}
```

**Analysis:**
```
color    HOST         → blur     CUDA_DEVICE   ✗ Need H2D
blur     CUDA_DEVICE  → resize   CUDA_DEVICE   ✓ OK
resize   CUDA_DEVICE  → effects  CUDA_DEVICE   ✓ OK
effects  CUDA_DEVICE  → encoder  HOST          ✗ Need D2H
```

**Validator Output:**
```
[INFO] Final pipeline:
       color → [H2D] → blur → resize → effects → [D2H] → encoder
               ~~~~                               ~~~~

       Memory: HOST → CUDA_DEVICE → CUDA_DEVICE → CUDA_DEVICE → CUDA_DEVICE → HOST
                      └──────────────── GPU BLOCK ────────────────┘
```

### Example 3: Suboptimal Pattern Warning

**User Input:**
```json
{
  "modules": {
    "blur": { "type": "GaussianBlur" },
    "resize": { "type": "ImageResizeCV" },
    "effects": { "type": "EffectsNPPI" }
  }
}
```

**Validator Output:**
```
[WARNING] Suboptimal memory pattern detected!

  Current flow requires GPU → CPU → GPU roundtrip:
  blur (CUDA_DEVICE) → [D2H] → resize (HOST) → [H2D] → effects (CUDA_DEVICE)

  This adds 2 unnecessary memory transfers.

  Suggestion: Use 'ResizeNPPI' instead of 'ImageResizeCV' to keep processing on GPU:
  blur → ResizeNPPI → effects
         └─── continuous GPU ───┘

[INFO] Proceeding with user's pipeline (with auto-inserted bridges)...
```

---

## Auto-Bridge Insertion

### Bridge Module Selection

| From MemType | To MemType | Bridge Module | Props |
|--------------|------------|---------------|-------|
| HOST | CUDA_DEVICE | CudaMemCopy | kind: "HostToDevice" |
| CUDA_DEVICE | HOST | CudaMemCopy | kind: "DeviceToHost" |
| HOST | DMABUF | MemTypeConversion | — |
| DMABUF | HOST | MemTypeConversion | — |
| CUDA_DEVICE | DMABUF | *not supported* | error |
| DMABUF | CUDA_DEVICE | *not supported* | error |

### Bridge Naming Convention

Auto-inserted modules get generated names:
- `_bridge_H2D_color_blur` (between color and blur, HostToDevice)
- `_bridge_D2H_effects_encoder` (between effects and encoder, DeviceToHost)

The `_` prefix indicates framework-generated modules.

### cudastream Sharing

All CUDA modules (user-defined and auto-inserted) share the same `cudastream_sp`:

```cpp
// In ModuleFactory
cudastream_sp sharedStream = createCudaStream();

// When creating any CUDA module
if (moduleInfo.requiresCudaStream()) {
    module = moduleInfo.cudaFactory(props, sharedStream);
}

// Auto-inserted bridges also use shared stream
auto bridge = CudaMemCopy(bridgeProps, sharedStream);
```

---

## Edge Cases

### 1. User Manually Specifies CudaMemCopy

If user explicitly includes CudaMemCopy, the framework respects it:

```json
{
  "modules": {
    "upload": { "type": "CudaMemCopy", "props": { "kind": "HostToDevice" } },
    "blur": { "type": "GaussianBlur" }
  }
}
```

Validator sees: upload outputs CUDA_DEVICE, blur expects CUDA_DEVICE → compatible, no insertion.

### 2. Redundant Transfers

```json
{
  "connections": [
    { "from": "h2d", "to": "d2h" },
    { "from": "d2h", "to": "h2d_2" }
  ]
}
```

```
[WARNING] Redundant memory transfers detected:
  h2d (CUDA_DEVICE) → d2h (HOST) → h2d_2 (CUDA_DEVICE)

  This copies: GPU → CPU → GPU (wasteful)
  Consider removing intermediate transfers or restructuring pipeline.
```

### 3. No CUDA Modules

If pipeline has no CUDA modules, no analysis needed:
```
color → resize_cv → encoder → writer
HOST  → HOST     → HOST    → HOST     ✓ All compatible (no bridges needed)
```

### 4. Unsupported MemType Transition

```
[ERROR] Cannot bridge CUDA_DEVICE → DMABUF directly.
        No bridge module available for this transition.

        Possible workaround: Route through HOST:
        CUDA_DEVICE → HOST → DMABUF
```

### 5. HOST_PINNED Optimization Hint

```
[INFO] Performance tip: Module 'source' frequently transfers to GPU.
       Consider using HOST_PINNED memory for faster transfers.
       Set "usePinnedMemory": true in module props.
```

---

## Implementation Plan

### Phase 1: PIN MemType Registration

**Files to modify:**
- `base/include/declarative/Metadata.h` - Add memType to PinDef
- `base/src/declarative/ModuleRegistrations.cpp` - Update all registrations

**Tasks:**
1. Add `FrameMetadata::MemType memType` field to `PinDef` struct
2. Add `.inputMemType()` and `.outputMemType()` to registration builder
3. Default to HOST when not specified
4. Update CUDA module registrations to declare CUDA_DEVICE
5. Update DMA module registrations to declare DMABUF

**Acceptance Criteria:**
- All modules have correct memType declarations
- Unit tests verify memType is stored correctly
- Backward compatible (HOST default)

### Phase 2: Functional Tags

**Files to modify:**
- `base/src/declarative/ModuleRegistrations.cpp` - Add tags

**Tasks:**
1. Define tag taxonomy (function, backend, media, format)
2. Add tags to all module registrations
3. Add registry query methods: `getModulesByTag()`, `getModulesWithTags()`
4. Unit tests for tag queries

**Acceptance Criteria:**
- All modules have appropriate tags
- Can query "all resize modules" and get ImageResizeCV + ResizeNPPI
- Can query "cuda + resize" and get only ResizeNPPI

### Phase 3: Memory Type Analyzer

**Files to create:**
- `base/include/declarative/MemTypeAnalyzer.h`
- `base/src/declarative/MemTypeAnalyzer.cpp`

**Tasks:**
1. Build memtype flow graph from connections
2. Query registry for each module's PIN memtypes
3. Identify all mismatches
4. Return list of required bridges

**Acceptance Criteria:**
- Correctly identifies HOST→CUDA_DEVICE mismatches
- Correctly identifies CUDA_DEVICE→HOST mismatches
- Handles chains of CUDA modules (no false positives)
- Unit tests for various pipeline topologies

### Phase 4: Auto-Bridge Insertion

**Files to modify:**
- `base/src/declarative/ModuleFactory.cpp`

**Tasks:**
1. After module creation, run MemTypeAnalyzer
2. For each mismatch, create bridge module
3. Update connection graph to include bridges
4. Ensure bridges use shared cudastream
5. Generate meaningful bridge names

**Acceptance Criteria:**
- CUDA pipelines work without manual CudaMemCopy
- Bridges share cudastream with other CUDA modules
- Pipeline init succeeds

### Phase 5: User Feedback & Suggestions

**Files to modify:**
- `base/src/declarative/ModuleFactory.cpp`
- `base/include/declarative/BuildResult.h`

**Tasks:**
1. Log auto-inserted bridges as INFO messages
2. Detect suboptimal patterns (CPU in GPU chain)
3. Query registry for GPU alternatives using tags
4. Add suggestions to BuildResult
5. Optionally print final pipeline diagram

**Acceptance Criteria:**
- User sees what bridges were auto-inserted
- Warnings for suboptimal patterns include suggestions
- Suggestions reference actual alternative modules

---

## Testing Strategy

### Unit Tests

**MemType Registration Tests:**
```cpp
BOOST_AUTO_TEST_CASE(PinDef_DefaultMemType_IsHOST) {
    PinDef pin{"input", {FrameType::RAW_IMAGE}};
    BOOST_CHECK_EQUAL(pin.memType, FrameMetadata::HOST);
}

BOOST_AUTO_TEST_CASE(GaussianBlur_InputMemType_IsCUDA_DEVICE) {
    auto* info = registry.getModule("GaussianBlur");
    BOOST_CHECK_EQUAL(info->inputs[0].memType, FrameMetadata::CUDA_DEVICE);
}
```

**Tag Query Tests:**
```cpp
BOOST_AUTO_TEST_CASE(Query_ResizeModules_ReturnsBothCPUAndCUDA) {
    auto modules = registry.getModulesByTag("function:resize");
    BOOST_CHECK(contains(modules, "ImageResizeCV"));
    BOOST_CHECK(contains(modules, "ResizeNPPI"));
}

BOOST_AUTO_TEST_CASE(Query_CUDAResize_ReturnsOnlyResizeNPPI) {
    auto modules = registry.getModulesWithTags({"function:resize", "backend:cuda"});
    BOOST_CHECK_EQUAL(modules.size(), 1);
    BOOST_CHECK_EQUAL(modules[0], "ResizeNPPI");
}
```

**MemType Analyzer Tests:**
```cpp
BOOST_AUTO_TEST_CASE(Analyzer_HostToCUDA_RequiresBridge) {
    // ColorConversion (HOST) → GaussianBlur (CUDA_DEVICE)
    auto result = analyzer.analyze(connections);
    BOOST_CHECK_EQUAL(result.bridges.size(), 1);
    BOOST_CHECK_EQUAL(result.bridges[0].type, BridgeType::HOST_TO_CUDA);
}

BOOST_AUTO_TEST_CASE(Analyzer_CUDAToCUDA_NoBridge) {
    // GaussianBlur (CUDA) → ResizeNPPI (CUDA)
    auto result = analyzer.analyze(connections);
    BOOST_CHECK(result.bridges.empty());
}
```

**Integration Tests:**
```cpp
BOOST_AUTO_TEST_CASE(Factory_AutoInsertsBridges_PipelineInits) {
    auto json = R"({
        "modules": {
            "gen": {"type": "TestSignalGenerator"},
            "blur": {"type": "GaussianBlur"},
            "enc": {"type": "ImageEncoderCV"}
        },
        "connections": [
            {"from": "gen", "to": "blur"},
            {"from": "blur", "to": "enc"}
        ]
    })";

    auto result = factory.build(JsonParser::parse(json));
    BOOST_CHECK(result.success());
    BOOST_CHECK(result.pipeline->init());
}
```

### End-to-End Tests

**CUDA Pipeline Produces Output:**
```cpp
BOOST_AUTO_TEST_CASE(CUDAPipeline_ProducesJPEGFiles) {
    // Run pipeline with TestSignalGenerator → GaussianBlur → ImageEncoderCV → FileWriter
    // Verify JPEG files are created
    // Verify files are valid images
}
```

### CLI Tests

```bash
# Test verbose output shows auto-inserted bridges
./aprapipes_cli run cuda_pipeline.json -v 2>&1 | grep "Auto-inserting"

# Test suboptimal pattern warning
./aprapipes_cli run suboptimal_pipeline.json 2>&1 | grep "Suboptimal"
```

---

## Future: LLM Pipeline Building

### Why Tags Matter for LLMs

The tag system enables natural language queries:

| User Says | LLM Queries |
|-----------|-------------|
| "resize an image" | `function:resize` |
| "use GPU for speed" | `backend:cuda` |
| "encode to JPEG" | `function:encode`, `format:jpeg` |
| "GPU version of ImageResizeCV" | same `function:*`, `backend:cuda` |
| "read from camera" | `category:source`, `media:video` |

### LLM Pipeline Generation Flow

```
User: "Create a pipeline that reads video, resizes it on GPU, and saves as JPEG"

LLM:
1. Query: source + video → Mp4ReaderSource, RTSPClientSrc, WebCamSource
2. Query: resize + cuda → ResizeNPPI
3. Query: encode + jpeg → ImageEncoderCV, JPEGEncoderNVJPEG
4. Query: sink + file → FileWriterModule

LLM generates:
{
  "modules": {
    "source": {"type": "Mp4ReaderSource", "props": {"path": "..."}},
    "resize": {"type": "ResizeNPPI", "props": {"width": 640, "height": 480}},
    "encode": {"type": "ImageEncoderCV"},
    "writer": {"type": "FileWriterModule", "props": {"path": "..."}}
  }
}

Framework auto-inserts memory bridges as needed.
```

### Registry Query API for LLMs

```cpp
// Query modules by capability
std::vector<std::string> getModulesWithTags(
    const std::vector<std::string>& requiredTags,
    const std::vector<std::string>& excludeTags = {}
);

// Get module description for LLM context
ModuleDescription getModuleDescription(const std::string& name);
// Returns: name, description, props with types/defaults, input/output pins, tags

// Get all tags in use
std::set<std::string> getAllTags();

// Get modules by category with optional backend filter
std::vector<std::string> getSourceModules(const std::string& backend = "");
std::vector<std::string> getTransformModules(const std::string& backend = "");
std::vector<std::string> getSinkModules(const std::string& backend = "");
```

---

## Summary

This design transforms the declarative framework from a passive builder into an intelligent assistant that:

1. **Understands** memory requirements via per-PIN memType registration
2. **Classifies** modules via functional tags (function, backend, media, format)
3. **Detects** memory type incompatibilities automatically
4. **Resolves** them by auto-inserting bridge modules (CudaMemCopy)
5. **Optimizes** by recognizing consecutive CUDA modules
6. **Educates** users about suboptimal patterns with actionable suggestions
7. **Enables** future LLM-based pipeline generation

The user experience changes from "must understand CUDA memory model" to "just pick the modules you want."
