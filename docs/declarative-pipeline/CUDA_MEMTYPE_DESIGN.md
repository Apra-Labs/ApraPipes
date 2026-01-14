# Automatic Bridging Design: Memory Types and Pixel Formats

> Design document for automatic memory type and pixel format handling in the declarative pipeline framework.

**Status**: Implemented (Sprint 7 Complete)
**Author**: Claude Code + Akhil
**Date**: 2026-01-12 (Design) / 2026-01-13 (Implementation Complete)
**Branch**: `feat-declarative-pipeline-v2`

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Existing Infrastructure](#existing-infrastructure)
3. [The Three Dimensions of PIN Compatibility](#the-three-dimensions-of-pin-compatibility)
4. [Solution Overview](#solution-overview)
5. [Per-PIN Memory Type Registration](#per-pin-memory-type-registration)
6. [Per-PIN Pixel Format Registration](#per-pin-pixel-format-registration)
7. [Functional Tags for Module Equivalence](#functional-tags-for-module-equivalence)
8. [Expert Validator Behavior](#expert-validator-behavior)
9. [Auto-Bridge Insertion](#auto-bridge-insertion)
10. [Edge Cases](#edge-cases)
11. [Implementation Plan](#implementation-plan)
12. [Testing Strategy](#testing-strategy)
13. [Future: LLM Pipeline Building](#future-llm-pipeline-building)

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

## The Three Dimensions of PIN Compatibility

Understanding why `frame_types` exists in PINInfo is key to the overall design.

### Dimension 1: Frame Type (Coarse-Grained)

**What it is**: The high-level data format category.

**Values**: `RAW_IMAGE`, `RAW_IMAGE_PLANAR`, `ENCODED_IMAGE`, `H264_DATA`, `AUDIO`, etc.

**Purpose**: Prevents obviously incompatible connections.
- Audio output → Video input = ERROR
- H264 data → Raw image processor = ERROR
- Encoded JPEG → Face detector expecting raw pixels = ERROR

**Already exists** in PINInfo as `frame_types: std::vector<std::string>`.

### Dimension 2: Pixel Format (Fine-Grained, within RAW_IMAGE)

**What it is**: The exact pixel arrangement for raw image data.

**Values** (from `ImageMetadata::ImageType`):
- `MONO` - Grayscale
- `BGR`, `BGRA`, `RGB`, `RGBA` - Packed color formats
- `YUV420`, `YUV411_I`, `YUV444` - Planar YUV formats
- `NV12` - Semi-planar (Y plane + interleaved UV)
- `UYVY`, `YUYV` - Packed YUV formats

**Purpose**: Enables automatic ColorConversion bridging.
- TestSignalGenerator outputs YUV420
- ResizeNPPI expects NV12
- Framework auto-inserts ColorConversion (or CCNPPI for GPU)

**Needs to be added** to PINInfo as `imageTypes: std::vector<ImageMetadata::ImageType>`.

### Dimension 3: Memory Type (Location)

**What it is**: Where the frame data resides in memory.

**Values** (from `FrameMetadata::MemType`):
- `HOST` - Regular CPU memory
- `HOST_PINNED` - Page-locked CPU memory
- `CUDA_DEVICE` - NVIDIA GPU memory
- `DMABUF` - DMA buffer (Linux/Jetson)

**Purpose**: Enables automatic CudaMemCopy bridging.
- ColorConversion outputs HOST
- GaussianBlur expects CUDA_DEVICE
- Framework auto-inserts CudaMemCopy(H2D)

**Needs to be added** to PINInfo as `memType: FrameMetadata::MemType`.

### The Complete PIN Model

```cpp
struct PinInfo {
    std::string name;                                    // "input", "output"

    // Dimension 1: Frame Type (coarse)
    std::vector<std::string> frame_types;               // ["RawImage", "RawImagePlanar"]

    // Dimension 2: Pixel Format (fine, for raw images only)
    std::vector<ImageMetadata::ImageType> imageTypes;   // [YUV420, NV12] - NEW

    // Dimension 3: Memory Location
    FrameMetadata::MemType memType = FrameMetadata::HOST;  // NEW

    bool required = true;
};
```

### Validation Hierarchy

The validator checks compatibility in order:

```
1. Frame Type Check
   └─ RAW_IMAGE → RAW_IMAGE ✓ (compatible high-level types)

2. Pixel Format Check (only for RAW_IMAGE types)
   └─ YUV420 → NV12 ✗ MISMATCH
   └─ Action: Auto-insert ColorConversion bridge

3. Memory Type Check
   └─ HOST → CUDA_DEVICE ✗ MISMATCH
   └─ Action: Auto-insert CudaMemCopy bridge
```

### Why This Matters

Users and LLMs shouldn't need to understand:
- That ResizeNPPI only accepts NV12 format
- That GaussianBlur requires CUDA_DEVICE memory
- The correct sequence of color conversion and memory transfers

The framework handles all of this automatically, making pipelines "just work."

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

## Per-PIN Pixel Format Registration

### Model

Every PIN that handles raw image data has a set of accepted pixel formats:
- **Input PIN**: accepts one or more imageTypes
- **Output PIN**: produces exactly one imageType (determined by props or input)
- **Default**: empty set (accepts any / not applicable for non-image types)

### Registration API Extension

```cpp
// In Metadata.h - extend PinDef further
struct PinDef {
    std::string name;
    std::vector<FrameMetadata::FrameType> types;
    FrameMetadata::MemType memType = FrameMetadata::HOST;
    std::vector<ImageMetadata::ImageType> imageTypes;  // NEW: [YUV420, NV12, RGB]
    bool required = true;
};
```

### Registration Examples

```cpp
// Module that only accepts YUV420 (common for video processing)
ModuleRegistrationBuilder<H264EncoderV4L2>("H264EncoderV4L2")
    .inputPin("input", FrameType::RAW_IMAGE, FrameMetadata::HOST, {ImageType::YUV420})
    .outputPin("output", FrameType::H264_DATA, FrameMetadata::HOST)
    ...

// Module that accepts multiple formats
ModuleRegistrationBuilder<ImageEncoderCV>("ImageEncoderCV")
    .inputPin("input", FrameType::RAW_IMAGE, FrameMetadata::HOST,
              {ImageType::BGR, ImageType::RGB, ImageType::MONO})
    .outputPin("output", FrameType::ENCODED_IMAGE, FrameMetadata::HOST)
    ...

// CUDA module with specific format requirement
ModuleRegistrationBuilder<ResizeNPPI>("ResizeNPPI")
    .inputPin("input", FrameType::RAW_IMAGE, FrameMetadata::CUDA_DEVICE,
              {ImageType::NV12, ImageType::YUV420})
    .outputPin("output", FrameType::RAW_IMAGE, FrameMetadata::CUDA_DEVICE)
    ...

// Color conversion module - input can be anything, output depends on props
ModuleRegistrationBuilder<ColorConversion>("ColorConversion")
    .inputPin("input", FrameType::RAW_IMAGE, FrameMetadata::HOST)  // any format
    .dynamicOutputImageType([](const auto& props) {
        return props.outFormat;  // BGR, RGB, YUV420, etc.
    })
    ...
```

### Module Classification by ImageType Support

| Category | Accepts | Examples |
|----------|---------|----------|
| Any format | {} (empty) | FileWriterModule, StatSink |
| BGR/RGB only | {BGR, RGB} | ImageEncoderCV (JPEG/PNG), FaceDetectorXform |
| YUV only | {YUV420, NV12} | H264EncoderV4L2, Mp4WriterSink |
| NV12 preferred | {NV12} | CUDA NPP modules |
| Format converter | any → specified | ColorConversion, CCNPPI |

### Common Pixel Format Mismatches

| Producer | Consumer | Auto-Bridge |
|----------|----------|-------------|
| TestSignalGenerator (YUV420) | ImageEncoderCV (BGR) | ColorConversion(YUV420→BGR) |
| ImageDecoderCV (BGR) | Mp4WriterSink (YUV420) | ColorConversion(BGR→YUV420) |
| WebCamSource (YUYV) | FaceDetectorXform (BGR) | ColorConversion(YUYV→BGR) |
| Mp4ReaderSource (NV12) | ImageEncoderCV (BGR) | ColorConversion(NV12→BGR) |

### Bridge Module Selection for Pixel Formats

| From Format | To Format | Bridge Module (HOST) | Bridge Module (CUDA) |
|-------------|-----------|---------------------|---------------------|
| YUV420 | BGR | ColorConversion | CCNPPI |
| BGR | YUV420 | ColorConversion | CCNPPI |
| NV12 | BGR | ColorConversion | CCNPPI |
| RGB | BGR | ColorConversion | CCNPPI |
| YUYV | BGR | ColorConversion | CCNPPI |
| Any | Any | ColorConversion | CCNPPI |

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

The expert validator performs three levels of optimization:
1. **Auto-insert bridges** for format and memory mismatches
2. **Suggest replacements** for better performance (module-level substitution)
3. **Warn about suboptimal patterns** (D→H→D roundtrips, etc.)

### Example 1: Simple CUDA Pipeline with Encoder Replacement Suggestion

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
generator    HOST         → blur     CUDA_DEVICE   ✗ MemType MISMATCH
blur         CUDA_DEVICE  → encoder  HOST          ✗ MemType MISMATCH
encoder accepts: {BGR, RGB}
blur outputs: YUV420 (inherited from generator)                ✗ ImageType MISMATCH
```

**Validator Output:**
```
[INFO] Building pipeline...

[INFO] Memory type mismatch: generator (HOST) → blur (CUDA_DEVICE)
       Auto-inserting CudaMemCopy (HostToDevice)

[INFO] Memory type mismatch: blur (CUDA_DEVICE) → encoder (HOST)
       Auto-inserting CudaMemCopy (DeviceToHost)

[INFO] Pixel format mismatch: blur (YUV420) → encoder (expects BGR/RGB)
       Auto-inserting ColorConversion (YUV420→BGR)

[SUGGEST] GPU-accelerated encoder available!
          Current:     ImageEncoderCV (CPU, requires D2H transfer)
          Alternative: JPEGEncoderNVJPEG (CUDA, no D2H needed)
          Benefit:     Eliminates 1 memory transfer, ~2x faster encoding

[INFO] Final pipeline (with auto-bridges):
       generator → [H2D] → blur → [D2H] → [YUV→BGR] → encoder
                   ~~~~          ~~~~     ~~~~~~~~~
                   (auto)        (auto)     (auto)

[INFO] Suggested pipeline (optimal):
       generator → [H2D] → blur → JPEGEncoderNVJPEG → FileWriter
                   ~~~~          └── stays on GPU! ──┘
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

### Example 3: Suboptimal Pattern Warning with Module Replacement

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

[SUGGEST] GPU equivalent module available!
          Current:     ImageResizeCV (HOST, function:resize, backend:cpu)
          Alternative: ResizeNPPI (CUDA_DEVICE, function:resize, backend:cuda)

          Replace 'resize' with:
          "resize": { "type": "ResizeNPPI", "props": { "width": 640, "height": 480 } }

          Result: blur → ResizeNPPI → effects
                        └─── continuous GPU ───┘

[INFO] Proceeding with user's pipeline (with auto-inserted bridges)...
```

### Example 4: Complex Pipeline with Combined Format + Memory Bridging

**User Input:**
```json
{
  "modules": {
    "camera": { "type": "WebCamSource" },
    "face": { "type": "FaceDetectorXform" },
    "blur": { "type": "GaussianBlur" },
    "encode": { "type": "ImageEncoderCV" },
    "writer": { "type": "FileWriterModule" }
  },
  "connections": [
    { "from": "camera", "to": "face" },
    { "from": "face", "to": "blur" },
    { "from": "blur", "to": "encode" },
    { "from": "encode", "to": "writer" }
  ]
}
```

**Analysis:**
```
camera (YUYV/HOST) → face (BGR/HOST)        ✗ ImageType MISMATCH
face (BGR/HOST) → blur (CUDA_DEVICE)        ✗ MemType MISMATCH
blur (BGR/CUDA) → encode (BGR/HOST)         ✗ MemType MISMATCH (format OK)
encode (JPEG/HOST) → writer (any/HOST)      ✓ OK
```

**Validator Output:**
```
[INFO] Building pipeline...

[INFO] Pixel format mismatch: camera (YUYV) → face (expects BGR)
       Auto-inserting ColorConversion (YUYV→BGR)

[INFO] Memory type mismatch: face (HOST) → blur (CUDA_DEVICE)
       Auto-inserting CudaMemCopy (HostToDevice)

[INFO] Memory type mismatch: blur (CUDA_DEVICE) → encode (HOST)
       Auto-inserting CudaMemCopy (DeviceToHost)

[SUGGEST] GPU-accelerated encoder available!
          Current:     ImageEncoderCV (CPU)
          Alternative: JPEGEncoderNVJPEG (CUDA)
          Note:        Would eliminate D2H transfer before encoding

[INFO] Final pipeline (with auto-bridges):
       camera → [YUYV→BGR] → face → [H2D] → blur → [D2H] → encode → writer
                ~~~~~~~~~          ~~~~          ~~~~
                 (auto)           (auto)        (auto)

       Memory:  HOST → HOST → HOST → CUDA → CUDA → HOST → HOST → HOST
       Format:  YUYV → BGR  → BGR  → BGR  → BGR  → JPEG → JPEG
```

### Example 5: Full GPU Pipeline (Optimal)

**User Input:**
```json
{
  "modules": {
    "generator": { "type": "TestSignalGenerator" },
    "upload": { "type": "CudaMemCopy", "props": { "kind": "HostToDevice" } },
    "blur": { "type": "GaussianBlur" },
    "resize": { "type": "ResizeNPPI" },
    "effects": { "type": "EffectsNPPI" },
    "encode": { "type": "JPEGEncoderNVJPEG" },
    "writer": { "type": "FileWriterModule" }
  }
}
```

**Validator Output:**
```
[INFO] Building pipeline...
[INFO] Excellent! This pipeline is already optimized:
       - Single H2D transfer at the start
       - All processing stays on GPU
       - JPEGEncoderNVJPEG handles encoding on GPU, outputs HOST
       - No D2H transfer needed (encoder does it internally)

[INFO] Final pipeline:
       generator → upload → blur → resize → effects → encode → writer
       HOST      → CUDA   → CUDA → CUDA   → CUDA   → HOST   → HOST
                   └───────────── GPU BLOCK ───────────────┘

       Memory transfers: 1 (minimum possible)
```

---

## Auto-Bridge Insertion

### Bridge Types

The framework auto-inserts two types of bridges:
1. **Memory bridges**: Transfer data between memory locations (CudaMemCopy)
2. **Format bridges**: Convert pixel formats (ColorConversion, CCNPPI)

### Memory Bridge Selection

| From MemType | To MemType | Bridge Module | Props |
|--------------|------------|---------------|-------|
| HOST | CUDA_DEVICE | CudaMemCopy | kind: "HostToDevice" |
| CUDA_DEVICE | HOST | CudaMemCopy | kind: "DeviceToHost" |
| HOST | DMABUF | MemTypeConversion | — |
| DMABUF | HOST | MemTypeConversion | — |
| CUDA_DEVICE | DMABUF | *not supported* | error |
| DMABUF | CUDA_DEVICE | *not supported* | error |

### Format Bridge Selection

| Current MemType | Bridge Module | Props |
|-----------------|---------------|-------|
| HOST | ColorConversion | inFormat, outFormat |
| CUDA_DEVICE | CCNPPI | inFormat, outFormat |

**Note**: Format bridge is selected based on the *current* memory type at that point in the pipeline. If data is on GPU, use CCNPPI to avoid unnecessary transfers.

### Combined Bridge Order

When BOTH format and memory mismatch exist, the order matters:

**Scenario A: HOST → CUDA with format change**
```
Producer (HOST, YUV420) → Consumer (CUDA_DEVICE, NV12)

Option 1: Format first (on HOST), then upload
  [ColorConversion YUV420→NV12] → [CudaMemCopy H2D]
  Cost: CPU conversion + H2D transfer

Option 2: Upload first, then format (on GPU)
  [CudaMemCopy H2D] → [CCNPPI YUV420→NV12]
  Cost: H2D transfer + GPU conversion

Preferred: Option 2 (GPU conversion is faster)
```

**Scenario B: CUDA → HOST with format change**
```
Producer (CUDA_DEVICE, YUV420) → Consumer (HOST, BGR)

Option 1: Format first (on GPU), then download
  [CCNPPI YUV420→BGR] → [CudaMemCopy D2H]
  Cost: GPU conversion + D2H transfer

Option 2: Download first, then format (on HOST)
  [CudaMemCopy D2H] → [ColorConversion YUV420→BGR]
  Cost: D2H transfer + CPU conversion

Preferred: Option 1 (do conversion while still on GPU)
```

### Bridge Naming Convention

Auto-inserted modules get generated names:
- `_bridge_H2D_color_blur` (between color and blur, HostToDevice)
- `_bridge_D2H_effects_encoder` (between effects and encoder, DeviceToHost)
- `_bridge_cvt_YUV_BGR_source_face` (format conversion, YUV to BGR)
- `_bridge_cvt_gpu_YUV_NV12_blur_resize` (GPU format conversion)

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

### Phase 1b: PIN ImageType Registration

**Files to modify:**
- `base/include/declarative/Metadata.h` - Add imageTypes to PinDef
- `base/include/declarative/ModuleRegistry.h` - Update PinInfo
- `base/src/declarative/ModuleRegistrations.cpp` - Update all registrations

**Tasks:**
1. Add `std::vector<ImageMetadata::ImageType> imageTypes` field to `PinDef` struct
2. Add `.inputImageTypes()` and `.outputImageType()` to registration builder
3. Default to empty (accepts any) when not specified
4. Update modules with specific format requirements:
   - ImageEncoderCV: {BGR, RGB, MONO}
   - FaceDetectorXform: {BGR}
   - H264EncoderV4L2: {YUV420}
   - ResizeNPPI: {NV12, YUV420}
   - etc.

**Acceptance Criteria:**
- All modules have correct imageType declarations
- Unit tests verify imageTypes stored correctly
- Backward compatible (empty = any format)

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

### Phase 3: Pipeline Compatibility Analyzer

**Files to create:**
- `base/include/declarative/PipelineAnalyzer.h`
- `base/src/declarative/PipelineAnalyzer.cpp`

**Tasks:**
1. Build compatibility flow graph from connections
2. Query registry for each module's PIN specs (memType, imageTypes)
3. Identify all mismatches in order:
   a. Frame type mismatches (ERROR - cannot bridge)
   b. Pixel format mismatches (can bridge with ColorConversion/CCNPPI)
   c. Memory type mismatches (can bridge with CudaMemCopy)
4. Return list of required bridges with proper ordering

**Analyzer Output Structure:**
```cpp
struct AnalysisResult {
    bool hasErrors;                    // Frame type incompatibilities
    std::vector<BridgeSpec> bridges;   // Required bridges in order
    std::vector<Suggestion> suggestions; // Module replacements
    std::vector<Warning> warnings;     // Suboptimal patterns
};

struct BridgeSpec {
    std::string fromModule;
    std::string toModule;
    BridgeType type;                   // FORMAT or MEMORY
    std::string bridgeModule;          // "ColorConversion", "CudaMemCopy", etc.
    nlohmann::json props;              // Props for bridge module
};
```

**Acceptance Criteria:**
- Correctly identifies frame type mismatches (errors)
- Correctly identifies pixel format mismatches (bridges)
- Correctly identifies memory type mismatches (bridges)
- Handles combined format+memory mismatches with correct order
- Unit tests for various pipeline topologies

### Phase 4: Auto-Bridge Insertion

**Files to modify:**
- `base/src/declarative/ModuleFactory.cpp`

**Tasks:**
1. After module creation, run PipelineAnalyzer
2. For each bridge spec:
   a. Create bridge module with correct props
   b. Insert into connection graph
   c. For CUDA bridges: use shared cudastream
   d. For format bridges: configure in/out formats
3. Generate meaningful bridge names
4. Track bridge insertion for user feedback

**Bridge Creation Rules:**
```cpp
// Memory bridges
if (bridge.type == MEMORY) {
    if (currentMemType == HOST && targetMemType == CUDA_DEVICE) {
        createCudaMemCopy(HostToDevice, sharedStream);
    } else if (currentMemType == CUDA_DEVICE && targetMemType == HOST) {
        createCudaMemCopy(DeviceToHost, sharedStream);
    }
}

// Format bridges (select based on current memory location)
if (bridge.type == FORMAT) {
    if (currentMemType == CUDA_DEVICE) {
        createCCNPPI(inFormat, outFormat, sharedStream);
    } else {
        createColorConversion(inFormat, outFormat);
    }
}
```

**Acceptance Criteria:**
- CUDA pipelines work without manual CudaMemCopy
- Pixel format conversions happen automatically
- Bridges share cudastream with other CUDA modules
- Pipeline init succeeds
- Correct bridge ordering (format before memory when going D→H)

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

**ImageType Registration Tests:**
```cpp
BOOST_AUTO_TEST_CASE(PinDef_DefaultImageTypes_IsEmpty) {
    PinDef pin{"input", {FrameType::RAW_IMAGE}};
    BOOST_CHECK(pin.imageTypes.empty());  // Empty = accepts any
}

BOOST_AUTO_TEST_CASE(ImageEncoderCV_InputImageTypes_IsBGR_RGB) {
    auto* info = registry.getModule("ImageEncoderCV");
    auto& types = info->inputs[0].imageTypes;
    BOOST_CHECK(contains(types, ImageType::BGR));
    BOOST_CHECK(contains(types, ImageType::RGB));
    BOOST_CHECK(!contains(types, ImageType::YUV420));  // Not supported
}

BOOST_AUTO_TEST_CASE(FaceDetectorXform_RequiresBGR) {
    auto* info = registry.getModule("FaceDetectorXform");
    auto& types = info->inputs[0].imageTypes;
    BOOST_CHECK_EQUAL(types.size(), 1);
    BOOST_CHECK_EQUAL(types[0], ImageType::BGR);
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

**Pipeline Analyzer Tests - Memory:**
```cpp
BOOST_AUTO_TEST_CASE(Analyzer_HostToCUDA_RequiresBridge) {
    // ColorConversion (HOST) → GaussianBlur (CUDA_DEVICE)
    auto result = analyzer.analyze(connections);
    BOOST_CHECK_EQUAL(result.bridges.size(), 1);
    BOOST_CHECK_EQUAL(result.bridges[0].type, BridgeType::MEMORY);
}

BOOST_AUTO_TEST_CASE(Analyzer_CUDAToCUDA_NoBridge) {
    // GaussianBlur (CUDA) → ResizeNPPI (CUDA)
    auto result = analyzer.analyze(connections);
    BOOST_CHECK(result.bridges.empty());
}
```

**Pipeline Analyzer Tests - Pixel Format:**
```cpp
BOOST_AUTO_TEST_CASE(Analyzer_YUV_To_BGR_RequiresColorConversion) {
    // TestSignalGenerator (YUV420) → ImageEncoderCV (BGR)
    auto result = analyzer.analyze(connections);
    BOOST_CHECK_EQUAL(result.bridges.size(), 1);
    BOOST_CHECK_EQUAL(result.bridges[0].type, BridgeType::FORMAT);
    BOOST_CHECK_EQUAL(result.bridges[0].bridgeModule, "ColorConversion");
}

BOOST_AUTO_TEST_CASE(Analyzer_BGR_To_BGR_NoBridge) {
    // ImageDecoderCV (BGR) → FaceDetectorXform (BGR)
    auto result = analyzer.analyze(connections);
    BOOST_CHECK(result.bridges.empty());
}

BOOST_AUTO_TEST_CASE(Analyzer_GPU_FormatMismatch_UsesCCNPPI) {
    // GaussianBlur (YUV420/CUDA) → EffectsNPPI (NV12/CUDA)
    auto result = analyzer.analyze(connections);
    BOOST_CHECK_EQUAL(result.bridges[0].bridgeModule, "CCNPPI");  // Not ColorConversion
}
```

**Pipeline Analyzer Tests - Combined:**
```cpp
BOOST_AUTO_TEST_CASE(Analyzer_Combined_Format_And_Memory) {
    // TestSignalGenerator (YUV420/HOST) → ResizeNPPI (NV12/CUDA)
    auto result = analyzer.analyze(connections);

    // Should have 2 bridges
    BOOST_CHECK_EQUAL(result.bridges.size(), 2);

    // Order matters: H2D first, then CCNPPI (do format on GPU)
    BOOST_CHECK_EQUAL(result.bridges[0].type, BridgeType::MEMORY);
    BOOST_CHECK_EQUAL(result.bridges[1].type, BridgeType::FORMAT);
    BOOST_CHECK_EQUAL(result.bridges[1].bridgeModule, "CCNPPI");  // On GPU
}

BOOST_AUTO_TEST_CASE(Analyzer_CUDA_To_HOST_Format_And_Memory) {
    // GaussianBlur (YUV420/CUDA) → ImageEncoderCV (BGR/HOST)
    auto result = analyzer.analyze(connections);

    // Order: CCNPPI first (while on GPU), then D2H
    BOOST_CHECK_EQUAL(result.bridges[0].type, BridgeType::FORMAT);
    BOOST_CHECK_EQUAL(result.bridges[0].bridgeModule, "CCNPPI");
    BOOST_CHECK_EQUAL(result.bridges[1].type, BridgeType::MEMORY);
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
2. **Understands** pixel format requirements via per-PIN imageType registration
3. **Classifies** modules via functional tags (function, backend, media, format)
4. **Detects** memory type incompatibilities automatically
5. **Detects** pixel format incompatibilities automatically
6. **Resolves** memory mismatches by auto-inserting CudaMemCopy bridges
7. **Resolves** format mismatches by auto-inserting ColorConversion/CCNPPI bridges
8. **Optimizes** bridge order (do format conversion on GPU when possible)
9. **Suggests** GPU-accelerated alternatives to avoid unnecessary transfers
10. **Educates** users about suboptimal patterns with actionable suggestions
11. **Enables** future LLM-based pipeline generation

The user experience changes from "must understand CUDA memory model and pixel formats" to "just pick the modules you want."

---

## Appendix A: Complete Module Specification

This appendix provides exact specifications for every registered module including:
- Input/output PIN memory types
- Input/output PIN pixel formats (imageTypes)
- Required functional tags
- Backend tags

### A.1 Memory Type Legend

| MemType | Value | Description |
|---------|-------|-------------|
| HOST | 1 | Regular CPU memory (default) |
| HOST_PINNED | 2 | Page-locked CPU memory |
| CUDA_DEVICE | 3 | NVIDIA CUDA GPU memory |
| DMABUF | 4 | DMA buffer (Linux/Jetson) |

### A.2 Source Modules

All source modules output to HOST memory by default.

| Module | Output MemType | Function Tags | Backend Tags | Media Tags |
|--------|---------------|---------------|--------------|------------|
| FileReaderModule | HOST | `function:read` | `backend:cpu` | `media:file` |
| TestSignalGenerator | HOST | `function:generate`, `function:test` | `backend:cpu` | `media:image` |
| WebCamSource | HOST | `function:capture` | `backend:cpu`, `backend:opencv` | `media:video`, `media:camera` |
| RTSPClientSrc | HOST | `function:capture`, `function:stream` | `backend:cpu` | `media:video`, `media:network` |
| ExternalSourceModule | HOST | `function:inject` | `backend:cpu` | `media:any` |
| Mp4ReaderSource | HOST | `function:read`, `function:decode` | `backend:cpu` | `media:video`, `media:file` |
| AudioCaptureSrc | HOST | `function:capture` | `backend:cpu` | `media:audio` |

### A.3 Transform Modules (CPU)

| Module | Input MemType | Output MemType | Function Tags | Backend Tags |
|--------|--------------|----------------|---------------|--------------|
| ImageDecoderCV | HOST | HOST | `function:decode` | `backend:cpu`, `backend:opencv` |
| ImageEncoderCV | HOST | HOST | `function:encode` | `backend:cpu`, `backend:opencv` |
| ImageResizeCV | HOST | HOST | `function:resize` | `backend:cpu`, `backend:opencv` |
| RotateCV | HOST | HOST | `function:rotate` | `backend:cpu`, `backend:opencv` |
| ColorConversion | HOST | HOST | `function:color-convert` | `backend:cpu`, `backend:opencv` |
| VirtualPTZ | HOST | HOST | `function:crop`, `function:ptz` | `backend:cpu` |
| TextOverlayXForm | HOST | HOST | `function:overlay`, `function:text` | `backend:cpu` |
| BrightnessContrastControl | HOST | HOST | `function:adjust`, `function:brightness`, `function:contrast` | `backend:cpu` |
| OverlayModule | HOST | HOST | `function:overlay` | `backend:cpu` |
| HistogramOverlay | HOST | HOST | `function:overlay`, `function:histogram` | `backend:cpu` |
| BMPConverter | HOST | HOST | `function:encode` | `backend:cpu` |
| AffineTransform | HOST | HOST | `function:transform`, `function:rotate`, `function:scale` | `backend:cpu`, `backend:opencv` |
| AudioToTextXForm | HOST | HOST | `function:transcribe`, `function:speech-to-text` | `backend:cpu`, `backend:whisper` |

### A.4 Transform Modules (CUDA)

| Module | Input MemType | Output MemType | Function Tags | Backend Tags |
|--------|--------------|----------------|---------------|--------------|
| GaussianBlur | **CUDA_DEVICE** | **CUDA_DEVICE** | `function:blur` | `backend:cuda`, `backend:npp` |
| ResizeNPPI | **CUDA_DEVICE** | **CUDA_DEVICE** | `function:resize` | `backend:cuda`, `backend:npp` |
| RotateNPPI | **CUDA_DEVICE** | **CUDA_DEVICE** | `function:rotate` | `backend:cuda`, `backend:npp` |
| CCNPPI | **CUDA_DEVICE** | **CUDA_DEVICE** | `function:color-convert` | `backend:cuda`, `backend:npp` |
| EffectsNPPI | **CUDA_DEVICE** | **CUDA_DEVICE** | `function:adjust`, `function:effects` | `backend:cuda`, `backend:npp` |
| OverlayNPPI | **CUDA_DEVICE** | **CUDA_DEVICE** | `function:overlay` | `backend:cuda`, `backend:npp` |
| JPEGDecoderNVJPEG | HOST | **CUDA_DEVICE** | `function:decode` | `backend:cuda`, `backend:nvjpeg` |
| JPEGEncoderNVJPEG | **CUDA_DEVICE** | HOST | `function:encode` | `backend:cuda`, `backend:nvjpeg` |
| H264Decoder | HOST | HOST* | `function:decode` | `backend:cpu` |

*Note: H264Decoder outputs HOST on x64, DMABUF on ARM64/Jetson.

### A.5 Analytics Modules

| Module | Input MemType | Output MemType | Function Tags | Backend Tags |
|--------|--------------|----------------|---------------|--------------|
| FaceDetectorXform | HOST | HOST | `function:detect`, `function:face` | `backend:cpu`, `backend:opencv` |
| QRReader | HOST | HOST | `function:detect`, `function:qr`, `function:barcode` | `backend:cpu` |
| CalcHistogramCV | HOST | HOST | `function:analyze`, `function:histogram` | `backend:cpu`, `backend:opencv` |
| FacialLandmarkCV | HOST | HOST | `function:detect`, `function:face`, `function:landmarks` | `backend:cpu`, `backend:opencv` |
| MotionVectorExtractor | HOST | HOST | `function:analyze`, `function:motion` | `backend:cpu` |

### A.6 Sink Modules

All sink modules accept HOST memory by default.

| Module | Input MemType | Function Tags | Backend Tags | Media Tags |
|--------|--------------|---------------|--------------|------------|
| FileWriterModule | HOST | `function:write` | `backend:cpu` | `media:file` |
| StatSink | HOST | `function:stats`, `function:debug` | `backend:cpu` | — |
| Mp4WriterSink | HOST | `function:write`, `function:encode` | `backend:cpu` | `media:video`, `media:file` |
| ExternalSinkModule | HOST | `function:export` | `backend:cpu` | `media:any` |
| RTSPPusher | HOST | `function:stream`, `function:push` | `backend:cpu` | `media:video`, `media:network` |
| ThumbnailListGenerator | HOST | `function:generate`, `function:thumbnail` | `backend:cpu` | `media:image` |
| VirtualCameraSink | HOST | `function:output` | `backend:cpu`, `backend:v4l2` | `media:video`, `media:camera` |

### A.7 Utility Modules

| Module | Input MemType | Output MemType | Function Tags | Backend Tags | Notes |
|--------|--------------|----------------|---------------|--------------|-------|
| ValveModule | HOST | HOST | `function:control`, `function:flow` | `backend:cpu` | Pass-through |
| Split | HOST | HOST | `function:split`, `function:route` | `backend:cpu` | Pass-through |
| Merge | HOST | HOST | `function:merge`, `function:sync` | `backend:cpu` | Pass-through |
| MultimediaQueueXform | HOST | HOST | `function:buffer`, `function:queue` | `backend:cpu` | Pass-through |
| FramesMuxer | HOST | HOST | `function:mux`, `function:sync` | `backend:cpu` | Pass-through |
| ArchiveSpaceManager | — | — | `function:manage`, `function:storage` | `backend:cpu` | No pins |
| **CudaMemCopy** | **DYNAMIC** | **DYNAMIC** | `function:transfer` | `backend:cuda` | See A.8 |
| CudaStreamSynchronize | CUDA_DEVICE | CUDA_DEVICE | `function:sync` | `backend:cuda` | Pass-through |
| MemTypeConversion | DYNAMIC | DYNAMIC | `function:transfer`, `function:convert` | `backend:cuda` | See A.8 |
| CuCtxSynchronize | CUDA_DEVICE | CUDA_DEVICE | `function:sync` | `backend:cuda` | Pass-through |

### A.8 Memory Transfer Modules (Special Handling)

These modules change memory type based on their properties.

#### CudaMemCopy

| props.kind | Input MemType | Output MemType |
|------------|--------------|----------------|
| HostToDevice | HOST | CUDA_DEVICE |
| DeviceToHost | CUDA_DEVICE | HOST |
| DeviceToDevice | CUDA_DEVICE | CUDA_DEVICE |

#### MemTypeConversion

| props.outputMemType | Input MemType | Output MemType |
|--------------------|--------------|----------------|
| HOST | ANY | HOST |
| DEVICE | ANY | CUDA_DEVICE |
| DMA | ANY | DMABUF |

### A.9 Linux-Only Modules

| Module | Input MemType | Output MemType | Function Tags | Backend Tags |
|--------|--------------|----------------|---------------|--------------|
| VirtualCameraSink | HOST | — | `function:output` | `backend:cpu`, `backend:v4l2` |
| H264EncoderV4L2 | HOST | HOST | `function:encode` | `backend:cpu`, `backend:v4l2` |

### A.10 Format Tags

| Module | Format Tags |
|--------|-------------|
| ImageDecoderCV | `format:jpeg`, `format:png`, `format:bmp` |
| ImageEncoderCV | `format:jpeg`, `format:png` |
| JPEGDecoderNVJPEG | `format:jpeg` |
| JPEGEncoderNVJPEG | `format:jpeg` |
| BMPConverter | `format:bmp` |
| Mp4ReaderSource | `format:mp4`, `format:h264` |
| Mp4WriterSink | `format:mp4`, `format:h264` |
| H264Decoder | `format:h264` |
| H264EncoderV4L2 | `format:h264` |
| RTSPClientSrc | `format:h264`, `format:rtsp` |
| RTSPPusher | `format:h264`, `format:rtsp` |

### A.11 ImageType Specifications (Pixel Formats)

This section specifies which pixel formats each module accepts/produces.

**ImageType Legend** (from `ImageMetadata::ImageType`):

| ImageType | Description |
|-----------|-------------|
| MONO | Grayscale, single channel |
| BGR | Blue-Green-Red packed (OpenCV default) |
| BGRA | BGR + Alpha channel |
| RGB | Red-Green-Blue packed |
| RGBA | RGB + Alpha channel |
| YUV420 | Planar YUV 4:2:0 (Y plane + U plane + V plane) |
| YUV411_I | Interleaved YUV 4:1:1 |
| YUV444 | Planar YUV 4:4:4 |
| NV12 | Semi-planar YUV (Y plane + interleaved UV) |
| UYVY | Packed YUV (U-Y-V-Y) |
| YUYV | Packed YUV (Y-U-Y-V) |

**Source Module Output Formats:**

| Module | Output ImageTypes | Notes |
|--------|-------------------|-------|
| TestSignalGenerator | YUV420 | Planar YUV output |
| WebCamSource | YUYV, UYVY, BGR | Depends on camera driver |
| Mp4ReaderSource | NV12, YUV420 | Depends on video codec |
| RTSPClientSrc | NV12, YUV420 | Typically H264→NV12 |
| ImageDecoderCV | BGR | Always outputs BGR |

**Transform Module ImageType Requirements:**

| Module | Input ImageTypes | Output ImageTypes | Notes |
|--------|------------------|-------------------|-------|
| ImageEncoderCV | {BGR, RGB, MONO} | — | Only accepts BGR/RGB/MONO |
| ImageResizeCV | {any} | same as input | Format pass-through |
| RotateCV | {any} | same as input | Format pass-through |
| ColorConversion | {any} | configured | Configurable output |
| FaceDetectorXform | {BGR} | BGR | **Only BGR** (OpenCV DNN) |
| FacialLandmarkCV | {BGR} | BGR | **Only BGR** (OpenCV) |
| VirtualPTZ | {any} | same as input | Format pass-through |
| AffineTransform | {any} | same as input | Format pass-through |
| BrightnessContrastControl | {BGR, RGB} | same as input | Color images only |
| TextOverlayXForm | {BGR, RGB} | same as input | Color images only |
| QRReader | {any} | same as input | Internally converts |

**CUDA Module ImageType Requirements:**

| Module | Input ImageTypes | Output ImageTypes | Notes |
|--------|------------------|-------------------|-------|
| GaussianBlur | {NV12, YUV420} | same as input | NPP format |
| ResizeNPPI | {NV12, YUV420} | same as input | NPP format |
| RotateNPPI | {NV12, YUV420} | same as input | NPP format |
| EffectsNPPI | {NV12, YUV420} | same as input | NPP format |
| OverlayNPPI | {NV12} | NV12 | **Only NV12** |
| CCNPPI | {any} | configured | GPU color converter |
| JPEGDecoderNVJPEG | — | BGR, RGB | JPEG → color |
| JPEGEncoderNVJPEG | {BGR, RGB, YUV420} | — | Color → JPEG |

**Sink Module ImageType Requirements:**

| Module | Input ImageTypes | Notes |
|--------|------------------|-------|
| FileWriterModule | {any} | Writes raw bytes |
| Mp4WriterSink | {YUV420, NV12} | Video encoder requires YUV |
| H264EncoderV4L2 | {YUV420} | **Only YUV420** |
| RTSPPusher | {YUV420, NV12} | For H264 encoding |
| StatSink | {any} | Ignores format |
| VirtualCameraSink | {YUV420, YUYV} | V4L2 formats |

---

## Appendix B: Functional Equivalence Groups

Modules with the same `function:*` tag are functionally equivalent and can be suggested as alternatives.

### B.1 Resize Group (`function:resize`)

| Module | Backend | MemType |
|--------|---------|---------|
| ImageResizeCV | cpu, opencv | HOST |
| ResizeNPPI | cuda, npp | CUDA_DEVICE |

**Auto-suggestion**: When ImageResizeCV breaks a CUDA chain, suggest ResizeNPPI.

### B.2 Rotate Group (`function:rotate`)

| Module | Backend | MemType |
|--------|---------|---------|
| RotateCV | cpu, opencv | HOST |
| RotateNPPI | cuda, npp | CUDA_DEVICE |
| AffineTransform | cpu, opencv | HOST |

**Auto-suggestion**: When RotateCV breaks a CUDA chain, suggest RotateNPPI.

### B.3 Color Convert Group (`function:color-convert`)

| Module | Backend | MemType |
|--------|---------|---------|
| ColorConversion | cpu, opencv | HOST |
| CCNPPI | cuda, npp | CUDA_DEVICE |

**Auto-suggestion**: When ColorConversion breaks a CUDA chain, suggest CCNPPI.

### B.4 Blur Group (`function:blur`)

| Module | Backend | MemType |
|--------|---------|---------|
| GaussianBlur | cuda, npp | CUDA_DEVICE |

**Note**: No CPU equivalent registered. Consider registering a CPU blur module in future.

### B.5 Encode Group (`function:encode`)

| Module | Backend | Format | Input MemType |
|--------|---------|--------|---------------|
| ImageEncoderCV | cpu, opencv | jpeg, png | HOST |
| JPEGEncoderNVJPEG | cuda, nvjpeg | jpeg | CUDA_DEVICE |
| BMPConverter | cpu | bmp | HOST |
| H264EncoderV4L2 | cpu, v4l2 | h264 | HOST |

### B.6 Decode Group (`function:decode`)

| Module | Backend | Format | Output MemType |
|--------|---------|--------|----------------|
| ImageDecoderCV | cpu, opencv | jpeg, png, bmp | HOST |
| JPEGDecoderNVJPEG | cuda, nvjpeg | jpeg | CUDA_DEVICE |
| H264Decoder | cpu | h264 | HOST |

### B.7 Overlay Group (`function:overlay`)

| Module | Backend | MemType |
|--------|---------|---------|
| OverlayModule | cpu | HOST |
| TextOverlayXForm | cpu | HOST |
| HistogramOverlay | cpu | HOST |
| OverlayNPPI | cuda, npp | CUDA_DEVICE |

### B.8 Adjust/Effects Group (`function:adjust`)

| Module | Backend | MemType |
|--------|---------|---------|
| BrightnessContrastControl | cpu | HOST |
| EffectsNPPI | cuda, npp | CUDA_DEVICE |


---

## Implementation Status

**All phases (1-5) are complete.** See [PROGRESS.md](./PROGRESS.md) for implementation details, commits, and test results.

Key implementation files:
- `base/include/declarative/ModuleRegistry.h` - PinInfo with memType/imageTypes
- `base/include/declarative/PipelineAnalyzer.h` - BridgeSpec, AnalysisResult structs
- `base/src/declarative/PipelineAnalyzer.cpp` - Connection analysis and bridge detection
- `base/src/declarative/ModuleFactory.cpp` - Auto-bridge insertion
- `base/src/declarative/ModuleRegistrations.cpp` - CUDA module registrations with memType

Test coverage: 13+ PipelineAnalyzerTests, integration tests in `examples/cuda/`.
