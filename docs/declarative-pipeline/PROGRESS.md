# Declarative Pipeline - Progress Tracker

> Last Updated: 2026-01-06

---

## Current Status

**Sprints 1-3:** ✅ COMPLETE
**Sprint 4 (Node.js):** ✅ COMPLETE

```
Core Infrastructure:  ✅ Complete (Metadata, Registry, Factory, Validator, CLI)
JSON Migration:       ✅ Complete (TOML removed, JsonParser added)
Module Coverage:      31+ modules (all commonly used modules)
Node.js Addon:        ✅ Complete (Phase 5 - testing, docs, examples)
```

---

## Sprint 4 Progress

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ Complete | TOML removal |
| Phase 1 | ✅ Complete | JSON parser (24 tests) |
| Phase 2 | ✅ Complete | Node addon foundation |
| Phase 3 | ✅ Complete | Core JS API (createPipeline, Pipeline, ModuleHandle) |
| Phase 4 | ✅ Complete | Event system (on/off, health/error callbacks) |
| Phase 4.5 | ✅ Complete | Dynamic property support (getProperty/setProperty) |
| Phase 5 | ✅ Complete | Testing, docs, examples |

---

## Registered Modules (31+)

### Source Modules
- TestSignalGenerator (with GRADIENT, CHECKERBOARD, COLOR_BARS, GRID patterns)
- FileReaderModule
- Mp4ReaderSource
- WebCamSource
- RTSPClientSrc
- ExternalSourceModule

### Transform Modules
- ImageDecoderCV
- ImageEncoderCV
- ImageResizeCV
- RotateCV
- ColorConversion
- VirtualPTZ (dynamic properties: roiX, roiY, roiWidth, roiHeight)
- TextOverlayXForm (dynamic properties)
- BrightnessContrastControl (dynamic properties)
- AffineTransform
- BMPConverter
- OverlayModule
- HistogramOverlay

### Analytics Modules
- FaceDetectorXform
- QRReader
- CalcHistogramCV
- MotionVectorExtractor

### Sink Modules
- FileWriterModule
- Mp4WriterSink
- StatSink
- ExternalSinkModule

### Utility Modules
- Split
- Merge
- ValveModule
- FramesMuxer
- MultimediaQueueXform

### CUDA Modules (conditional)
- H264Decoder (ENABLE_CUDA)

---

## Modules Not Registered (Known Limitations)

### CUDA Modules (require CUDA context/stream)
- H264EncoderNVCodec
- JPEGDecoderNVJPEG
- JPEGEncoderNVJPEG
- ResizeNPPI, RotateNPPI, CCNPPI, EffectsNPPI, OverlayNPPI
- CudaMemCopy, CudaStreamSynchronize, MemTypeConversion

### Platform-Specific Modules
- H264EncoderV4L2 (V4L2/Linux)
- NvV4L2Camera, NvTransform, JPEGEncoderL4TM (Jetson)
- GtkGlRenderer, EglRenderer (Display)
- DMAFDToHostCopy (Jetson)

### Complex Constructor Modules
- VirtualCameraSink (requires device path)
- RTSPPusher (requires network config)
- ArchiveSpaceManager (requires storage config)

### Specialized Modules
- FacialLandmarksCV (requires OpenCV model files)
- ThumbnailListGenerator (needs applyProperties)
- AudioCaptureSrc, AudioToTextXForm (audio pipeline)
- ImageViewerModule, KeyboardListener (UI)

---

## Build Status

| Platform | Status | Node Addon |
|----------|--------|------------|
| macOS | ✅ Pass | ✅ |
| Linux x64 | ✅ Pass | ✅ |
| Windows | ✅ Pass | ✅ |
| ARM64 | ✅ Pass | ❌ (issue #493) |

---

## Test Results

### C++ Unit Tests (Boost.Test)

| Test Suite | Tests |
|------------|-------|
| MetadataTests | 36 |
| ModuleRegistryTests | 21 |
| PipelineDescriptionTests | 37 |
| JsonParserTests | 24 |
| ModuleFactoryTests | 42 |
| FrameTypeRegistryTests | 34 |
| PipelineValidatorTests | 28 |
| PropertyMacrosTests | 28 |
| PropertyValidatorTests | 26 |
| ModuleRegistrationTests | 11 |

**Total: 268+ declarative C++ tests passing**

### Node.js Tests

| Test File | Tests |
|-----------|-------|
| event_tests.js | 10 |
| ptz_dynamic_props_test.js | 10 |

**Total: 20 Node.js tests passing**

---

## Documentation

- [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) - Module registration guide
- [PIPELINE_AUTHOR_GUIDE.md](./PIPELINE_AUTHOR_GUIDE.md) - JSON/JS pipeline creation
- [node-api.md](../node-api.md) - Node.js API reference
- [examples/node/](../../examples/node/) - Working Node.js examples

---

## Node.js Examples

| Example | Description |
|---------|-------------|
| basic_pipeline.js | Generate test frames, encode to JPEG, write to files |
| ptz_control.js | VirtualPTZ with real-time property changes |
| event_handling.js | Health/error event handling |
| image_processing.js | Color bars + brightness/contrast control |

---

## Future Work

1. **CUDA Module Support** - Add CUDA context factory for declarative CUDA pipelines
2. **Platform Modules** - Add conditional registration for Jetson/V4L2 modules
3. **Additional Modules** - Add applyProperties to remaining modules
4. **ARM64 Node Addon** - Fix -fPIC issue for ARM64 builds
