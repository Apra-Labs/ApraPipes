# Declarative Pipeline - Progress Tracker

> Last Updated: 2026-01-07

---

## Current Status

**Sprints 1-3:** ✅ COMPLETE
**Sprint 4 (Node.js):** ✅ COMPLETE

```
Core Infrastructure:  ✅ Complete (Metadata, Registry, Factory, Validator, CLI)
JSON Migration:       ✅ Complete (TOML removed, JsonParser added)
Module Coverage:      37 modules (cross-platform) + platform-specific
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

## Registered Modules

### Source Modules (7)
- TestSignalGenerator (patterns: GRADIENT, CHECKERBOARD, COLOR_BARS, GRID)
- FileReaderModule
- Mp4ReaderSource (outputFormat: "h264"/"jpeg" for declarative)
- WebCamSource
- RTSPClientSrc
- ExternalSourceModule
- AudioCaptureSrc

### Transform Modules (13)
- ImageDecoderCV
- ImageEncoderCV
- ImageResizeCV
- RotateCV
- ColorConversion
- VirtualPTZ (dynamic: roiX, roiY, roiWidth, roiHeight)
- TextOverlayXForm (dynamic properties)
- BrightnessContrastControl (dynamic: contrast, brightness)
- AffineTransform
- BMPConverter
- OverlayModule
- HistogramOverlay
- AudioToTextXForm (Whisper speech-to-text)

### Analytics Modules (5)
- FaceDetectorXform
- FacialLandmarkCV
- QRReader
- CalcHistogramCV
- MotionVectorExtractor

### Sink Modules (6)
- FileWriterModule
- Mp4WriterSink
- StatSink
- ExternalSinkModule
- RTSPPusher
- ThumbnailListGenerator

### Utility Modules (6)
- Split
- Merge
- ValveModule
- FramesMuxer
- MultimediaQueueXform
- ArchiveSpaceManager

### Linux-Only Modules (ENABLE_LINUX)
- VirtualCameraSink (virtual camera device output)
- H264EncoderV4L2 (V4L2 hardware encoder)

### CUDA-Only Modules (ENABLE_CUDA)
- H264Decoder (NVDEC decoder)

**Total: 37 cross-platform + 2 Linux + 1 CUDA = 40 modules**

---

## Modules Not Registered

### CUDA Modules (require cudastream_sp in constructor)
| Module | Reason |
|--------|--------|
| GaussianBlur | Requires CUDA stream |
| H264EncoderNVCodec | Requires CUDA stream |
| JPEGDecoderNVJPEG | Requires CUDA stream |
| JPEGEncoderNVJPEG | Requires CUDA stream |
| ResizeNPPI | Requires CUDA stream |
| RotateNPPI | Requires CUDA stream |
| CCNPPI | Requires CUDA stream |
| EffectsNPPI | Requires CUDA stream |
| OverlayNPPI | Requires CUDA stream |
| CudaMemCopy | Requires CUDA stream |
| CudaStreamSynchronize | Requires CUDA stream |
| CuCtxSynchronize | Requires CUDA context |
| MemTypeConversion | Requires CUDA stream |

**Blocker:** Need CUDA context factory to create/pass cudastream_sp declaratively.

### Jetson-Only Modules (L4T/Tegra)
| Module | Reason |
|--------|--------|
| NvArgusCamera | Jetson Argus camera API |
| NvV4L2Camera | Jetson V4L2 camera |
| NvTransform | Jetson multimedia transform |
| JPEGDecoderL4TM | L4T Multimedia decoder |
| JPEGEncoderL4TM | L4T Multimedia encoder |
| DMAFDToHostCopy | Jetson DMA file descriptor |

**Status:** Need conditional registration with `#ifdef` guards on Jetson builds.

### Display/UI Modules
| Module | Reason |
|--------|--------|
| GtkGlRenderer | Requires GTK display |
| EglRenderer | Requires EGL display |
| ImageViewerModule | Requires OpenCV GUI |
| KeyboardListener | Requires keyboard input |

**Status:** Could be registered but typically not used in headless pipelines.

---

## Build Status

| Platform | Status | Node Addon | Modules |
|----------|--------|------------|---------|
| macOS | ✅ Pass | ✅ | 37 |
| Linux x64 | ✅ Pass | ✅ | 39 (+V4L2) |
| Windows | ✅ Pass | ✅ | 37 |
| Linux ARM64 | ✅ Pass | ❌ (#493) | 39 |
| Linux CUDA | ✅ Pass | ✅ | 40 (+H264Decoder) |
| Jetson | ✅ Pass | ❌ | 40+ (+Jetson modules) |

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
| rtsp_pusher_demo.js | Mp4ReaderSource -> RTSPPusher streaming |

---

## Future Work

### Priority 1: CUDA Module Support
- Create CUDA context factory for declarative CUDA pipelines
- Allow cudastream_sp to be created/passed via pipeline config
- Enable 13 CUDA modules for declarative use

### Priority 2: Jetson Module Registration
- Add `#ifdef` guards for Jetson-specific modules
- Test on Jetson device with L4T

### Priority 3: Display Modules (Optional)
- Register UI modules with platform checks
- Low priority - mainly for debugging

### Priority 4: ARM64 Node Addon
- Fix -fPIC linking issue (#493)
