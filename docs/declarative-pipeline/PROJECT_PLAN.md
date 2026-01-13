# Declarative Pipeline Construction - Project Plan

> Last Updated: 2026-01-12

## Project Status: Sprint 7 (Auto-Bridging)

**Package:** `@apralabs/aprapipes`
**Format:** JSON only
**Branch:** `feat-declarative-pipeline-v2`

---

## Sprints Overview

| Sprint | Status | Theme |
|--------|--------|-------|
| Sprint 1 | ✅ Complete | Foundations (Metadata, Registry, Parser) |
| Sprint 2 | ✅ Complete | Core Engine (Factory, CLI, Validator) |
| Sprint 3 | ✅ Complete | Module Expansion (37 modules) |
| Sprint 4 | ✅ Complete | Node.js Addon |
| Sprint 5 | ✅ Complete | CUDA Module Registration (12 modules) |
| Sprint 6 | ✅ Complete | DRY Refactoring |
| Sprint 7 | ✅ Complete | Auto-Bridging (Memory + Pixel Format) |

---

## Sprint 7: Auto-Bridging Implementation

**Design Document:** `docs/declarative-pipeline/CUDA_MEMTYPE_DESIGN.md`

### Phase 1: PIN MemType Registration ✅ Complete

| Task | Status | Description |
|------|--------|-------------|
| 1.1 | ✅ Done | Add `memType` field to `PinInfo` struct in ModuleRegistry.h |
| 1.2 | ✅ Done | Add `cudaInput()`/`cudaOutput()` to ModuleRegistrationBuilder |
| 1.3 | ✅ Done | Update CUDA module registrations (12 modules) to declare CUDA_DEVICE |
| 1.4 | ⏳ Skip | DMA modules not yet registered (future work) |
| 1.5 | ✅ Done | Add unit tests for memType registration |

**Commit:** `4ad6f7da6 feat(declarative): Add memType registration for CUDA modules`

**Files to modify:**
- `base/include/declarative/ModuleRegistry.h`
- `base/include/declarative/ModuleRegistrationBuilder.h`
- `base/src/declarative/ModuleRegistrations.cpp`
- `base/test/declarative/module_registry_tests.cpp`

### Phase 1b: PIN ImageType Registration ✅ Complete

| Task | Status | Description |
|------|--------|-------------|
| 1b.1 | ✅ Done | Add `imageTypes` field to `PinInfo` struct |
| 1b.2 | ✅ Done | Add `inputImageTypes()`/`outputImageTypes()` to builder |
| 1b.3 | ⏳ Pending | Update modules with specific format requirements (incremental) |
| 1b.4 | ✅ Done | Add unit tests for imageType registration |

**Commit:** `1ada7a15e feat(declarative): Add imageType registration for pixel format validation`

**Modules requiring imageTypes:**
- ImageEncoderCV: {BGR, RGB, MONO}
- FaceDetectorXform: {BGR}
- FacialLandmarkCV: {BGR}
- H264EncoderV4L2: {YUV420}
- Mp4WriterSink: {YUV420, NV12}
- GaussianBlur: {NV12, YUV420}
- ResizeNPPI: {NV12, YUV420}
- OverlayNPPI: {NV12}

### Phase 2: Functional Tags ✅ Complete

| Task | Status | Description |
|------|--------|-------------|
| 2.1 | ✅ Done | Add `tags` field to ModuleInfo struct (already existed) |
| 2.2 | ✅ Done | Add `.tag()` method to ModuleRegistrationBuilder (already existed) |
| 2.3 | ✅ Done | Add `getModulesByTag()` and `getModulesWithAllTags()` to registry |
| 2.4 | ✅ Done | Tag modules with function/backend/media tags |
| 2.5 | ✅ Done | Add unit tests for tag queries |

**Commit:** `7a836c6b0 feat(declarative): Add getModulesWithAllTags() for multi-tag queries`

**Tag taxonomy:**
- `function:resize`, `function:blur`, `function:encode`, `function:decode`, `function:color-convert`, `function:overlay`, `function:detect`
- `backend:cpu`, `backend:cuda`, `backend:npp`, `backend:nvjpeg`, `backend:opencv`, `backend:v4l2`
- `format:jpeg`, `format:png`, `format:h264`, `format:mp4`

### Phase 3: Pipeline Compatibility Analyzer ✅ Complete

| Task | Status | Description |
|------|--------|-------------|
| 3.1 | ✅ Done | Create `PipelineAnalyzer.h` with AnalysisResult struct |
| 3.2 | ✅ Done | Create `PipelineAnalyzer.cpp` with connection analysis |
| 3.3 | ✅ Done | Implement frame type compatibility check (ERROR on mismatch) |
| 3.4 | ✅ Done | Implement pixel format mismatch detection |
| 3.5 | ✅ Done | Implement memory type mismatch detection |
| 3.6 | ✅ Done | Generate BridgeSpec list with correct ordering |
| 3.7 | ✅ Done | Add 13 unit tests for analyzer |

**Commit:** `c1b922df3 feat(declarative): Add PipelineAnalyzer for auto-bridging detection`

**Files created:**
- `base/include/declarative/PipelineAnalyzer.h`
- `base/src/declarative/PipelineAnalyzer.cpp`
- `base/test/declarative/pipeline_analyzer_tests.cpp`

### Phase 4: Auto-Bridge Insertion ✅ Complete

| Task | Status | Description |
|------|--------|-------------|
| 4.1 | ✅ Done | Integrate PipelineAnalyzer into ModuleFactory |
| 4.2 | ✅ Done | Create CudaMemCopy bridges for memory mismatches |
| 4.3 | ✅ Done | Create ColorConversion/CCNPPI bridges for format mismatches |
| 4.4 | ✅ Done | Bridges use shared cudastream (via existing CUDA factory pattern) |
| 4.5 | ✅ Done | Generate bridge names (_bridge_N_ModuleType) |
| 4.6 | ✅ Done | 42 ModuleFactory tests verify auto-bridging |

**Commit:** `a79396b10 feat(declarative): Integrate auto-bridge insertion into ModuleFactory`

### Phase 5: User Feedback & Suggestions ✅ Complete

| Task | Status | Description |
|------|--------|-------------|
| 5.1 | ✅ Done | Log auto-inserted bridges as INFO messages (I_BRIDGE_INSERTED) |
| 5.2 | ✅ Done | Detect suboptimal patterns (CPU in GPU chain) |
| 5.3 | ✅ Done | Query registry for GPU alternatives using tags |
| 5.4 | ✅ Done | Add suggestions to BuildResult (as INFO issues) |
| 5.5 | ✅ Done | Add formatPipelineGraph() for verbose output |

**Commit:** `2f01c53f3 feat(declarative): Add pipeline graph formatting for verbose output`

---

## Testing Protocol

After each phase completion:

1. **Build:** `cmake --build build --parallel`
2. **Unit Tests:** `./build/aprapipesut --run_test="*" --log_level=test_suite`
3. **Example Pipelines:** Run all working examples to verify no regressions
4. **Commit:** Create atomic commit for the phase

### Example Pipeline Test Script

```bash
#!/bin/bash
# Run all working declarative pipeline examples
EXAMPLES_DIR="examples"
CLI="./build/aprapipes_cli"

for json in "$EXAMPLES_DIR"/*.json; do
    echo "Testing: $json"
    timeout 10 $CLI run "$json" --max-frames 10 || echo "FAILED: $json"
done
```

---

## Key Files

```
base/
├── include/declarative/
│   ├── ModuleRegistry.h         # PinInfo with memType, imageTypes
│   ├── ModuleRegistrationBuilder.h  # Builder with .memType(), .imageTypes(), .tag()
│   ├── PipelineAnalyzer.h       # NEW: Compatibility analysis
│   ├── ModuleFactory.h          # Auto-bridge insertion
│   └── PipelineValidator.h      # Uses analyzer
├── src/declarative/
│   ├── ModuleRegistrations.cpp  # All module registrations
│   ├── PipelineAnalyzer.cpp     # NEW: Analysis implementation
│   └── ModuleFactory.cpp        # Bridge creation
└── test/declarative/
    ├── module_registry_tests.cpp
    ├── pipeline_analyzer_tests.cpp  # NEW
    └── module_factory_tests.cpp
```

---

## Module Count: 52 Total

| Category | Count | Modules |
|----------|-------|---------|
| Source | 7 | FileReaderModule, TestSignalGenerator, Mp4ReaderSource, WebCamSource, RTSPClientSrc, ExternalSourceModule, AudioCaptureSrc |
| Transform (CPU) | 13 | ImageDecoderCV, ImageEncoderCV, ImageResizeCV, RotateCV, ColorConversion, VirtualPTZ, TextOverlayXForm, BrightnessContrastControl, AffineTransform, BMPConverter, OverlayModule, HistogramOverlay, AudioToTextXForm |
| Transform (CUDA) | 12 | GaussianBlur, ResizeNPPI, RotateNPPI, CCNPPI, EffectsNPPI, OverlayNPPI, CudaMemCopy, CudaStreamSynchronize, JPEGDecoderNVJPEG, JPEGEncoderNVJPEG, MemTypeConversion, CuCtxSynchronize |
| Analytics | 5 | FaceDetectorXform, FacialLandmarkCV, QRReader, CalcHistogramCV, MotionVectorExtractor |
| Sink | 7 | FileWriterModule, Mp4WriterSink, StatSink, ExternalSinkModule, RTSPPusher, ThumbnailListGenerator, VirtualCameraSink |
| Utility | 6 | Split, Merge, ValveModule, FramesMuxer, MultimediaQueueXform, ArchiveSpaceManager |
| Linux-Only | 2 | H264EncoderV4L2, H264Decoder (with V4L2) |

---

## Success Criteria

Sprint 7 is complete when:

1. ✅ All CUDA modules have memType=CUDA_DEVICE declared
2. ✅ All modules with format restrictions have imageTypes declared
3. ✅ All modules have appropriate function/backend tags
4. ✅ PipelineAnalyzer correctly detects all mismatches
5. ✅ ModuleFactory auto-inserts bridges for:
   - HOST → CUDA_DEVICE (CudaMemCopy H2D)
   - CUDA_DEVICE → HOST (CudaMemCopy D2H)
   - YUV420 → BGR (ColorConversion or CCNPPI)
   - Any format mismatch
6. ✅ CUDA example pipelines work without manual CudaMemCopy
7. ✅ All existing tests pass (no regressions)
8. ✅ New tests cover all bridging scenarios
