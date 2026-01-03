# Declarative Pipeline Construction - Project Plan

> Last Updated: 2026-01-02

## Project Status: Sprint 3 (Documentation & Expansion)

All core infrastructure is complete. The system is functional with end-to-end TOML-to-pipeline execution working. Current focus is on documentation and expanding module coverage.

---

## Sprints Overview

| Sprint | Status | Theme | Key Deliverables |
|--------|--------|-------|------------------|
| **Sprint 1** | ✅ Complete | Foundations | Metadata, Registries, IR, Parser |
| **Sprint 2** | ✅ Complete | Core Engine | Factory, CLI, Validator, Schema Generator |
| **Sprint 3** | 🔄 In Progress | Documentation & Expansion | Guides, Module Coverage, Examples |

---

## Completed Work Summary

### Sprint 1: Foundations ✅

| Task | Description | Status |
|------|-------------|--------|
| A1 | Core Metadata Types | ✅ Complete |
| B1 | PipelineDescription IR | ✅ Complete |
| A2 | Module Registry | ✅ Complete |
| B2 | TOML Parser | ✅ Complete |
| A3 | FrameType Registry | ✅ Complete |
| C1 | Validator Shell | ✅ Complete |
| M1 | FileReaderModule Metadata | ✅ Complete |
| M2 | H264Decoder Metadata | ✅ Complete |

### Sprint 2: Core Engine ✅

| Task | Description | Status |
|------|-------------|--------|
| D1 | Module Factory | ✅ Complete |
| D2 | Property Binding System | ✅ Complete (20 modules, 32%) |
| D3 | Multi-Pin Connection Support | ✅ Complete |
| E1 | CLI Tool | ✅ Complete |
| E2 | Schema Generator | ✅ Complete |
| C2 | Validator: Module Checks | ✅ Complete |
| C3 | Validator: Property Checks | ✅ Complete |
| C4 | Validator: Connection Checks | ✅ Complete |
| C5 | Validator: Graph Checks | ✅ Complete |
| F1-F4 | Frame Type Metadata | ✅ Complete |

### Current Module Registration Coverage

**20 modules registered (32% of 62 total)**

| Category | Registered Modules |
|----------|-------------------|
| Source | FileReaderModule, TestSignalGenerator, Mp4ReaderSource |
| Sink | FileWriterModule, StatSink, Mp4WriterSink |
| Transform | ImageDecoderCV, ImageEncoderCV, ImageResizeCV, RotateCV, ColorConversion, VirtualPTZ, TextOverlayXForm, BrightnessContrastControl |
| Analytics | FaceDetectorXform, QRReader, CalcHistogramCV |
| Utility | ValveModule, Split, Merge |

---

## Sprint 3: Documentation & Expansion (Current)

### Goals
- Create comprehensive Developer Guide for module registration
- Create Pipeline Author Guide with schema generator usage
- Expand module registration coverage to 80%+
- Create example pipelines demonstrating all registered modules
- Fix any failing pipelines

### Phase 1: Documentation

| Task | Description | Status |
|------|-------------|--------|
| DOC1 | Developer Guide for Module Registration | 🔄 In Progress |
| DOC2 | Pipeline Author Guide | ⏳ Pending |
| DOC3 | Update README with quickstart | ⏳ Pending |

### Phase 2: Module Registration Expansion

| Batch | Modules | Status |
|-------|---------|--------|
| Batch 1 | Source modules (WebcamSource, RTSPClientSrc, etc.) | ⏳ Pending |
| Batch 2 | Transform modules (AffineTransform, OverlayModule, etc.) | ⏳ Pending |
| Batch 3 | Sink modules (RTSPPusher, etc.) | ⏳ Pending |
| Batch 4 | CUDA modules (H264Encoder, JPEGDecoder, etc.) | ⏳ Pending |
| Batch 5 | Remaining utility modules | ⏳ Pending |

### Phase 3: Example Pipelines

| Task | Description | Status |
|------|-------------|--------|
| EX1 | Create examples for each batch | ⏳ Pending |
| EX2 | Document not_working pipelines with reasons | ⏳ Pending |
| EX3 | Fix identified pipeline issues | ⏳ Pending |

---

## Working Pipelines

| Pipeline | Description | Modules Used |
|----------|-------------|--------------|
| 01_simple_source_sink.toml | Minimal test | TestSignalGenerator → StatSink |
| 02_three_module_chain.toml | Basic chain | FileReader → ImageDecoder → StatSink |
| 03_split_pipeline.toml | Fan-out | TestSignal → Split → 2x StatSink |
| 04_ptz_with_conversion.toml | Type bridge | TestSignal → ColorConversion → VirtualPTZ → StatSink |
| 09_face_detection_demo.toml | Full demo | FileReader → ImageDecoder → FaceDetector → FileWriter |

---

## Integration Test Status

| Test | Description | Status |
|------|-------------|--------|
| FaceDetectionPipeline_FromToml | End-to-end face detection | ✅ Passing |
| Validates | TOML parse → Build → Init → Run → Verify 5 faces | ✅ |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     TOML Pipeline File                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TomlParser                                 │
│  Parses TOML → PipelineDescription IR                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PipelineValidator                             │
│  C2: Module checks | C3: Property checks                        │
│  C4: Connection checks | C5: Graph checks                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ModuleRegistry                               │
│  Looks up module metadata, creates instances via factory        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ModuleFactory                                │
│  Creates modules | Applies properties | Connects pipeline       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Running Pipeline                             │
│  init() → run_all_threaded() → stop() → term()                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
base/
├── include/declarative/
│   ├── Metadata.h                 # Core type definitions
│   ├── ModuleRegistry.h           # Module registration
│   ├── ModuleRegistrationBuilder.h # Fluent builder API
│   ├── ModuleRegistrations.h      # Registration entry point
│   ├── FrameTypeRegistry.h        # Frame type hierarchy
│   ├── FrameTypeRegistrations.h   # Frame type entry point
│   ├── PipelineDescription.h      # IR types
│   ├── PipelineValidator.h        # Validation API
│   ├── ModuleFactory.h            # Factory API
│   ├── TomlParser.h               # TOML parsing
│   ├── Issue.h                    # Error/warning types
│   ├── PropertyMacros.h           # Property utilities
│   └── PropertyValidators.h       # Property validation
├── src/declarative/
│   ├── ModuleRegistrations.cpp    # All module registrations
│   ├── FrameTypeRegistrations.cpp # All frame type registrations
│   ├── ModuleRegistry.cpp
│   ├── FrameTypeRegistry.cpp
│   ├── PipelineDescription.cpp
│   ├── PipelineValidator.cpp
│   ├── ModuleFactory.cpp
│   └── TomlParser.cpp
├── test/declarative/              # Unit tests (268 tests)
└── tools/
    ├── aprapipes_cli.cpp          # CLI tool
    └── schema_generator.cpp       # Schema export
```

---

## CLI Commands

```bash
# Validate a pipeline
./aprapipes_cli validate pipeline.toml

# Run a pipeline
./aprapipes_cli run pipeline.toml

# List registered modules
./aprapipes_cli list-modules
./aprapipes_cli list-modules --category Source
./aprapipes_cli list-modules --tag opencv

# Describe a module
./aprapipes_cli describe FileReaderModule

# Generate schema
./apra_schema_generator --all --output-dir ./schema
```

---

## Success Metrics

### Sprint 1 ✅
- [x] 10+ unit tests passing (268 tests)
- [x] 2+ modules with Metadata (20 modules)
- [x] Parser handles all TOML features

### Sprint 2 ✅
- [x] End-to-end test: TOML → running pipeline
- [x] 5+ pilot modules registered (20 modules)
- [x] CLI has 4 commands working
- [x] Schema JSON generation working

### Sprint 3 (In Progress)
- [ ] Developer Guide complete
- [ ] Pipeline Author Guide complete
- [ ] 50+ modules registered (80%+ coverage)
- [ ] Example pipelines for all module categories
- [ ] Validator catches all common errors

---

## Next Steps

1. **Documentation** - Complete developer and author guides
2. **Module Expansion** - Register remaining 42 modules in batches
3. **Examples** - Create example pipelines for each batch
4. **Testing** - Move failing pipelines to not_working, fix issues
5. **CI Verification** - Ensure all platforms pass

---

## Risk Register

| Risk | Impact | Status |
|------|--------|--------|
| REGISTER_MODULE macro complexity | High | ✅ Resolved - Using fluent builder |
| ApraPipes connection API mismatch | Medium | ✅ Resolved - appendModule ordering fixed |
| Frame type compatibility | Medium | ✅ Resolved - Suggestion system implemented |
| CUDA module registration | Low | ⏳ Pending - needs #ifdef guards |
