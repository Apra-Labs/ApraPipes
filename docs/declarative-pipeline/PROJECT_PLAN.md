# Declarative Pipeline Construction - Project Plan

> Last Updated: 2026-01-05

## Project Status: Sprint 4 (Node.js Interface)

**Package:** `@apralabs/aprapipes`
**Format:** JSON only (TOML removed)

---

## Sprints Overview

| Sprint | Status | Theme |
|--------|--------|-------|
| Sprint 1 | ✅ Complete | Foundations (Metadata, Registry, Parser) |
| Sprint 2 | ✅ Complete | Core Engine (Factory, CLI, Validator) |
| Sprint 3 | ✅ Complete | Module Expansion (31/62 = 50%) |
| Sprint 4 | 🔄 In Progress | Node.js Addon |

---

## Sprint 4: Node.js Addon

### Phase 0-1: JSON Migration ✅ COMPLETE
- Removed all TOML code and dependencies
- Created JsonParser with 24 tests
- All 268+ tests pass

### Phase 2: Node Addon Foundation ✅ COMPLETE
- Added node-addon-api to vcpkg
- Created addon.cpp with basic exports
- CMake integration for .node build

### Phase 3: Core JS API ✅ COMPLETE
- `createPipeline()` - Create from JSON config
- `validatePipeline()` - Validate without running
- `listModules()` / `describeModule()` - Registry access
- `PipelineWrapper` class (init, run, stop, terminate, pause, play)
- `ModuleWrapper` class (id, type, getProps, isInputQueFull)
- TypeScript definitions

### Phase 4: Event System (Pending)
- Thread-safe callback bridge
- Health/Error callbacks to JS
- Pipeline lifecycle events (started, stopped, error)
- `pipeline.on(event, callback)` API

### Phase 5: Testing & Documentation (Pending)
- Node.js unit tests
- Integration tests
- Example applications
- API documentation

---

## Module Registration: 31/62 (50%)

| Category | Modules |
|----------|---------|
| Source | FileReaderModule, TestSignalGenerator, Mp4ReaderSource |
| Sink | FileWriterModule, StatSink, Mp4WriterSink |
| Transform | ImageDecoderCV, ImageEncoderCV, ImageResizeCV, RotateCV, ColorConversion, VirtualPTZ, TextOverlayXForm, BrightnessContrastControl, BMPConverter, AffineTransform |
| Analytics | FaceDetectorXform, QRReader, CalcHistogramCV, MotionVectorExtractor |
| Utility | ValveModule, Split, Merge, MultimediaQueueXform |

---

## JavaScript API

```typescript
import * as aprapipes from '@apralabs/aprapipes';

const pipeline = aprapipes.createPipeline({
  modules: {
    source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
    sink: { type: 'StatSink' }
  },
  connections: [{ from: 'source', to: 'sink' }]
});

await pipeline.init();
await pipeline.run();
// ... later
await pipeline.stop();
await pipeline.terminate();
```

---

## Key Files

```
base/
├── bindings/node/
│   ├── addon.cpp           # Main entry point
│   ├── pipeline_wrapper.*  # Pipeline class
│   └── module_wrapper.*    # ModuleHandle class
├── include/declarative/
│   ├── JsonParser.h
│   ├── ModuleRegistry.h
│   ├── ModuleFactory.h
│   └── PipelineValidator.h
└── src/declarative/
    └── *.cpp

types/aprapipes.d.ts        # TypeScript definitions
package.json                # npm package
```
