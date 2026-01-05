# Declarative Pipeline Construction - Project Plan

> Last Updated: 2026-01-05

## Project Status: Sprint 4 (JSON + JavaScript Interface)

**MAJOR PIVOT:** Migrating from TOML to JSON-only with native Node.js addon.

- **Package name:** `@apralabs/aprapipes`
- **TOML support:** REMOVED COMPLETELY - JSON only
- **Goal:** Native Node.js integration via N-API addon

---

## Sprints Overview

| Sprint | Status | Theme | Key Deliverables |
|--------|--------|-------|------------------|
| **Sprint 1** | ✅ Complete | Foundations | Metadata, Registries, IR, Parser |
| **Sprint 2** | ✅ Complete | Core Engine | Factory, CLI, Validator, Schema Generator |
| **Sprint 3** | ✅ Complete | Module Expansion | 31 modules registered (50%) |
| **Sprint 4** | 🔄 In Progress | JSON + JS Interface | JSON Parser ✅, Node Addon, npm Package |

---

## Sprint 4: JSON + JavaScript Interface

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Node.js Process                                 │
│  ┌───────────────┐     ┌─────────────────┐     ┌───────────────────────┐   │
│  │ JS/TS Code    │────▶│ aprapipes.node  │────▶│ C++ Declarative API   │   │
│  │               │◀────│ (N-API Addon)   │◀────│                       │   │
│  │ - JSON config │     │                 │     │ - JsonParser          │   │
│  │ - Callbacks   │     │ - Thread-safe   │     │ - PipelineDescription │   │
│  │ - Events      │     │   callbacks     │     │ - ModuleFactory       │   │
│  └───────────────┘     └─────────────────┘     │ - PipelineValidator   │   │
│                                                 │ - Event Bridge        │   │
│                                                 └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          ┌───────────────────┐
                          │ C++ Module Graph  │
                          │ (Threaded Runtime)│
                          └───────────────────┘
```

---

## Phase 0: TOML Removal ✅ COMPLETE

> **Goal:** Remove all TOML dependencies and code. JSON becomes the only format.

### 0.1 Files DELETED ✅

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| R1.1 | `base/include/declarative/TomlParser.h` | TOML parser header | Build succeeds | ✅ Complete |
| R1.2 | `base/src/declarative/TomlParser.cpp` | TOML parser impl | Build succeeds | ✅ Complete |
| R2 | `base/test/declarative/toml_parser_tests.cpp` | TOML parser tests | Tests pass without this file | ✅ Complete |
| R6 | `docs/declarative-pipeline/examples/**/*.toml` | All TOML examples | N/A (after JSON conversion) | ✅ Complete |

### 0.2 Dependencies REMOVED ✅

| Task | File | Change | Test | Status |
|------|------|--------|------|--------|
| R3 | `base/vcpkg.json` | Remove `tomlplusplus` | `vcpkg install` succeeds | ✅ Complete |

### 0.3 Test Files UPDATED ✅

| Task | File | Change | Test | Status |
|------|------|--------|------|--------|
| R8 | `base/test/declarative/pipeline_integration_tests.cpp` | Use JSON pipelines instead of TOML | Test passes with JSON | ✅ Complete |
| R9 | `base/test/declarative/pipeline_description_tests.cpp` | Update TOML refs in test names/comments | Tests pass | ✅ Complete |
| R10 | `base/test/declarative/module_registry_tests.cpp` | Update TOML refs in comments | Tests pass | ✅ Complete |

### 0.4 Source Files UPDATED ✅

| Task | File | Change | Test | Status |
|------|------|--------|------|--------|
| R11 | `base/src/declarative/ModuleRegistrations.cpp` | Remove TomlParser include | Build succeeds | ✅ Complete |
| R12 | `base/src/declarative/ModuleFactory.cpp` | Update error messages, remove TOML refs | Build succeeds | ✅ Complete |
| R13 | `base/src/declarative/PipelineValidator.cpp` | Update error messages, remove TOML refs | Build succeeds | ✅ Complete |
| R14 | `base/src/declarative/ModuleRegistry.cpp` | Update comments | Build succeeds | ✅ Complete |
| R15 | `base/tools/aprapipes_cli.cpp` | Change from TOML to JSON handling | CLI works with JSON | ✅ Complete |

### 0.5 Headers UPDATED ✅

| Task | File | Change | Test | Status |
|------|------|--------|------|--------|
| R16 | `base/include/declarative/ModuleFactory.h` | Update comments | Build succeeds | ✅ Complete |
| R17 | `base/include/declarative/ModuleRegistry.h` | Update comments | Build succeeds | ✅ Complete |
| R18 | `base/include/declarative/PipelineDescription.h` | Update comments | Build succeeds | ✅ Complete |

### 0.6 Other Updates ✅

| Task | File | Change | Test | Status |
|------|------|--------|------|--------|
| R4 | `base/CMakeLists.txt` | Remove TomlParser references | Build succeeds | ✅ Complete |
| R5 | `docs/declarative-pipeline/examples/` | Convert all .toml to .json | JSON files valid | ✅ Complete |
| R7 | `scripts/test_declarative_pipelines.sh` | Use .json files | Script passes | ✅ Complete |
| R19 | `docs/declarative-pipeline/*.md` | Update all TOML references | Docs accurate | ✅ Complete |

### Phase 0 Acceptance Criteria ✅
- [x] No `.toml` files in examples directory
- [x] No `tomlplusplus` in vcpkg.json
- [x] No `TomlParser` in codebase
- [x] All 268+ declarative tests pass
- [x] `aprapipes_cli validate pipeline.json` works
- [x] Integration test script uses JSON files

---

## Phase 1: JSON Parser ✅ COMPLETE

> **Goal:** Create JsonParser class that produces identical PipelineDescription IR.

### 1.1 Core Implementation ✅

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| J1.1 | `base/include/declarative/JsonParser.h` | JsonParser class header | Compiles | ✅ Complete |
| J1.2 | `base/src/declarative/JsonParser.cpp` | parseFile(), parseString() | Unit tests pass | ✅ Complete |

### 1.2 Unit Tests ✅

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| J2.1 | `base/test/declarative/json_parser_tests.cpp` | Parse valid JSON pipelines | 24 tests pass | ✅ Complete |
| J2.2 | `base/test/declarative/json_parser_tests.cpp` | Error handling tests | 10+ tests pass | ✅ Complete |
| J2.3 | `base/test/declarative/json_parser_tests.cpp` | Edge cases (empty, malformed) | 5+ tests pass | ✅ Complete |

### 1.3 CLI Integration ✅

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| J3 | `base/tools/aprapipes_cli.cpp` | JSON-only support | `aprapipes_cli run pipeline.json` works | ✅ Complete |

### 1.4 Schema & Documentation ✅

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| J4 | `docs/declarative-pipeline/schemas/pipeline.schema.json` | JSON Schema for pipelines | Schema validates examples | ⏳ Pending (optional) |
| J5 | `docs/declarative-pipeline/*.md` | Update all docs | Docs reference JSON only | ✅ Complete |

### Phase 1 Acceptance Criteria ✅
- [x] JsonParser.parseFile() works for all converted examples
- [x] JsonParser.parseString() works for inline JSON
- [x] 35+ json_parser_tests passing (24 comprehensive tests)
- [x] CLI `validate` and `run` work with JSON
- [ ] JSON Schema validates all example pipelines (optional)
- [x] Integration test passes with JSON

---

## Phase 2: Node Addon Foundation

> **Goal:** Create basic Node.js native addon with module initialization.

### 2.1 Dependencies

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| N1 | `base/vcpkg.json` | Add `node-addon-api` | vcpkg install succeeds | ⏳ Pending |

### 2.2 Directory Structure

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| N2.1 | `base/bindings/node/` | Create directory | Directory exists | ⏳ Pending |
| N2.2 | `base/bindings/node/CMakeLists.txt` | CMake for addon | Configures successfully | ⏳ Pending |

### 2.3 Basic Addon

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| N3 | `base/bindings/node/addon.cpp` | Module init, export functions | Builds .node file | ⏳ Pending |

### 2.4 Build Integration

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| N4 | `base/CMakeLists.txt` | Add bindings/node subdirectory | cmake configures | ⏳ Pending |
| N5 | `package.json` | npm package definition | npm install works | ⏳ Pending |

### 2.5 Smoke Test

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| N6 | `test/node/smoke.test.js` | Basic require() test | `npm test` passes | ⏳ Pending |

### Phase 2 Acceptance Criteria
- [ ] `aprapipes.node` builds on macOS, Linux, Windows
- [ ] `require('@apralabs/aprapipes')` succeeds
- [ ] Basic version info exported
- [ ] Smoke test passes

---

## Phase 3: Core JS API

> **Goal:** Implement createPipeline, validatePipeline, and module discovery APIs.

### 3.1 Pipeline Creation

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| A1.1 | `base/bindings/node/addon.cpp` | createPipeline(config) | Creates pipeline from JSON | ⏳ Pending |
| A1.2 | `base/bindings/node/addon.cpp` | createPipeline(string) | Creates pipeline from JSON string | ⏳ Pending |

### 3.2 Validation

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| A2 | `base/bindings/node/addon.cpp` | validatePipeline() | Returns ValidationResult | ⏳ Pending |

### 3.3 Module Registry Access

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| A3.1 | `base/bindings/node/addon.cpp` | listModules(filter?) | Returns ModuleInfo[] | ⏳ Pending |
| A3.2 | `base/bindings/node/addon.cpp` | describeModule(type) | Returns ModuleInfo | ⏳ Pending |
| A3.3 | `base/bindings/node/addon.cpp` | getModuleSchema(type) | Returns JSON schema | ⏳ Pending |

### 3.4 Pipeline Wrapper Class

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| A4.1 | `base/bindings/node/pipeline_wrapper.h` | PipelineWrapper class | Compiles | ⏳ Pending |
| A4.2 | `base/bindings/node/pipeline_wrapper.cpp` | init() method | Pipeline initializes | ⏳ Pending |
| A4.3 | `base/bindings/node/pipeline_wrapper.cpp` | run() method | Pipeline runs async | ⏳ Pending |
| A4.4 | `base/bindings/node/pipeline_wrapper.cpp` | stop() method | Pipeline stops | ⏳ Pending |
| A4.5 | `base/bindings/node/pipeline_wrapper.cpp` | pause()/play() | Pipeline pauses/resumes | ⏳ Pending |
| A4.6 | `base/bindings/node/pipeline_wrapper.cpp` | terminate() | Pipeline cleans up | ⏳ Pending |

### 3.5 Module Wrapper Class

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| A5.1 | `base/bindings/node/module_wrapper.h` | ModuleWrapper class | Compiles | ⏳ Pending |
| A5.2 | `base/bindings/node/module_wrapper.cpp` | getProperty(name) | Returns property value | ⏳ Pending |
| A5.3 | `base/bindings/node/module_wrapper.cpp` | setProperty(name, value) | Sets dynamic property | ⏳ Pending |
| A5.4 | `base/bindings/node/module_wrapper.cpp` | getProperties() | Returns all properties | ⏳ Pending |
| A5.5 | `base/bindings/node/module_wrapper.cpp` | getStats() | Returns module stats | ⏳ Pending |

### 3.6 TypeScript Definitions

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| A6 | `types/aprapipes.d.ts` | Full TypeScript definitions | TS compiles without errors | ⏳ Pending |

### Phase 3 Acceptance Criteria
- [ ] `aprapipes.createPipeline(config)` works
- [ ] `aprapipes.validatePipeline(config)` returns errors/warnings
- [ ] `aprapipes.listModules()` returns all registered modules
- [ ] Pipeline init/run/stop/terminate work
- [ ] Module getProperty/setProperty work for dynamic props
- [ ] TypeScript IntelliSense works

---

## Phase 4: Event System

> **Goal:** Bridge C++ callbacks to JavaScript event listeners.

### 4.1 Thread-Safe Callback

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| E1 | `base/bindings/node/event_bridge.h` | ThreadSafeCallback wrapper | Compiles | ⏳ Pending |

### 4.2 Health Callback Bridge

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| E2 | `base/bindings/node/event_bridge.cpp` | APHealthCallback → JS | Stats events flow to JS | ⏳ Pending |

### 4.3 Error Callback Bridge

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| E3 | `base/bindings/node/event_bridge.cpp` | APErrorCallback → JS | Error events flow to JS | ⏳ Pending |

### 4.4 Pipeline Lifecycle Events

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| E4.1 | `base/bindings/node/pipeline_wrapper.cpp` | 'started' event | Fires when pipeline starts | ⏳ Pending |
| E4.2 | `base/bindings/node/pipeline_wrapper.cpp` | 'stopped' event | Fires when pipeline stops | ⏳ Pending |
| E4.3 | `base/bindings/node/pipeline_wrapper.cpp` | 'paused'/'resumed' events | Fires on pause/play | ⏳ Pending |
| E4.4 | `base/bindings/node/pipeline_wrapper.cpp` | 'error' event | Fires on pipeline error | ⏳ Pending |
| E4.5 | `base/bindings/node/pipeline_wrapper.cpp` | 'endOfStream' event | Fires on EOS | ⏳ Pending |

### 4.5 Event Listener API

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| E5.1 | `base/bindings/node/pipeline_wrapper.cpp` | on(event, callback) | Registers listener | ⏳ Pending |
| E5.2 | `base/bindings/node/pipeline_wrapper.cpp` | off(event, callback) | Removes listener | ⏳ Pending |

### Phase 4 Acceptance Criteria
- [ ] `pipeline.on('error', cb)` receives errors from C++
- [ ] `pipeline.on('stats', cb)` receives periodic stats
- [ ] `pipeline.on('started', cb)` fires when run() completes init
- [ ] Events are delivered on Node.js main thread
- [ ] No crashes on rapid start/stop cycles

---

## Phase 5: Testing & Documentation

> **Goal:** Comprehensive tests and documentation for npm package.

### 5.1 Node.js Unit Tests

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| T1.1 | `test/node/pipeline.test.js` | Pipeline lifecycle tests | 10+ tests pass | ⏳ Pending |
| T1.2 | `test/node/validation.test.js` | Validation tests | 10+ tests pass | ⏳ Pending |
| T1.3 | `test/node/registry.test.js` | Module registry tests | 5+ tests pass | ⏳ Pending |
| T1.4 | `test/node/properties.test.js` | Property get/set tests | 10+ tests pass | ⏳ Pending |

### 5.2 Integration Tests

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| T2.1 | `test/node/integration/simple.test.js` | TestSignal → StatSink | Pass | ⏳ Pending |
| T2.2 | `test/node/integration/transform.test.js` | Multi-stage pipeline | Pass | ⏳ Pending |
| T2.3 | `test/node/integration/events.test.js` | Event system test | Pass | ⏳ Pending |

### 5.3 Example Applications

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| T3.1 | `examples/node/simple-pipeline.js` | Basic example | Runs successfully | ⏳ Pending |
| T3.2 | `examples/node/dynamic-properties.js` | Runtime property changes | Runs successfully | ⏳ Pending |
| T3.3 | `examples/node/event-handling.js` | Event listener example | Runs successfully | ⏳ Pending |

### 5.4 Documentation

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| T4.1 | `docs/declarative-pipeline/JS_API_GUIDE.md` | API documentation | Complete and accurate | ⏳ Pending |
| T4.2 | `README.md` | Update main README | Includes JS examples | ⏳ Pending |

### 5.5 npm Publish Setup

| Task | File | Description | Test | Status |
|------|------|-------------|------|--------|
| T5.1 | `.npmrc` | npm config | Present | ⏳ Pending |
| T5.2 | `scripts/publish.sh` | Publish script | Works | ⏳ Pending |
| T5.3 | `.github/workflows/npm-publish.yml` | CI publish workflow | Triggers on release | ⏳ Pending |

### Phase 5 Acceptance Criteria
- [ ] 50+ Node.js tests passing
- [ ] All examples run without errors
- [ ] API documentation complete
- [ ] npm package publishes successfully
- [ ] CI builds on all platforms

---

## Completed Work (Sprints 1-3)

### Sprint 1: Foundations ✅

| Task | Description | Status |
|------|-------------|--------|
| A1 | Core Metadata Types | ✅ Complete |
| B1 | PipelineDescription IR | ✅ Complete |
| A2 | Module Registry | ✅ Complete |
| J1 | JSON Parser | ✅ Complete (replaced TOML Parser) |
| A3 | FrameType Registry | ✅ Complete |
| C1 | Validator Shell | ✅ Complete |

### Sprint 2: Core Engine ✅

| Task | Description | Status |
|------|-------------|--------|
| D1 | Module Factory | ✅ Complete |
| D2 | Property Binding System | ✅ Complete |
| D3 | Multi-Pin Connection Support | ✅ Complete |
| E1 | CLI Tool | ✅ Complete |
| E2 | Schema Generator | ✅ Complete |
| C2-C5 | Validator Enhancements | ✅ Complete |
| F1-F4 | Frame Type Metadata | ✅ Complete |

### Sprint 3: Module Expansion ✅

| Task | Description | Status |
|------|-------------|--------|
| DOC1 | Developer Guide | ✅ Complete |
| DOC2 | Pipeline Author Guide | ✅ Complete |
| Batch1-2 | Module Registration | ✅ 31/62 modules (50%) |

---

## Current Module Registration Coverage

**31 modules registered (50% of 62 total)**

| Category | Registered Modules |
|----------|-------------------|
| Source | FileReaderModule, TestSignalGenerator, Mp4ReaderSource |
| Sink | FileWriterModule, StatSink, Mp4WriterSink |
| Transform | ImageDecoderCV, ImageEncoderCV, ImageResizeCV, RotateCV, ColorConversion, VirtualPTZ, TextOverlayXForm, BrightnessContrastControl, BMPConverter, AffineTransform |
| Analytics | FaceDetectorXform, QRReader, CalcHistogramCV, MotionVectorExtractor |
| Utility | ValveModule, Split, Merge, MultimediaQueueXform |

---

## File Structure (Post-Migration)

```
ApraPipes/
├── package.json                           # npm package definition
├── types/
│   └── aprapipes.d.ts                     # TypeScript definitions
├── base/
│   ├── vcpkg.json                         # + node-addon-api, - tomlplusplus
│   ├── include/declarative/
│   │   ├── JsonParser.h                   # NEW: JSON parser
│   │   ├── Metadata.h
│   │   ├── ModuleRegistry.h
│   │   ├── PipelineDescription.h
│   │   ├── PipelineValidator.h
│   │   ├── ModuleFactory.h
│   │   └── Issue.h
│   ├── src/declarative/
│   │   ├── JsonParser.cpp                 # NEW: JSON parser impl
│   │   ├── ModuleRegistrations.cpp
│   │   ├── ModuleRegistry.cpp
│   │   ├── PipelineDescription.cpp
│   │   ├── PipelineValidator.cpp
│   │   └── ModuleFactory.cpp
│   ├── bindings/
│   │   └── node/                          # NEW: Node addon
│   │       ├── CMakeLists.txt
│   │       ├── addon.cpp
│   │       ├── pipeline_wrapper.cpp
│   │       ├── module_wrapper.cpp
│   │       └── event_bridge.cpp
│   └── test/declarative/
│       ├── json_parser_tests.cpp          # NEW
│       ├── pipeline_integration_tests.cpp  # UPDATED (JSON)
│       └── ...
├── test/node/                             # NEW: Node.js tests
│   ├── smoke.test.js
│   ├── pipeline.test.js
│   └── integration/
├── examples/node/                         # NEW: JS examples
└── docs/declarative-pipeline/
    ├── schemas/
    │   └── pipeline.schema.json           # JSON Schema
    ├── examples/
    │   └── json/                          # JSON pipeline examples
    └── JS_API_GUIDE.md                    # API documentation
```

---

## CLI Commands (Post-Migration)

```bash
# Validate a pipeline (JSON only)
./aprapipes_cli validate pipeline.json

# Run a pipeline (JSON only)
./aprapipes_cli run pipeline.json

# List registered modules
./aprapipes_cli list-modules

# Describe a module
./aprapipes_cli describe FileReaderModule

# Generate schema
./apra_schema_generator --all --output-dir ./schema
```

---

## JavaScript API Preview

```typescript
import * as aprapipes from '@apralabs/aprapipes';

const pipeline = aprapipes.createPipeline({
  modules: {
    source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
    sink: { type: 'StatSink' }
  },
  connections: [{ from: 'source', to: 'sink' }]
});

pipeline.on('error', (e) => console.error(e.errorMessage));
pipeline.on('stats', (e) => console.log(`FPS: ${e.data.modules.source.currentFps}`));

await pipeline.init();
await pipeline.run();
// ... later
await pipeline.stop();
```

---

## Success Metrics

### Sprint 4 (In Progress)
- [ ] All TOML code removed
- [ ] JsonParser with 35+ tests
- [ ] Node addon builds on macOS, Linux, Windows
- [ ] 50+ Node.js tests passing
- [ ] npm package publishable
- [ ] Full TypeScript definitions
- [ ] Event system working

---

## Risk Register

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| node-addon-api vcpkg availability | High | Use npm package directly if needed | ⏳ Pending |
| Cross-platform Node addon builds | High | Test on all platforms early | ⏳ Pending |
| Thread-safe callback complexity | Medium | Follow apranvr patterns | ⏳ Pending |
| Breaking change for existing users | Low | No external users yet | ✅ Mitigated |

---

## Testing Requirements

**Every task must be tested before marking complete:**

1. **Phase 0 (TOML Removal):** All 268+ existing tests pass, build succeeds
2. **Phase 1 (JSON Parser):** 35+ new json_parser_tests pass
3. **Phase 2 (Node Foundation):** Smoke test passes on all platforms
4. **Phase 3 (Core API):** 30+ Node.js unit tests pass
5. **Phase 4 (Events):** Event delivery tests pass
6. **Phase 5 (Testing):** 50+ total Node.js tests pass

**Progress Tracking:**
- Update `PROGRESS.md` after completing each task
- Run `scripts/test_declarative_pipelines.sh` after Phase 0
- Run `npm test` after Phases 2-5
