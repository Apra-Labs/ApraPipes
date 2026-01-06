# Declarative Pipeline - Progress Tracker

> Last Updated: 2026-01-06

---

## Current Status

**Sprints 1-3:** ✅ COMPLETE
**Sprint 4 (Node.js):** 🔄 Phase 5 in progress

```
Core Infrastructure:  ✅ Complete (Metadata, Registry, Factory, Validator, CLI)
JSON Migration:       ✅ Complete (TOML removed, JsonParser added)
Module Coverage:      31/62 (50%)
Node.js Addon:        ✅ Phase 4.5 complete (events + dynamic props)
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
| Phase 5 | 🔄 In Progress | Testing & docs |

---

## Pending Work

### Module Registration (Batch 3-5)
- Batch 3: Sink modules (RTSPPusher, etc.)
- Batch 4: CUDA modules (H264Encoder, JPEGDecoder, etc.)
- Batch 5: Remaining utility modules

### Node.js Addon (Phase 5 - Remaining)
- ~~Node.js unit tests~~ ✅ event_tests.js, ptz_dynamic_props_test.js
- Example applications
- API documentation

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
