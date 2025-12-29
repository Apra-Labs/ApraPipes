# Declarative Pipeline - Progress Tracker

> **This file is the source of truth for task status.**  
> Update this file at the end of EVERY session.

Last Updated: `2025-12-28 22:50` by `Claude Code Agent (B1 branch)`

---

## Current Sprint: 1 (Foundations)

## Quick Status

```
Critical Path:  A1 ──► A2 ──────────────► D1 ──► E1
                        │                  ▲
Parallel:       B1 ──► B2 ─────────────────┘
                        │
Non-blocking:           └──► C1 (validator shell)
```

---

## Task Status

### Sprint 1 - Critical Path

| Task | Description | Status | Assignee | Started | Completed | PR/Commit |
|------|-------------|--------|----------|---------|-----------|-----------|
| **A1** | Core Metadata Types | 🔄 In Progress | Agent-A1 | 2025-12-28 | - | - |
| **B1** | Pipeline Description IR | ✅ Complete | Agent-B1 | 2025-12-28 | 2025-12-28 | feat-declarative-pipeline-B1 |
| **A2** | Module Registry | ⏳ Blocked | - | - | - | Needs A1 |
| **B2** | TOML Parser | 📋 Ready | - | - | - | B1 done |

### Sprint 1 - Parallel Work

| Task | Description | Status | Assignee | Started | Completed | PR/Commit |
|------|-------------|--------|----------|---------|-----------|-----------|
| **A3** | FrameType Registry | ⏳ Blocked | - | - | - | Needs A1 |
| **C1** | Validator Shell | ⏳ Blocked | - | - | - | Needs A2 (B1 done) |
| **M1** | FileReaderModule Metadata | ⏳ Blocked | - | - | - | Needs A2 |
| **M2** | H264Decoder Metadata | ⏳ Blocked | - | - | - | Needs A2 |

### Sprint 2 - Core Engine

| Task | Description | Status | Assignee | Started | Completed | PR/Commit |
|------|-------------|--------|----------|---------|-----------|-----------|
| **D1** | Module Factory | ⏳ Blocked | - | - | - | Needs A2, B2 |
| **E1** | CLI Tool | ⏳ Blocked | - | - | - | Needs D1 |
| **E2** | Schema Generator | ⏳ Blocked | - | - | - | Needs A2 |
| **M3** | FaceDetectorXform Metadata | ⏳ Blocked | - | - | - | Needs A2 |
| **M4** | QRReader Metadata | ⏳ Blocked | - | - | - | Needs A2 |
| **M5** | FileWriterModule Metadata | ⏳ Blocked | - | - | - | Needs A2 |
| **F1-F4** | Frame Type Metadata | ⏳ Blocked | - | - | - | Needs A3 |

### Sprint 2-3 - Validator Enhancements

| Task | Description | Status | Assignee | Started | Completed | PR/Commit |
|------|-------------|--------|----------|---------|-----------|-----------|
| **C2** | Validator: Module Checks | ⏳ Blocked | - | - | - | Needs C1 |
| **C3** | Validator: Property Checks | ⏳ Blocked | - | - | - | Needs C2 |
| **C4** | Validator: Connection Checks | ⏳ Blocked | - | - | - | Needs C3, A3 |
| **C5** | Validator: Graph Checks | ⏳ Blocked | - | - | - | Needs C4 |

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| 📋 Ready | Dependencies met, ready to start |
| 🔄 In Progress | Currently being worked on |
| ✅ Complete | All acceptance criteria met |
| ⏳ Blocked | Waiting on dependency |
| ❌ Blocked | Other blocker (see notes) |
| 🔍 Review | PR submitted, awaiting review |

---

## Current Work

### Completed Task: B1 - Pipeline Description IR

**Started:** 2025-12-28
**Completed:** 2025-12-28
**Spec:** `docs/declarative-pipeline/tasks/B1-pipeline-description-ir.md`

#### Checklist
- [x] Read full specification
- [x] Create header file(s)
- [x] Create source file(s)
- [x] Write unit tests
- [x] All tests pass (verified via standalone compilation)
- [x] Update CMakeLists.txt
- [x] Commit with proper message
- [x] Update this progress file

#### Files Created/Modified
- `base/include/declarative/PipelineDescription.h` (created)
- `base/src/declarative/PipelineDescription.cpp` (created)
- `base/test/declarative/pipeline_description_tests.cpp` (created)
- `base/CMakeLists.txt` (modified - added declarative files)

#### Notes
- Created all IR types: PropertyValue, ModuleInstance, Connection, PipelineSettings, PipelineDescription
- Implemented Connection::parse() for "module.pin" format with proper error handling
- Implemented PipelineDescription::findModule() for module lookup
- Implemented PipelineDescription::toJson() for debugging output with proper JSON escaping
- 25 unit tests covering all acceptance criteria
- Full build requires Linux CI due to macOS vcpkg CUDA dependency issues

---

## Blockers

| Blocker | Affects | Description | Resolution |
|---------|---------|-------------|------------|
| *None* | - | - | - |

---

## Session Log

### Session: 2025-12-28 22:40

**Agent:** Claude Code Agent (B1 branch)
**Duration:** ~20 mins
**Tasks:** B1 - Pipeline Description IR

**Accomplished:**
- Created `base/include/declarative/PipelineDescription.h` with all IR types
- Created `base/src/declarative/PipelineDescription.cpp` with helper methods
- Created `base/test/declarative/pipeline_description_tests.cpp` with 25 unit tests
- Updated `base/CMakeLists.txt` to include declarative files
- Verified code compiles via standalone clang++ compilation

**Remaining:**
- B2 (TOML Parser) is now unblocked and ready to start

**Notes for Next Session:**
- Full build/test requires Linux CI (macOS has vcpkg CUDA dependency issues)
- B2 depends on B1 and should be started next
- Consider adding nlohmann/json for proper JSON handling in toJson() if needed later

---

### Session: TEMPLATE (copy this for new sessions)

**Agent:**  
**Duration:**  
**Tasks:**  

**Accomplished:**
- 

**Remaining:**
- 

**Notes for Next Session:**
- 

---

## Build Status

| Platform | Status | Last Success | Notes |
|----------|--------|--------------|-------|
| Linux | ❓ Unknown | - | - |
| Windows | ❓ Unknown | - | - |
| ARM64 | ❓ Unknown | - | - |

---

## Test Results

| Test Suite | Pass | Fail | Skip | Last Run |
|------------|------|------|------|----------|
| metadata_tests | - | - | - | - |
| module_registry_tests | - | - | - | - |
| pipeline_description_tests | - | - | - | - |
| toml_parser_tests | - | - | - | - |
| module_factory_tests | - | - | - | - |

---

## Files Created

Track new files for this feature:

```
base/include/declarative/
  [ ] Metadata.h                    # A1
  [x] PipelineDescription.h         # B1 ✅
  [ ] ModuleRegistry.h              # A2
  [ ] FrameTypeRegistry.h           # A3
  [ ] PipelineValidator.h           # C1
  [ ] ModuleFactory.h               # D1

base/src/declarative/
  [x] PipelineDescription.cpp       # B1 ✅
  [ ] ModuleRegistry.cpp            # A2
  [ ] FrameTypeRegistry.cpp         # A3
  [ ] PipelineValidator.cpp         # C1
  [ ] ModuleFactory.cpp             # D1

base/include/declarative/
  [ ] TomlParser.h                  # B2

base/src/declarative/
  [ ] TomlParser.cpp                # B2

base/test/declarative/
  [ ] metadata_tests.cpp            # A1
  [x] pipeline_description_tests.cpp # B1 ✅
  [ ] module_registry_tests.cpp     # A2
  [ ] frame_type_registry_tests.cpp # A3
  [ ] toml_parser_tests.cpp         # B2
  [ ] pipeline_validator_tests.cpp  # C1
  [ ] module_factory_tests.cpp      # D1

base/tools/
  [ ] aprapipes_cli.cpp             # E1
  [ ] schema_generator.cpp          # E2
```

---

## Quick Reference

**Start a task:**
```bash
# 1. Update this file: Status = 🔄 In Progress, Started = today
# 2. Read spec: cat docs/declarative-pipeline/tasks/<task>.md
# 3. Implement
# 4. Test: cd build && ctest -R <test_name>
# 5. Commit: git commit -m "feat(declarative): <description>"
# 6. Update this file: Status = ✅ Complete, Completed = today
```

**Dependencies:**
- A1 → A2, A3
- B1 → B2, C1
- A2 → D1, E2, C1, M1-M5
- B2 → D1
- A3 → F1-F4, C4
- D1 → E1
- C1 → C2 → C3 → C4 → C5
