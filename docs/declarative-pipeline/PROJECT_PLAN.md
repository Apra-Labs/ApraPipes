# Declarative Pipeline Construction - Project Plan

## Sprints Overview

| Sprint | Duration | Theme | Key Deliverables |
|--------|----------|-------|------------------|
| **Sprint 1** | Week 1-2 | Foundations | Metadata, Registries, IR, Parser |
| **Sprint 2** | Week 3-4 | Core Engine | Factory, CLI, Pilot Modules |
| **Sprint 3** | Week 5+ | Polish | Validator rules, More modules, Docs |

---

## Sprint 1: Foundations (Week 1-2)

### Goals
- ✅ All type definitions in place
- ✅ Registries working with static registration
- ✅ TOML parsing to IR
- ✅ First modules with Metadata

### Critical Path (Must Complete)
| ID | Task | Days | Owner |
|----|------|------|-------|
| A1 | Core Metadata Types | 3 | Agent 1 |
| B1 | PipelineDescription IR | 2 | Agent 2 |
| A2 | Module Registry | 4 | Agent 1 |
| B2 | TOML Parser | 4 | Agent 2 |

### Parallel Work
| ID | Task | Days | Owner |
|----|------|------|-------|
| A3 | FrameType Registry | 3 | Agent 1 (after A1) |
| C1 | Validator Shell | 1 | Agent 2 (after B1) |
| M1 | FileReaderModule Metadata | 1 | Agent 3 (after A2) |
| M2 | H264Decoder Metadata | 1 | Agent 3 (after A2) |

### Sprint 1 Exit Criteria
- [ ] `Metadata.h` compiles with all types
- [ ] `REGISTER_MODULE` works in test
- [ ] `TomlParser` parses reference pipeline.toml
- [ ] At least 2 modules have Metadata
- [ ] Unit tests pass

---

## Sprint 2: Core Engine (Week 3-4)

### Goals
- ✅ Working end-to-end flow: TOML → Run
- ✅ CLI tool usable
- ✅ All 5 pilot modules with Metadata
- ✅ Schema generation working

### Critical Path
| ID | Task | Days | Owner |
|----|------|------|-------|
| D1 | Module Factory | 4 | Agent 1 |
| E1 | CLI Tool | 3 | Agent 1 (after D1) |

### Parallel Work
| ID | Task | Days | Owner |
|----|------|------|-------|
| E2 | Schema Generator | 2 | Agent 2 |
| M3 | FaceDetectorXform Metadata | 1 | Agent 3 |
| M4 | QRReader Metadata | 1 | Agent 3 |
| M5 | FileWriterModule Metadata | 1 | Agent 3 |
| F1-F4 | Frame Type Metadata | 2 | Agent 2 |
| C2 | Validator Module Checks | 1 | Agent 3 |
| C3 | Validator Property Checks | 2 | Agent 3 |

### Sprint 2 Exit Criteria
- [ ] `aprapipes validate pipeline.toml` works
- [ ] `aprapipes run pipeline.toml` starts pipeline
- [ ] `aprapipes list-modules` shows all modules
- [ ] All 5 pilot modules registered
- [ ] Schema JSON generated at build time
- [ ] Integration test passes

---

## Sprint 3: Polish (Week 5+)

### Goals
- ✅ Validator catches real errors
- ✅ Documentation generated
- ✅ Ready for team use

### Tasks
| ID | Task | Days | Owner |
|----|------|------|-------|
| C4 | Validator Connection Checks | 2 | - |
| C5 | Validator Graph Checks | 1 | - |
| - | Add Metadata to more modules | ongoing | - |
| - | Integration tests | 2 | - |
| - | Documentation | 1 | - |

---

## Task Dependency Graph

```
Week 1                    Week 2                    Week 3                    Week 4
──────────────────────────────────────────────────────────────────────────────────────

A1 ─────────► A2 ─────────────────────┐
(Metadata)    (Registry)               │
                │                      │
                ├───► A3               │
                │   (FrameTypes)       │
                │                      │
                ├───► M1, M2           │
                │   (Modules)          │
                │                      ▼
B1 ─────────► B2 ─────────────────────► D1 ─────────► E1
(IR)          (Parser)                  (Factory)     (CLI)
                │                       
                ├───► C1 ───► C2 ───► C3 ───► C4 ───► C5
                │   (Validator Shell → Enhancements)
                │
                └───► E2
                    (Schema Gen)
```

---

## GitHub Project Structure

### Columns
1. **Backlog** - All tasks not yet started
2. **In Progress** - Currently being worked on
3. **Review** - PR submitted, awaiting review
4. **Done** - Merged to main

### Labels
- `P0-critical` - Critical path, blocks others
- `P1-high` - High priority, should be in sprint
- `P2-medium` - Nice to have in sprint
- `P3-low` - Can defer
- `sprint-1` - Sprint 1 tasks
- `sprint-2` - Sprint 2 tasks
- `type:infrastructure` - Core framework
- `type:module` - Module metadata
- `type:tooling` - CLI, generators
- `type:validation` - Validator rules

### Milestones
- **v0.1-foundations** - Sprint 1 complete
- **v0.2-mvp** - Sprint 2 complete (MVP)
- **v0.3-polish** - Sprint 3 complete

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| REGISTER_MODULE macro complexity | High | Test early with simple mock module |
| ApraPipes connection API mismatch | Medium | Review existing test code patterns |
| TOML library integration | Low | toml++ is header-only, easy to integrate |
| Constexpr string handling | Medium | Use runtime strings in registry, constexpr in metadata |

---

## Success Metrics

### Sprint 1
- [ ] 10+ unit tests passing
- [ ] 2+ modules with Metadata
- [ ] Parser handles all TOML features

### Sprint 2 (MVP)
- [ ] End-to-end test: TOML → running pipeline
- [ ] 5 pilot modules registered
- [ ] CLI has 4 commands working
- [ ] Schema JSON < 100KB for 5 modules

### Sprint 3
- [ ] Validator catches 80% of common errors
- [ ] All existing modules have Metadata
- [ ] Documentation generated and readable
