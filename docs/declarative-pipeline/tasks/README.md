# Declarative Pipeline Construction - Task Specifications

> Full specifications for Claude Code agents and developers.  
> GitHub Issues link here for the complete implementation details.

## Quick Reference

| ID | Task | Effort | Sprint | Spec |
|----|------|--------|--------|------|
| **A1** | Core Metadata Types | 3d | 1 | [A1-core-metadata-types.md](./A1-core-metadata-types.md) |
| **A2** | Module Registry | 4d | 1 | [A2-module-registry.md](./A2-module-registry.md) |
| **A3** | FrameType Registry | 3d | 1 | [A3-frame-type-registry.md](./A3-frame-type-registry.md) |
| **B1** | Pipeline Description IR | 2d | 1 | [B1-pipeline-description-ir.md](./B1-pipeline-description-ir.md) |
| **B2** | TOML Parser | 4d | 1 | [B2-toml-parser.md](./B2-toml-parser.md) |
| **C1** | Validator Shell | 1d | 1 | [C1-validator-shell.md](./C1-validator-shell.md) |
| **C2-C5** | Validator Enhancements | 6d | 2-3 | [C2-C5-validator-enhancements.md](./C2-C5-validator-enhancements.md) |
| **D1** | Module Factory | 4d | 2 | [D1-module-factory.md](./D1-module-factory.md) |
| **E1** | CLI Tool | 3d | 2 | [E1-cli-tool.md](./E1-cli-tool.md) |
| **E2** | Schema Generator | 2d | 2 | [E2-schema-generator.md](./E2-schema-generator.md) |
| **M1-M5** | Pilot Module Metadata | 5d | 1-2 | [M1-M5-pilot-modules.md](./M1-M5-pilot-modules.md) |
| **F1-F4** | Frame Type Metadata | 2d | 2 | [F1-F4-frame-types.md](./F1-F4-frame-types.md) |

## Dependency Graph

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
                │   (Validator - non-blocking, incremental)
                │
                └───► E2
                    (Schema Gen)
```

## Critical Path

**Minimum to reach MVP:**
```
A1 → A2 → D1 → E1
      ↑
B1 → B2
```

Validator (C1-C5) runs in parallel and does NOT block Factory.

## For Claude Code Agents

When assigned a task:

1. **Read the full spec** in this directory
2. **Check dependencies** - ensure prerequisite tasks are complete
3. **Follow the file paths** specified in "Implementation Notes"
4. **Run the acceptance criteria** - all tests must pass
5. **Check Definition of Done** before marking complete

Example:
```bash
# Starting task A1
cat docs/declarative-pipeline/tasks/A1-core-metadata-types.md

# Implement in specified location
# base/include/core/Metadata.h

# Run tests
cd build && ctest -R metadata
```

## Sprint Breakdown

### Sprint 1 (Week 1-2): Foundations
- [x] A1: Core Metadata Types
- [x] B1: Pipeline Description IR
- [x] A2: Module Registry  
- [x] B2: TOML Parser
- [x] A3: FrameType Registry
- [x] C1: Validator Shell
- [x] M1: FileReaderModule Metadata
- [x] M2: H264Decoder Metadata

### Sprint 2 (Week 3-4): Core Engine
- [ ] D1: Module Factory
- [ ] E1: CLI Tool
- [ ] E2: Schema Generator
- [ ] M3-M5: Remaining Pilot Modules
- [ ] F1-F4: Frame Type Metadata
- [ ] C2-C3: Validator Enhancements

### Sprint 3 (Week 5+): Polish
- [ ] C4-C5: Connection & Graph Validation
- [ ] Additional modules
- [ ] Documentation
- [ ] Integration tests

## Related Documents

- [RFC v4](../RFC.md) - Full RFC document
- [Project Plan](../PROJECT_PLAN.md) - Sprint overview and risk register
- [Discussion #471](https://github.com/Apra-Labs/ApraPipes/discussions/471) - Original RFC discussion
