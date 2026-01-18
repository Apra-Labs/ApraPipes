# Declarative Pipeline - Project Plan

> Last Updated: 2026-01-17

---

## Overview

The Declarative Pipeline project transforms ApraPipes from imperative C++ construction to declarative JSON configuration. The project is now in the SDK Packaging phase.

---

## Current Sprint: Sprint 10 - SDK Packaging

**Goal:** Create consistent SDK packaging across all 4 CI workflows.

**Objectives:**
1. Package all artifacts (CLI, Node addon, libraries, examples)
2. Works out of the box for end users
3. Enable GitHub Releases (Phase 2)

**Artifacts per platform:**
- `bin/` - CLI, test executable, Node addon, shared libraries
- `lib/` - Static libraries
- `include/` - Header files
- `examples/` - JSON pipeline examples, Node.js examples
- `data/` - Sample input files (frame.jpg, faces.jpg)
- `VERSION` - Version string
- `README.md` - SDK usage documentation

**Platform Matrix:**

| Component | Windows | Linux x64 | macOS | ARM64/Jetson |
|-----------|---------|-----------|-------|--------------|
| aprapipes_cli | ✅ | ✅ | ✅ | ✅ |
| aprapipes.node | ✅ | ✅ | ✅ | ✅ |
| libaprapipes | ✅ | ✅ | ✅ | ✅ |
| examples/basic | ✅ | ✅ | ✅ | ✅ |
| examples/cuda | ✅ | ✅ | ❌ | ✅ |
| examples/jetson | ❌ | ❌ | ❌ | ✅ |
| examples/node | ✅ | ✅ | ✅ | ✅ |

---

## Completed Sprints

### Sprint 9: Node.js on Jetson (J2)
**Completed:** 2026-01-17

- Fixed Node.js addon build on Jetson ARM64
- GCC 9 workaround: include Boost.Serialization in --whole-archive
- Node addon verified working on Jetson

### Sprint 8: Jetson Integration
**Completed:** 2026-01-16

- 8 Jetson modules registered (NvArgusCamera, NvV4L2Camera, etc.)
- L4TM libjpeg conflict resolved via dlopen wrapper
- DMABUF auto-bridging implemented
- 7 L4TM tests passing in CI

### Sprint 7: Auto-Bridging
**Completed:** 2026-01-13

- PipelineAnalyzer for automatic bridge insertion
- CudaMemCopy for HOST↔DEVICE transitions
- ColorConversion for pixel format mismatches

### Sprint 6: DRY Refactoring
**Completed:** 2026-01-12

- Fixed property defaults
- Type validation improvements

### Sprint 5: CUDA Modules
**Completed:** 2026-01-11

- 15 CUDA modules (NPP + NVCodec)
- Shared cudastream_sp mechanism

### Sprint 4: Node.js Addon
**Completed:** 2026-01-10

- @apralabs/aprapipes package
- Event system for health/errors
- Pipeline lifecycle management

### Sprints 1-3: Core Infrastructure
**Completed:** 2026-01-09

- ModuleRegistry, ModuleFactory, PipelineValidator
- JSON parser (TOML removed)
- CLI tool (aprapipes_cli)
- 37 cross-platform modules

---

## Architecture

### CI Workflows

| Workflow | Platform | Build Type |
|----------|----------|------------|
| CI-Windows | Windows x64 | CUDA + NoCUDA |
| CI-Linux | Linux x64 | CUDA + Docker |
| CI-Linux-ARM64 | Jetson ARM64 | CUDA (JetPack 5.0+) |
| CI-MacOSX-NoCUDA | macOS ARM64 | NoCUDA only |

### Reusable Workflows

| Workflow | Used By |
|----------|---------|
| build-test.yml | CI-Windows, CI-Linux |
| build-test-macosx.yml | CI-MacOSX-NoCUDA |
| build-test-lin.yml | CI-Linux-ARM64 |
| CI-CUDA-Tests.yml | GPU tests on self-hosted |

---

## Key Decisions

1. **SDK naming:** Fixed names for CI (`aprapipes-sdk-{platform}`), versioned for releases
2. **Include unit tests:** Yes, for installation validation
3. **Data files:** Minimal set (frame.jpg, faces.jpg, ~202KB)
4. **Versioning:** `{major}.{minor}.{patch}-g{short-hash}`
5. **GPU test impact:** No breaking changes - fixed artifact names preserved

---

## References

- [SDK_PACKAGING_PLAN.md](./SDK_PACKAGING_PLAN.md) - Detailed packaging plan
- [PROGRESS.md](./PROGRESS.md) - Current sprint progress
- [JETSON_KNOWN_ISSUES.md](./JETSON_KNOWN_ISSUES.md) - Jetson platform issues
