# Declarative Pipeline - Project Plan

> Last Updated: 2026-01-16

**Branch:** `feat-declarative-pipeline-v2`
**Progress:** See [PROGRESS.md](./PROGRESS.md) for detailed status

---

## Sprints Overview

| Sprint | Status | Theme |
|--------|--------|-------|
| Sprint 1 | ✅ Complete | Foundations (Metadata, Registry, Parser) |
| Sprint 2 | ✅ Complete | Core Engine (Factory, CLI, Validator) |
| Sprint 3 | ✅ Complete | Module Expansion (37 modules) |
| Sprint 4 | ✅ Complete | Node.js Addon |
| Sprint 5 | ✅ Complete | CUDA Module Registration (15 modules) |
| Sprint 6 | ✅ Complete | DRY Refactoring |
| Sprint 7 | ✅ Complete | Auto-Bridging (Memory + Pixel Format) |
| Sprint 8 | ✅ Complete | Jetson Integration |
| Sprint 9 | ✅ Complete | Node.js Addon on Jetson (J2) |

---

## Sprint 8: Jetson Integration

> Started: 2026-01-13 | Completed: 2026-01-16

**Documentation:** [JETSON_KNOWN_ISSUES.md](./JETSON_KNOWN_ISSUES.md)

### Objectives

1. Register Jetson-specific modules (NvArgus, NvV4L2, L4TM, etc.)
2. Add DMABUF memory type bridging support
3. Create Jetson JSON pipeline examples
4. Test Node.js addon on Jetson

### Status

| Phase | Status | Description |
|-------|--------|-------------|
| Prerequisites | ✅ Done | CI, SSH, workspace setup |
| Phase 1 | ✅ Done | Jetson build with declarative pipeline |
| Phase 2 | ✅ Done | Register 8 Jetson modules |
| Phase 2.5 | ✅ Done | DMABUF auto-bridging in PipelineAnalyzer |
| Phase 3 | ✅ Done | JSON examples (L4TM working via dlopen) |
| Phase 4 | ✅ Done | Documentation, CI re-enabled |

### Registered Jetson Modules

| Module | MemType | Description |
|--------|---------|-------------|
| NvArgusCamera | DMABUF | CSI camera via Argus API |
| NvV4L2Camera | DMABUF | USB camera via V4L2 |
| NvTransform | DMABUF | GPU resize/crop/transform |
| JPEGDecoderL4TM | HOST | L4T hardware JPEG decoder ✅ |
| JPEGEncoderL4TM | HOST | L4T hardware JPEG encoder ✅ |
| EglRenderer | DMABUF | EGL display output |
| DMAFDToHostCopy | DMABUF→HOST | DMA buffer bridge |

### Known Issues

| Issue | Status | Workaround |
|-------|--------|------------|
| ~~J1: libjpeg conflict~~ | ✅ RESOLVED | dlopen wrapper isolates symbols |
| **J2: Boost.Serialization** | Open | Use CLI instead of Node.js |
| **J3: H264EncoderV4L2** | Open | Use H264EncoderNVCodec |

See [JETSON_KNOWN_ISSUES.md](./JETSON_KNOWN_ISSUES.md) for detailed analysis.

### Outcome

Sprint 8 is **complete**:
- ✅ Jetson modules registered and building
- ✅ DMABUF bridging implemented
- ✅ CI re-enabled and passing
- ✅ L4TM modules working (7 tests passing)
- ✅ L4TM CLI pipelines working on Jetson
- ⚠️ Node.js addon blocked by linking issue (J2)

---

## Sprint 9: Node.js Addon on Jetson (J2)

> Started: 2026-01-16 | Completed: 2026-01-17

**Documentation:** [JETSON_KNOWN_ISSUES.md](./JETSON_KNOWN_ISSUES.md) → Issue J2

### Objective

Fix the Node.js addon (`aprapipes.node`) to build and load correctly on Jetson ARM64.

### Solution

Added GCC version check in `CMakeLists.txt` (Option A):
- For GCC < 10: Include `Boost_SERIALIZATION_LIBRARY` in `--whole-archive`
- For GCC 10+: Standard linking (workaround not needed)

This is version-gated so it automatically stops being applied when upgrading to JetPack 6.x (which has GCC 11+).

### Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Added GCC 9 workaround in CMakeLists.txt |
| Phase 2 | ✅ Complete | Build succeeded on Jetson (1h 21m) |
| Phase 3 | ✅ Complete | CI-Linux-ARM64 passed |
| Phase 4 | ✅ Complete | Node addon loads and works on Jetson |

### Outcome

Node.js addon now works on Jetson ARM64:
```
node -e "require('./build/aprapipes.node')"
SUCCESS: Node addon loaded!
Methods: [ 'getVersion', 'listModules', 'describeModule', 'validatePipeline', ... ]
```

---

## Completed Sprints (Summary)

### Sprint 7: Auto-Bridging
- PipelineAnalyzer detects memory/format mismatches
- ModuleFactory auto-inserts bridge modules (CudaMemCopy, ColorConversion)
- Design doc: [CUDA_MEMTYPE_DESIGN.md](./CUDA_MEMTYPE_DESIGN.md)

### Sprint 5: CUDA Modules
- 15 CUDA modules registered (NPP, NVCodec, nvJPEG)
- Shared cudastream_sp across pipeline
- Examples: `examples/cuda/`

### Sprint 4: Node.js Addon
- `@apralabs/aprapipes` npm package
- createPipeline(), Pipeline class, event system
- Examples: `examples/node/`

### Sprints 1-3: Core Infrastructure
- ModuleRegistry, ModuleFactory, PipelineValidator
- JSON parser, CLI tool, schema generator
- 37 cross-platform modules registered
