# Declarative Pipeline - Project Plan

> Last Updated: 2026-01-15

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
| Sprint 8 | ⚠️ Partial | Jetson Integration (Known Issues) |

---

## Sprint 8: Jetson Integration

> Started: 2026-01-13

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
| Phase 3 | ⚠️ Partial | 7 JSON examples (some blocked by J1) |
| Phase 4 | ✅ Done | Documentation, CI re-enabled |

### Registered Jetson Modules

| Module | MemType | Description |
|--------|---------|-------------|
| NvArgusCamera | DMABUF | CSI camera via Argus API |
| NvV4L2Camera | DMABUF | USB camera via V4L2 |
| NvTransform | DMABUF | GPU resize/crop/transform |
| JPEGDecoderL4TM | HOST | L4T hardware JPEG decoder |
| JPEGEncoderL4TM | HOST | L4T hardware JPEG encoder |
| EglRenderer | DMABUF | EGL display output |
| DMAFDToHostCopy | DMABUF→HOST | DMA buffer bridge |

### Known Issues

| Issue | Severity | Workaround |
|-------|----------|------------|
| **J1: libjpeg conflict** | High | Use JPEGEncoderNVJPEG |
| **J2: Boost.Serialization** | Medium | Use CLI instead of Node.js |
| **J3: H264EncoderV4L2** | Low | Use H264EncoderNVCodec |

See [JETSON_KNOWN_ISSUES.md](./JETSON_KNOWN_ISSUES.md) for detailed analysis.

### Outcome

Sprint 8 is **functionally complete** with known limitations:
- ✅ Jetson modules registered and building
- ✅ DMABUF bridging implemented
- ✅ CI re-enabled
- ⚠️ L4TM modules blocked by libjpeg conflict
- ⚠️ Node.js addon blocked by linking issue

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
