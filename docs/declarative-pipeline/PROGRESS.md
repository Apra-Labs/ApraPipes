# Declarative Pipeline - Progress Tracker

> Last Updated: 2026-01-15

**Branch:** `feat-declarative-pipeline-v2`

---

## Current Status

| Component | Status |
|-----------|--------|
| Core Infrastructure | ✅ Complete (Metadata, Registry, Factory, Validator, CLI) |
| JSON Parser | ✅ Complete (TOML removed) |
| Cross-platform Modules | ✅ 37 modules |
| CUDA Modules | ✅ 15 modules (NPP + NVCodec) |
| Jetson Modules | ⚠️ 8 modules (L4TM blocked by libjpeg) |
| Node.js Addon | ✅ Complete (Jetson blocked by linking) |
| Auto-Bridging | ✅ Complete (memory + pixel format) |

**Module count:** Run `./build/aprapipes_cli list-modules` for current count per platform.

---

## Sprint 8: Jetson Integration (Current)

> Started: 2026-01-13

| Phase | Status | Description |
|-------|--------|-------------|
| Prerequisites | ✅ Complete | CI disabled, SSH, workspace |
| Phase 1 | ✅ Complete | Jetson build |
| Phase 2 | ✅ Complete | Register 8 Jetson modules |
| Phase 2.5 | ✅ Complete | DMABUF auto-bridging |
| Phase 3 | ⚠️ Partial | JSON examples (J1 blocks L4TM) |
| Phase 4 | ✅ Complete | Docs, CI re-enabled |

### Jetson Modules Registered

| Module | MemType | Description |
|--------|---------|-------------|
| NvArgusCamera | DMABUF | CSI camera via Argus |
| NvV4L2Camera | DMABUF | USB camera via V4L2 |
| NvTransform | DMABUF | GPU resize/crop |
| JPEGDecoderL4TM | HOST | L4T JPEG decoder |
| JPEGEncoderL4TM | HOST | L4T JPEG encoder |
| EglRenderer | DMABUF | EGL display |
| DMAFDToHostCopy | DMABUF→HOST | DMA bridge |

### Known Issues

> **Details:** [JETSON_KNOWN_ISSUES.md](./JETSON_KNOWN_ISSUES.md)

| Issue | Severity | Workaround |
|-------|----------|------------|
| J1: libjpeg conflict | High | Use JPEGEncoderNVJPEG |
| J2: Node.js linking | Medium | Use CLI |
| J3: H264EncoderV4L2 | Low | Use H264EncoderNVCodec |

**J1 Root Cause:** L4T Multimedia API links system libjpeg-62, vcpkg provides libjpeg-turbo-80. Struct size mismatch causes crashes.

**J2 Root Cause:** Boost.Serialization RTTI symbols not exported with `--whole-archive`. GCC 9.4 on Jetson has stricter symbol resolution.

---

## Build Status

| Platform | Build | Node Addon | Notes |
|----------|-------|------------|-------|
| macOS | ✅ | ✅ | 37 modules |
| Windows | ✅ | ✅ | 37 modules |
| Linux x64 | ✅ | ✅ | 39 modules (+V4L2) |
| Linux x64 CUDA | ✅ | ✅ | 52 modules (+15 CUDA) |
| Jetson ARM64 | ✅ | ❌ | 45+ modules (J2 blocks addon) |

---

## Completed Sprints

| Sprint | Theme | Key Deliverables |
|--------|-------|------------------|
| 7 | Auto-Bridging | PipelineAnalyzer, auto-insert CudaMemCopy/ColorConversion |
| 6 | DRY Refactoring | Fix defaults, type validation |
| 5 | CUDA | 15 modules, shared cudastream_sp |
| 4 | Node.js | @apralabs/aprapipes, event system |
| 1-3 | Core | Registry, Factory, Validator, CLI, 37 modules |

---

## Examples

### Node.js (`examples/node/`)
- basic_pipeline.js - Test frames → JPEG → files
- ptz_control.js - VirtualPTZ with dynamic props
- event_handling.js - Health/error callbacks

### CUDA (`examples/cuda/`)
- gaussian_blur.json - GPU blur with explicit bridges
- auto_bridge.json - Auto-bridging demo
- effects.json - NPP brightness/contrast/saturation
- processing_chain.json - Multi-stage GPU pipeline

### Jetson (`examples/jetson/`)
- 03_camera_preview.json - CSI camera → display
- 04_usb_camera_jpeg.json - USB camera → JPEG
- ⚠️ L4TM examples blocked by J1

---

## Documentation

| Document | Purpose |
|----------|---------|
| [PROJECT_PLAN.md](./PROJECT_PLAN.md) | Sprint overview |
| [JETSON_KNOWN_ISSUES.md](./JETSON_KNOWN_ISSUES.md) | Jetson platform issues |
| [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) | Module registration |
| [PIPELINE_AUTHOR_GUIDE.md](./PIPELINE_AUTHOR_GUIDE.md) | JSON authoring |
| [CUDA_MEMTYPE_DESIGN.md](./CUDA_MEMTYPE_DESIGN.md) | Auto-bridging design |

---

## Future Work

### Priority 1: Fix libjpeg Conflict (J1)
- Options: shared libjpeg-turbo, LD_PRELOAD, process isolation
- See [JETSON_KNOWN_ISSUES.md](./JETSON_KNOWN_ISSUES.md#issue-j1-libjpeg-version-conflict-l4tm-modules)

### Priority 2: Fix Node.js Addon on Jetson (J2)
- Potential: Extend `--whole-archive` to include Boost libraries
- Workaround: Use CLI

### Priority 3: Display Modules
- Register GtkGlRenderer, EglRenderer, ImageViewerModule
- Low priority - mainly for debugging

---

## Build Troubleshooting

### Ubuntu 24.04 + CUDA 11.8

CUDA 11.8 requires GCC-11 (not GCC-13 default):

```bash
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export CUDA_PATH=/usr/local/cuda
```

### FFmpeg on Ubuntu 24.04

Use custom overlay for binutils 2.41+ compatibility:
```bash
cmake -B build -DVCPKG_OVERLAY_PORTS=thirdparty/custom-overlay ...
```

CI uses Ubuntu 22.04 which doesn't need this.
