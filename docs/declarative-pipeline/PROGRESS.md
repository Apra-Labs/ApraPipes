# Declarative Pipeline - Progress Tracker

> Last Updated: 2026-01-16

**Branch:** `feat-declarative-pipeline-v2`

---

## Current Status

| Component | Status |
|-----------|--------|
| Core Infrastructure | ✅ Complete (Metadata, Registry, Factory, Validator, CLI) |
| JSON Parser | ✅ Complete (TOML removed) |
| Cross-platform Modules | ✅ 37 modules |
| CUDA Modules | ✅ 15 modules (NPP + NVCodec) |
| Jetson Modules | ✅ 8 modules (L4TM working via dlopen wrapper) |
| Node.js Addon | ✅ Complete (Jetson blocked by linking - J2) |
| Auto-Bridging | ✅ Complete (memory + pixel format) |

**Module count:** Run `./build/aprapipes_cli list-modules` for current count per platform.

---

## Sprint 8: Jetson Integration (Complete)

> Started: 2026-01-13 | Completed: 2026-01-16

| Phase | Status | Description |
|-------|--------|-------------|
| Prerequisites | ✅ Complete | CI disabled, SSH, workspace |
| Phase 1 | ✅ Complete | Jetson build |
| Phase 2 | ✅ Complete | Register 8 Jetson modules |
| Phase 2.5 | ✅ Complete | DMABUF auto-bridging |
| Phase 3 | ✅ Complete | JSON examples (L4TM working via dlopen) |
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

| Issue | Status | Notes |
|-------|--------|-------|
| J1: libjpeg conflict | ✅ RESOLVED | dlopen wrapper isolates symbols |
| J2: Node.js linking | ⚠️ Open | Use CLI as workaround |
| J3: H264EncoderV4L2 | ⚠️ Open | Use H264EncoderNVCodec |

**J1 Resolution:** L4TMJpegLoader uses `dlopen()` with `RTLD_LOCAL` to isolate NVIDIA's libjpeg symbols. 7 tests passing in CI.

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
- 01_test_signal_to_jpeg.json - L4TM JPEG encode/decode
- 01_jpeg_decode_transform.json - L4TM decode + resize
- 03_camera_preview.json - CSI camera → display
- 04_usb_camera_jpeg.json - USB camera → JPEG

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

### Priority 1: Fix Node.js Addon on Jetson (J2)
- Potential: Extend `--whole-archive` to include Boost libraries
- Workaround: Use CLI

### Priority 2: Display Modules
- Register GtkGlRenderer, ImageViewerModule
- Note: EglRenderer is already registered for Jetson
- Low priority - mainly for debugging

### Priority 3: H264EncoderV4L2 on ARM64 (J3)
- Add ARM64 guard to registration
- Low priority - H264EncoderNVCodec works

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
