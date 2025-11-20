# Migration & Feature Update Document

**Document Title:** _Migration from `jetpack 4` to `jetpack 6.2`_  
**Project / Module:**  
- V4L2CUYUV420Converter  
- H264EncoderV4L2  
- H264DecoderV4L2  
- MAFDWrapper  
- NvArgusCamera  
- NvV4L2Camera  
- EglRenderer  
- NvEglRenderer  
- DMAUtils  
- NvTransform  
- H264DecoderV4L2Helper  
- MemTypeConversionModule  
- JPEG Encoder  
- JPEG Decoder


**Author:** Chidanand / Rashmi  
**Date:** 10/11/2025  
**Reviewed By:**  
**Version:** —  

---

## 1. Overview
This migration updates multimedia pipeline modules to be compatible with JetPack 6.2, CUDA 12.6, and the newer L4T base.  
The work includes adaptation of memory handling, new rendering features, and major upgrades to EGL, transform, jpeg encoder modules.  
High-level updates include:  
- Migration from JetPack 4 → JetPack 6.2  
- Adaptation to new Jetson encoder/decoder device nodes  
- Implementation of new visual rendering features (text, image, mask)  
- Addition of rotation and flip support in NvTransform  
- Migration to NvBufSurface APIs
- Jpeg encoding and decoding using FD
- Full validation with Boost-based unit testing  

---

## 2. Scope

| In-Scope | Out-of-Scope |
|-----------|---------------|
| Migration of core camera, encoding, and rendering modules | Any non-multimedia peripheral libraries |
| EGL rendering, transform, and memory conversion modules | Application-level UI or network layers |
| Testing of image/text rendering on EGL surfaces | System-wide deployment scripts |

---

## 3. Migration Summary

| Item | Previous State | New State | Notes |
|------|----------------|-----------|-------|
| Dependency Version | CUDA 10.2 | CUDA 12.6 | Required for JetPack 6.2 |
| Build System | Manual Makefiles | CMake with vcpkg | Unified cross-platform build |
| API Interface | `/v1/stream` | `/v2/stream` | Added metadata and mask support |
| Device Node | `/dev/nvhost-encoder` | `/dev/v4l2-nvenc` | New Jetson encoder interface |
| Device Node | `/dev/nvhost-decoder` | `/dev/v4l2-nvdec` | New Jetson decoder interface |
| EGL Renderer | Basic YUV rendering | Supports text, image, and mask rendering | Added FreeType + TTF font support |
| Transform Module | Basic scaling only | Supports rotation and flip | Tested with all orientation combinations |
|Jpeg Encoding| Encoding using host buffer| Encoding using host buffer and Fd|Added fd support enabling zero-copy JPEG encoding|
|Jpeg Decoding| Decoding to host buffer | Decoding to host buffer and fd|Supports decode directly into fd for seamless integration with V4L2/NVMM pipelines|

---

## 4. Detailed Technical Changes

### 4.1 Core Updates
- Refactored for JetPack 6.2 SDK compatibility.  
- Added FreeType-based text rendering on EGL display.
- Introduced image and mask rendering support.
- Implemented window transparency control.
- Enhanced NvTransform with rotation and flip features.
- Added FD-based JPEG encoding for zero-copy operation.

- Added FD-based JPEG decoding for seamless V4L2/NVMM pipeline integration.
- Migrated to NvBufSurface APIs with automatic plane mapping, NVMM support, and CUDA/EGL zero-copy interop



### 4.2 Dependency Changes

| Package/Library | Old Version | New Version | Reason |
|-----------------|--------------|--------------|---------|
| CUDA | 10.2 | 12.6 | JetPack 6.2 support |
| EGL / OpenGL ES | JetPack 4.x | JetPack 6.2 runtime | Updated EGL API usage |
| FreeType | — | 2.13.2 | Added for text rendering |
| Jpeglib | jpeglib-turbo 7,8|  libnvjpeg.so | hardware accelerated jpeg encoding, decoding |




### 4.3 Configuration Updates
- EglRendererProps:
  - Added constructors to define renderer properties including:
  - Geometry: x/y offsets, width, height.
  - Text rendering: font path, message, scale, font size, color, position,opacity.
  - Image rendering: path, position, size,opacity.
  - Opacity & mask settings.
  - Supports combinations of geometry, text, image, opacity, and mask for flexible EGL renderer configuration.
- NvTransformProps:
  - Added constructors to define image transform properties including:
  - Cropping: width, height, top, left.
  - Rotation: 0°, 90°, 180°, 270°.
  - Flip: Horizontal, Vertical.
  - Supports combinations of crop + rotation, crop + flip, or default initialization.
  - Enables easy configuration of frame transformations for new JetPack 6.2 pipelines.
- JPEGDecoderL4TMProps:
  -  Added a configurable property that allows selecting the desired JPEG decoding method. (Decoding to FD/ Host buffer)
### 4.4 Database / Storage Changes
| Table/Collection | Change Description | Migration Query/Command |
|------------------|-------------------|-------------------------|
| — | No database changes | — |

---

## 5. New Features Added

| Feature Name | Description | User/Client Impact | Related Ticket/PR |
|---------------|-------------|-------------------|------------------|
| Image Rendering | Ability to render images on EGL display | Visual enhancement for overlay | #429 |
| Mask Rendering | Display alpha masks on frame | Enables blending/compositing | #429|
| Text Rendering | FreeType-based text display on EGL | Enables debugging, HUD overlays | #429 |
| Transparency Control | Dynamic window opacity control | Smooth UI blending | #429 |
| Rotation & Flip | Transform rotation (0°, 90°, 180°, 270°) and flip (H/V) | Flexible rendering pipeline | #429 |
| Jpeg Encoding enhancement | encoding from FD for zero copy operations|Improved performance, reduced CPU usage|
| Jpeg Decoding enhancement | Ability to decode JPEG images directly into a DMA-BUF (Fd) surface | Enables end-to-end zero-copy pipelines, better integration with V4L2/NVMM pipelines |
---

## 6. Modules Updated

- **H264 Encoder**: Adapted to `/dev/v4l2-nvenc` device node.  
- **H264 Decoder**:  Adapted to `/dev/v4l2-nvdec` device node.  
- **V4L2CUYUV420Converter**: Added proper YUV420 conversion for new api.  
- **NvTransform**: Added rotation/flip support with unit tests.  
- **NvEglRenderer**: Added image/mask/text rendering + transparency.  
- **MemTypeConversionModule**: Updated for new DMA buffer semantics.  
- **DMAFDWrapper, DMAUtils**: Migrated to NvBufSurface APIs with automatic plane mapping,   NVMM support, and CUDA/EGL zero-copy interop.
- **Jpeg Encoder**: Added jpeg encoding using FD for nv12, yuv formats
- **Jpeg Decoder**: Added jpeg decoding to FD for yuv formats



---

## 7. Testing & Validation

| Test Case | Expected Result | Status | Owner |
|------------|----------------|---------|-------|
| Image render on EGL display | Image overlays correctly |Passed|Chidanand|
| Mask rendering | Proper alpha blending | Passed|Chidanand|
| Text rendering | Font displayed with correct glyph mapping | Passed|Chidanand|
| Rotation tests (0°, 90°, 180°, 270°) | Frame correctly rotated | Passed|Chidanand|
| Flip tests (H/V/Both) | Proper mirroring of frame |Passed|Chidanand|
| Transparency setting | Window opacity update |Passed|Chidanand|
| Cuda Pointer for valid FD | Successfully maps FD to CUDA pointer | Passed | Rashmi|
| Cuda Pointer for valid FD across different formats | CUDA pointer obtained, resources freed correctly | Passed |Rashmi|
| Encoding using FD -yuv| Proper jpeg encoding| Passed |Rashmi|
| Encoding using FD -nv12| Proper jpeg encoding | Passed |Rashmi|
|exception handling-jpeg encoder| Throws necessary exceptions | Passed | Rashmi|
| Encoding of continuos argus camera frames | Proper jpeg encoding | Passed | Rashmi|

---

## 8. Known Issues / Risks

- FreeType initialization requires TTF font presence in `/data/` path.
- opencv version changed need to check all the modules behaviour  

---
