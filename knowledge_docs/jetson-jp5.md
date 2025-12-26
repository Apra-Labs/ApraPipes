# Jetson JetPack 5.x Technical Reference

## EGL Display Architecture

### Headless GPU Access
`EGL_DEFAULT_DISPLAY` with Xvfb returns software-rendered EGL, not NVIDIA GPU.

For NvBufSurface/CUDA-EGL interop, try `EGL_PLATFORM_DEVICE_EXT` FIRST:
```cpp
PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
    (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
    (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
if (eglQueryDevicesEXT && eglGetPlatformDisplayEXT) {
    EGLDeviceEXT devices[8];
    EGLint numDevices;
    if (eglQueryDevicesEXT(8, devices, &numDevices) && numDevices > 0) {
        mEGLDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[0], NULL);
        if (mEGLDisplay != EGL_NO_DISPLAY && eglInitialize(mEGLDisplay, NULL, NULL))
            return; // GPU-backed display
    }
}
// Fallback to default (works with real display attached)
mEGLDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
```

### DMA Capability Detection
- `isAvailable()` = EGL display initialized (may be software)
- `isDMACapable()` = eglImage from NvBuffer works (requires GPU)

Use `isDMACapable()` to guard DMA/NvBufSurface tests:
```cpp
#define SKIP_IF_NO_DMA_CAPABLE() \
    if (!ApraEGLDisplay::isDMACapable()) { \
        LOG_WARNING << "Skipping - DMA not available"; \
        return; \
    }
```

---

## NvBuffer to NvBufSurface API Migration

JetPack 5.x replaced legacy NvBuffer API with NvBufSurface API.

### Key Differences
- V4L2 mmap buffers have per-plane FDs, not single FD for all planes
- `NvBufSurfaceFromFd()` only works with NvBufSurface-created buffers, NOT V4L2 mmap FDs
- V4L2 driver handles cache coherency for mmap'd buffers

### V4L2 mmap Compatibility
```cpp
int ret = NvBufSurfaceFromFd(fd, (void**)&surface);
if (ret != 0) {
    // Buffer is V4L2 mmap, not NvBufSurface - expected
    // V4L2 driver handles coherency
    return 0;
}
```

---

## Motion Vector IOCTL Validation

V4L2 IOCTL can succeed but return garbage pointer for `pMVInfo`.

Validate both size and pointer address range:
```cpp
uint32_t expectedMaxMacroblocks = ((mWidth + 15) / 16) * ((mHeight + 15) / 16);
uint32_t expectedSize = expectedMaxMacroblocks * sizeof(MVInfo);
if (enc_mv_metadata.bufSize > expectedSize * 2 ||
    enc_mv_metadata.bufSize < sizeof(MVInfo)) return;

uintptr_t ptrVal = reinterpret_cast<uintptr_t>(enc_mv_metadata.pMVInfo);
if (ptrVal < 0x10000 || ptrVal > 0x0000FFFFFFFFFFFFULL) return;
```

---

## Library Migration

| JetPack 4.x | JetPack 5.x | Notes |
|-------------|-------------|-------|
| `libnvbuf_utils.so` | `libnvbufsurface.so` | API changed |
| `nvbuf_utils.h` | `nvbufsurface.h` | Different functions |
| `nveglstream_camconsumer` | Removed | Use NvBufSurface directly |

CMake pattern to accept both:
```cmake
find_library(NVBUFUTILSLIB NAMES nvbufsurface nvbuf_utils REQUIRED)
```
