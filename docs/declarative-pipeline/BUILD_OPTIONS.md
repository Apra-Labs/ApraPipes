# ApraPipes Build Options

This document describes build options for ApraPipes, focusing on FFmpeg assembly optimizations and Node.js addon compatibility.

## FFmpeg Assembly Optimizations

### Background

FFmpeg includes hand-written assembly optimizations for various architectures:
- **x86/x64**: MMX, SSE, AVX instructions
- **ARM64**: NEON SIMD instructions

These optimizations improve performance for FFmpeg operations but create a conflict with static linking: the assembly code uses PC-relative addressing that's incompatible with Position Independent Code (PIC) required for shared objects like Node.js native addons (`.node` files).

### The `USE_FFMPEG_ASM_OPTIMIZATIONS` Option

```cmake
option(USE_FFMPEG_ASM_OPTIMIZATIONS
       "Enable FFmpeg hand-written assembly optimizations"
       OFF)
```

#### Default: OFF (Recommended for most users)

When `USE_FFMPEG_ASM_OPTIMIZATIONS=OFF` (default):
- FFmpeg is built with `--disable-asm --disable-x86asm`
- Pure C implementations are used (slightly slower)
- Static FFmpeg can be linked into shared objects
- **Node.js addon includes ALL modules** including FFmpeg-dependent ones

#### When ON (For performance-critical deployments)

When `USE_FFMPEG_ASM_OPTIMIZATIONS=ON`:
- FFmpeg is built with assembly optimizations enabled
- Better performance for FFmpeg operations
- **Node.js addon EXCLUDES FFmpeg-dependent modules**:
  - `RTSPClientSrc` - RTSP streaming input
  - `RTSPPusher` - RTSP streaming output
  - `MotionVectorExtractor` - H.264 motion vector extraction
  - `H264ParserUtils` - H.264 SPS/PPS parsing

### How to Enable FFmpeg Assembly

To enable FFmpeg assembly optimizations:

1. **Update `base/vcpkg.json`** to request the `asm` feature:
   ```json
   {
     "dependencies": [
       {
         "name": "ffmpeg",
         "features": ["asm"]
       }
     ]
   }
   ```

2. **Set the CMake flag**:
   ```bash
   cmake -DUSE_FFMPEG_ASM_OPTIMIZATIONS=ON ...
   ```

3. **Rebuild vcpkg packages** (the triplet change invalidates cache):
   ```bash
   rm -rf vcpkg_installed/
   cmake --build . --target clean
   cmake -B build ...
   ```

### FFmpeg Usage in ApraPipes

FFmpeg is only used by 4 modules in ApraPipes:

| Module | Purpose | FFmpeg Libraries |
|--------|---------|------------------|
| `RTSPClientSrc` | RTSP stream demuxing (input) | avformat, avdevice |
| `RTSPPusher` | RTSP stream muxing (output) | avformat, avcodec |
| `MotionVectorExtractor` | H.264 motion vector extraction | avcodec, swscale |
| `H264ParserUtils` | SPS/PPS parsing for video dimensions | avcodec, avformat |

**Note**: The main video pipeline (file reading, NVIDIA encoding/decoding, image processing, ML inference) does NOT use FFmpeg.

### Performance Impact

The performance difference between FFmpeg with and without assembly depends on your workload:

| Scenario | Impact |
|----------|--------|
| RTSP streaming (RTSPClientSrc/RTSPPusher) | Moderate - affects demux/mux performance |
| Motion vector extraction (MotionVectorExtractor) | Significant - involves H.264 decoding |
| Most ApraPipes workloads (NVIDIA codecs, OpenCV) | **None** - don't use FFmpeg |

For typical ApraPipes usage with NVIDIA hardware acceleration, disabling FFmpeg assembly has minimal impact since the main video codec work is done by NVENC/NVDEC.

### Platform-Specific Notes

#### x86/x64 Linux
- Assembly: MMX, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2
- Without asm: Uses generic C code with compiler auto-vectorization

#### ARM64 Linux (Jetson, etc.)
- Assembly: ARM NEON SIMD
- Without asm: Uses generic C code
- Note: NVIDIA Jetson platforms use NVENC/NVDEC for most video work anyway

### Node.js Addon Implications

| Configuration | Node.js Addon Contents |
|---------------|------------------------|
| `USE_FFMPEG_ASM_OPTIMIZATIONS=OFF` | All modules (full functionality) |
| `USE_FFMPEG_ASM_OPTIMIZATIONS=ON` | All modules EXCEPT FFmpeg-dependent ones |

When asm is enabled, attempting to use FFmpeg-dependent modules from Node.js will fail since they're not included in the addon.

### Build Scripts

The local build scripts have been updated to use the default (asm OFF):

- `build_linux_cuda.sh` - Uses `USE_FFMPEG_ASM_OPTIMIZATIONS=OFF` (default)
- CI workflows - Use same default for consistency

### Troubleshooting

#### Link Error: "relocation R_X86_64_PC32 against symbol"
```
/usr/bin/ld: libavcodec.a(vc1dsp_mmx.o): relocation R_X86_64_PC32 against symbol `ff_pw_9'
can not be used when making a shared object; recompile with -fPIC
```

This error occurs when:
1. FFmpeg was built WITH assembly (asm feature enabled)
2. You're trying to link FFmpeg statically into a shared object (Node.js addon)

**Solution**: Ensure `USE_FFMPEG_ASM_OPTIMIZATIONS=OFF` (default) or rebuild FFmpeg without the `asm` feature.

#### MotionVectorExtractor Not Available in Node.js Addon

If using `USE_FFMPEG_ASM_OPTIMIZATIONS=ON`, the `MotionVectorExtractor` module won't be available in the Node.js addon. Options:
1. Use `OPENH264` backend instead of `FFMPEG` for motion vector extraction
2. Disable FFmpeg asm optimizations to include all modules

### Related Files

- `base/CMakeLists.txt` - `USE_FFMPEG_ASM_OPTIMIZATIONS` option definition
- `thirdparty/custom-overlay/ffmpeg/vcpkg.json` - FFmpeg `asm` feature
- `thirdparty/custom-overlay/ffmpeg/portfile.cmake` - Assembly configuration logic
- `vcpkg/triplets/community/x64-linux-cuda.cmake` - x64 Linux triplet
- `vcpkg/triplets/community/arm64-linux-release.cmake` - ARM64 Linux triplet
