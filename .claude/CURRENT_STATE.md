# Current State

> Overwrite this file after each significant action.
> Used to resume after /clear or session end.

## Active Task
**MISSION:** Fix ARM64 builds using new Jetson runner (Ubuntu 20.04 / JetPack 5.0)
Branch: `feat/ci-remove-debug`
PR: https://github.com/Apra-Labs/ApraPipes/pull/461

## Current Status: Fix Applied - Build Pending

Build 20439465235 FAILED - Missing transform flag compatibility defines.
Added transform flag mappings and pushing fix.

### What Was Fixed This Commit
1. **Build 20439465235 FAILED** - NVBUFFER_TRANSFORM_FILTER and NVBUFFER_TRANSFORM_CROP_SRC not declared
2. **Root cause:** JetPack 5.x renamed these transform flags to NVBUFSURF_* prefix
3. **Solution:** Added transform flag defines in nvbuf_utils.h:
   - `NVBUFFER_TRANSFORM_FILTER` → `NVBUFSURF_TRANSFORM_FILTER`
   - `NVBUFFER_TRANSFORM_CROP_SRC` → `NVBUFSURF_TRANSFORM_CROP_SRC`
   - `NVBUFFER_TRANSFORM_CROP_DST` → `NVBUFSURF_TRANSFORM_CROP_DST`
   - `NVBUFFER_TRANSFORM_FLIP` → `NVBUFSURF_TRANSFORM_FLIP`

### Previous Issues Fixed (This PR)
1. **CUDA libraries missing** - Installed libnpp-dev-11-4, libcublas-dev-11-4, libcufft-dev-11-4
2. **libpng path mismatch** - Created overlay port with symlink
3. **Static linking -ldl** - Added ${CMAKE_DL_LIBS} to baresip patch
4. **GCC version mismatch** - Made GCC-11 step conditional
5. **Disk space issues** - Set XDG_CACHE_HOME=/data/.cache
6. **PKG_CONFIG_PATH version mismatch** - Prepend vcpkg's pkgconfig path
7. **nvbuf_utils library renamed** - Use NAMES nvbufsurface nvbuf_utils
8. **nvbuf_utils.h header removed** - Created compatibility header with API mappings
9. **NvBufferParams missing layout** - Added layout[NVBUF_MAX_PLANES] to struct
10. **NvEGLImageFromFd not declared** - Added EGL image compatibility functions

## Jetson Status
```
Disk: 9.5G used, 3.5G free (74%)
CUDA: 11.4.315
cuDNN: 8.6.0
NPP: 11.4.0.287
cuBLAS: 11.6.6.84
cuFFT: 10.6.0.202
GCC: 9.4.0
Runner: 2.330.0
JetPack: 5.0 (L4T 35.x)
```

## Commits Made (This PR)
1. **a84e66df3** - XDG_CACHE_HOME, cleanup step, conditional GCC-11, CUDA symlink path
2. **dd417a32c** - Added `${CMAKE_DL_LIBS}` to baresip patch
3. **0cfc1fbb5** - Retry ARM64 with release triplet after cache clear
4. **b47abb488** - libpng overlay port with symlink
5. **638c3ba92** - Prepend vcpkg pkgconfig path for ARM64 builds
6. **1c3687241** - JetPack 5 library compatibility (nvbufsurface, optional nveglstream)
7. **91e97f829** - JetPack 5.x nvbuf_utils.h API compatibility layer
8. **541ab16a6** - Add missing layout field to NvBufferParams struct
9. **81fdfdd01** - Add NvEGLImageFromFd/NvDestroyEGLImage compatibility
10. **(PENDING)** - Add NVBUFFER_TRANSFORM_* flag compatibility

## Build History
| Run ID | Result | Issue |
|--------|--------|-------|
| 20434658500 | FAILED | CUDA libs missing |
| 20435839335 | FAILED | PKG_CONFIG_PATH version mismatch |
| 20438033643 | FAILED | nvbuf_utils library not found (JP5 renamed) |
| 20438402504 | FAILED | nvbuf_utils.h header not found (JP5 API change) |
| 20438835534 | FAILED | NvBufferParams missing 'layout' member |
| 20439019604 | FAILED | NvEGLImageFromFd not declared |
| 20439465235 | FAILED | NVBUFFER_TRANSFORM_FILTER not declared |
| PENDING | IN PROGRESS | Testing transform flag compatibility |

## Next Steps
- [ ] Commit and push transform flag fix
- [ ] Wait for build to complete
- [ ] If passes, re-enable all other workflows
- [ ] Merge PR

---
*Last updated: 2025-12-22 ~18:00 UTC*
