# Current State

> Overwrite this file after each significant action.
> Used to resume after /clear or session end.

## Active Task
**MISSION:** Fix ARM64 builds using new Jetson runner (Ubuntu 20.04 / JetPack 5.0)
Branch: `feat/ci-remove-debug`
PR: https://github.com/Apra-Labs/ApraPipes/pull/461

## Current Status: Adding Xvfb for Headless Tests

**BUILD SUCCEEDED!** (Run 20439893264)
**LDCONFIG FIX WORKED!** (Run 20440706046)
**TEST HANG IDENTIFIED:** Tests hang because GTK/EGL tests need a display

**NEW FIX APPLIED:**
- Installed Xvfb on Jetson for virtual display
- Modified build-test-lin.yml to use xvfb-run when available
- This provides a headless display for GTK/EGL tests

### Root Cause of Hang
Tests were hanging for 45+ minutes because:
1. `gtkglrenderer_tests.cpp`, `eglrenderer_test.cpp`, `apraegldisplay_tests.cpp` need a display
2. Jetson is headless (no physical display attached)
3. These tests block waiting for display indefinitely
4. No test output was produced (XML file empty)

### JetPack 5.x Compatibility Header Complete
Created `base/include/nvbuf_utils.h` with full API mapping:
- NvBufferParams struct with all fields including layout
- NvBufferCreateParams with all JetPack 5.x fields
- Transform flags: NVBUFFER_TRANSFORM_* -> NVBUFSURF_TRANSFORM_*
- EGL image functions: NvEGLImageFromFd, NvDestroyEGLImage
- Buffer functions: NvBufSurf* wrappers for legacy NvBuffer* calls
- NvBufSurfaceManager class for FD-to-surface mapping

### Issues Fixed (This PR)
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
11. **NVBUFFER_TRANSFORM_* flags** - Added transform flag defines
12. **NPP runtime libs not found** - Added CUDA targets to ldconfig
13. **Test timeout** - Increased nTestTimeoutMins from 20 to 45
14. **Headless display hang** - Using xvfb-run for virtual display

## Jetson Status
```
Disk: 9.5G used, 3.5G free (74%)
CUDA: 11.4.315
cuDNN: 8.6.0
NPP: 11.4.0.287 (in ldconfig)
cuBLAS: 11.6.6.84
cuFFT: 10.6.0.202
GCC: 9.4.0
Runner: 2.330.0
JetPack: 5.0 (L4T 35.x)
ldconfig: /usr/local/cuda/targets/aarch64-linux/lib added
Xvfb: installed (2:1.20.13)
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
10. **7463357cd** - Add NVBUFFER_TRANSFORM_* flag compatibility
11. **ea24f5d17** - Increase test timeout from 20 to 45 mins
12. **(pending)** - Use xvfb-run for headless display in tests

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
| 20439893264 | BUILD OK | Tests failed - NPP libs not in ldconfig |
| 20440706046 | HANG | ldd OK, tests hung (no display) at 20 mins |
| 20441936650 | HANG | Tests hung (no display) at 45 mins |
| (next) | PENDING | Testing with Xvfb virtual display |

## Next Steps
- [x] Increase test timeout to 45 mins
- [x] Identify test hang (headless display issue)
- [x] Install Xvfb on Jetson
- [x] Modify workflow to use xvfb-run
- [ ] Commit and push Xvfb changes
- [ ] Verify tests complete with Xvfb
- [ ] Re-enable all other workflows
- [ ] Merge PR

---
*Last updated: 2025-12-22 ~21:10 UTC*
