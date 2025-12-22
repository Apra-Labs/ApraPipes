# Current State

> Overwrite this file after each significant action.
> Used to resume after /clear or session end.

## Active Task
**MISSION:** Fix ARM64 builds using new Jetson runner (Ubuntu 20.04)
Branch: `feat/ci-remove-debug`
PR: https://github.com/Apra-Labs/ApraPipes/pull/461

## Current Fix: Use Standard Triplet for ARM64

The `arm64-linux-release` triplet failed because opencv4 couldn't find dependencies (png.h missing).
Release-only triplets have dependency resolution issues on ARM64.

**Solution:** Use standard `arm64-linux` triplet instead of `arm64-linux-release` for ARM64 builds.

### Changes Made This Session
1. Changed `vcpkg-triplet` from `arm64-linux-release` to `arm64-linux` in CI-Linux-ARM64.yml
2. Updated baresip patch to use `find_library()` to detect either .a or .so

## Commits Made
1. **a84e66df3** - XDG_CACHE_HOME, cleanup step, conditional GCC-11, CUDA symlink path
2. **dd417a32c** - Added `${CMAKE_DL_LIBS}` to baresip patch (patch format fixed)
3. **Pending** - Switch ARM64 to standard triplet

## Jetson Status
```
Disk: 6.6G used, 6.5G free (51%)
CUDA: 11.4.315
cuDNN: 8.6.0
GCC: 9.4.0
Runner: 2.330.0
```

## Next Steps
- [x] Switch ARM64 to standard `arm64-linux` triplet
- [ ] Commit and trigger build
- [ ] Re-enable all workflows
- [ ] Merge PR

---
*Last updated: 2025-12-22 07:00 UTC*
