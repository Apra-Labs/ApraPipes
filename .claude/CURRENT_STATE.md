# Current State

> Overwrite this file after each significant action.
> Used to resume after /clear or session end.

## Active Task
**MISSION:** Fix ARM64 builds using new Jetson runner (Ubuntu 20.04)
Branch: `feat/ci-remove-debug`
PR: https://github.com/Apra-Labs/ApraPipes/pull/461

## What Just Changed
1. **Fixed disk space issue on Jetson** - Root filesystem (/) was 100% full (34MB free)
   - Root cause: vcpkg registries cache was going to `~/.cache/vcpkg` on root filesystem
   - Solution: Added `XDG_CACHE_HOME=/data/.cache` for self-hosted runners
   - This redirects ALL caches (including vcpkg registries) to `/data` where there's 106GB free
2. **Added cleanup step** for self-hosted runners:
   - Removes old vcpkg registries from `~/.cache/vcpkg/registries`
   - Cleans apt cache
   - Removes pip cache
3. **Ensured XDG cache directory is created** alongside vcpkg binary cache

## New Jetson Environment (ap-gh-arm64-jp5)
```
Ubuntu 20.04.6 LTS (GLIBC 2.31)
git 2.50.1, cmake 3.29.3, ninja 1.10.0
gcc/g++ 9.4.0, autoconf 2.71, automake 1.16.1
CUDA 11.4, cuDNN 8.6.0
PowerShell 7.4.6
Runner 2.330.0 online
/data: 106GB free (root / has only 34MB!)
```

## Key Issue Found
The Jetson has two filesystems:
- `/` (root): 14GB, 100% used, only 34MB free
- `/data` (NVMe): 117GB, only 5% used, 106GB free

vcpkg was trying to write registries to `~/.cache/vcpkg/registries` on root filesystem.
Fix: Set `XDG_CACHE_HOME=/data/.cache` to redirect all caches to NVMe.

## Next Steps
- [ ] Commit and push XDG_CACHE_HOME fix
- [ ] Watch ARM64 workflow run
- [ ] Fix any build issues that arise
- [ ] Re-enable all workflows once ARM64 passes

---
*Last updated: 2025-12-21 19:50 UTC*
