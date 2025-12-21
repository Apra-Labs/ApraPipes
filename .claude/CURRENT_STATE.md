# Current State

> Overwrite this file after each significant action.
> Used to resume after /clear or session end.

## Active Task
**MISSION:** Fix ARM64 builds using new Jetson runner (Ubuntu 20.04)
Branch: `feat/ci-remove-debug`
PR: https://github.com/Apra-Labs/ApraPipes/pull/461

## What Just Changed
1. **Disabled 7 workflows** (renamed to `.yml.disabled`)
2. **Updated all reusable workflows to use `fromJson(inputs.runner)`** for flexible runner labels
3. **Updated all caller workflows to pass JSON arrays** for runner labels
4. **Updated CI-Linux-ARM64.yml** for new Jetson runner:
   - Runner: `["self-hosted", "Linux", "ARM64", "Ubuntu20.04"]`
   - Cache path: `/data/.cache/vcpkg`
   - CUDA path: `/usr/local/cuda` (symlink works with 11.4)
5. **Installed on new Jetson:**
   - PowerShell 7.4.6
   - autoconf 2.71
   - libcudnn8-dev 8.6.0
   - libtool-bin

## New Jetson Environment (ap-gh-arm64-jp5)
```
Ubuntu 20.04.6 LTS (GLIBC 2.31)
git 2.50.1, cmake 3.29.3, ninja 1.10.0
gcc/g++ 9.4.0, autoconf 2.71, automake 1.16.1
CUDA 11.4, cuDNN 8.6.0
PowerShell 7.4.6
Runner 2.330.0 online
/data: 107GB free
```

## Next Steps
- [ ] Commit and push changes
- [ ] Watch ARM64 workflow run
- [ ] Fix any build issues that arise
- [ ] Re-enable all workflows once ARM64 passes

---
*Last updated: 2025-12-21 17:25 UTC*
