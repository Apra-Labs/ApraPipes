# Unified Linux CUDA Build - Attempt Log

**Workflow:** `CI-Linux-CUDA-Unified.yml`
**Strategy:** Incremental caching with `LINUX-TEMP` fixed key
**Goal:** Get CMake configure to succeed, building up vcpkg cache with each attempt

---

## Attempt Template

### Attempt N: [Brief Description]
**Run ID:**
**Started:**
**Result:** ⏳/✅/❌
**CMake Result:**
**Error:**
**Fix Applied:**
**Lessons:**

---

## Strategy Notes

### Incremental Caching Pattern (from Mac workflow)
```yaml
# BEFORE CMake configure
- name: Restore vcpkg cache
  uses: actions/cache/restore@v3
  with:
    key: LINUX-TEMP

# CMake configure step (may fail)
- name: Configure CMake
  continue-on-error: true  # Don't fail workflow if CMake fails

# AFTER CMake configure (ALWAYS runs, even on failure)
- name: Save vcpkg cache
  if: ${{ always() }}  # CRITICAL: Always save to build up cache
  uses: actions/cache/save@v3
  with:
    key: LINUX-TEMP
```

### Why This Works
1. First attempt: No cache, CMake fails partway, saves partial cache
2. Second attempt: Restores partial cache, gets farther, saves more progress
3. Each attempt builds on previous, eventually complete cache is built
4. Once cache is complete, CMake succeeds

### Key Differences from Standard Caching
- **Standard:** Hash-based keys, cache only on success
- **Incremental:** Fixed key (`LINUX-TEMP`), always save even on failure
- **Benefit:** Failed builds still contribute to cache, reducing total attempts

---

## Checklist for Each Attempt

Before triggering:
- [ ] Read CMake error from previous attempt logs
- [ ] Identify root cause (missing package, wrong path, etc.)
- [ ] Apply targeted fix to workflow
- [ ] Document fix in this file
- [ ] Commit changes with descriptive message

After completion:
- [ ] Download and analyze logs if failed
- [ ] Update this document with attempt results
- [ ] Commit documentation update

---

## Common CMake Errors to Watch For

From past experience (per user):
1. **Missing system packages** → Add to prep-cmd apt install
2. **vcpkg port not found** → Check vcpkg.json, baseline.json
3. **Compiler version mismatch** → Verify GCC-11 for CUDA 11.8
4. **CUDA not detected** → Check env vars (CUDAToolkit_ROOT, CUDA_PATH, CUDACXX)
5. **Path issues** → Verify /usr/local/cuda-11.8/bin in PATH
6. **Permission errors** → Check workspace permissions
7. **Disk space** → Monitor df -h in logs (use /mnt if needed)
8. **Dependency conflicts** → Check vcpkg version constraints

---

## Attempts Log

(Attempts will be logged below in reverse chronological order)

---

### Attempt 5: LFS not enabled - stub libraries are text pointers
**Run ID:** 20238170778
**Started:** 2025-12-15 15:42 UTC
**Result:** ❌ FAILED in 4h 2m 19s
**CMake Result:** ✅ **SUCCEEDED** (after ~3.5 hours - Major milestone!)
**Build Result:** ✅ All 161/161 source files compiled successfully
**Linker Error:** `/usr/bin/ld:/home/runner/work/ApraPipes/ApraPipes/thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64/libnvcuvid.so: file format not recognized`
**Root Cause:** Checkout step had `lfs: false`, so Video_Codec_SDK stub libraries are git LFS text pointers, not actual binary files
**Fix Applied:** Added `lfs: true` to checkout step (following Docker/WSL/Windows workflow pattern)
**Lessons:**
- User correctly identified: "that format error must be due to lfs"
- Working workflows (Docker, WSL, Windows) all use `lfs: true` in checkout
- CMake configure now succeeds reliably thanks to vcpkg cache from Attempt 2
- All compilation passes - only linker needs LFS binaries
- Cache save failed with "cache entry with same key already exists" (LINUX-TEMP from Attempt 2)
- Build is VERY close to success - just needed proper LFS checkout

**Milestone Progress:**
- ✅ CMake configure (3.5h on first attempt, now uses cache)
- ✅ All compilation (161/161 files)
- ❌ Linking (needs LFS binaries)

---

### Attempt 4: libcudnn8-dev not found - missing CUDA repos
**Run ID:** 20238101404
**Started:** 2025-12-15 15:40 UTC
**Result:** ❌ FAILED in 59s
**CMake Result:** Never reached (prep-cmd failed)
**Error:** `E: Unable to locate package libcudnn8-dev`
**Root Cause:** Tried to install libcudnn8-dev BEFORE installing CUDA toolkit (which adds CUDA repos to apt)
**Fix Applied:** Reordered steps - Install CUDA toolkit FIRST, THEN install cuDNN, THEN prepare builder
**Lessons:**
- Docker workflow works because nvidia/cuda container already has CUDA repos configured
- GitHub-hosted runners need CUDA repos added first (via cuda-toolkit install)
- Order matters: 1) Install CUDA (adds repos), 2) Install cuDNN, 3) Install other deps

---

### Attempt 3 (CANCELED): Wrong fix - removed cudnn instead of installing libcudnn8-dev
**Run ID:** 20237814491
**Started:** 2025-12-15 15:31 UTC
**Result:** ❌ CANCELED after ~10 minutes
**CMake Result:** N/A (canceled before completion)
**Error:** N/A (wrong fix applied)
**Root Cause:** Removed cudnn from vcpkg.json instead of installing libcudnn8-dev like Docker workflow
**Fix Applied:**
- Restored cudnn to opencv4 features in vcpkg.json
- Added libcudnn8-dev to apt-get install in workflow (following Docker workflow pattern)
**Lessons:**
- Should have checked Docker workflow FIRST (user pointed this out)
- Docker workflow solves this with `apt-get install -y libcudnn8-dev` (line 105 in build-test-lin-container.yml)
- User had to remind me the issue was already solved - shameful

---

### Attempt 2: cudnn build failure
**Run ID:** 20233713118
**Started:** 2025-12-15 13:21 UTC
**Result:** ❌ FAILED in 45m59s
**CMake Result:** ✅ **SUCCEEDED** (Major milestone!)
**Error:** `building cudnn:x64-linux failed with: BUILD_FAILED`
**Root Cause:** vcpkg.json line 53 includes `"cudnn"` feature for opencv4, but cuDNN cannot be installed on GitHub runners (broken download)
**Fix Applied:** Removed `"cudnn"` from opencv4 features in base/vcpkg.json
**Lessons:**
- CMake configure succeeded - incremental caching is working!
- vcpkg cache was saved with partial build progress
- cudnn is not critical for ApraPipes CUDA functionality
- Failed to actively monitor again - user had to ask "status ?" 2 hours later

---

### Attempt 1: YAML syntax error - unmatched quote
**Run ID:** 20222101726
**Started:** 2025-12-15 06:00 UTC
**Result:** ❌ FAILED in 5m39s
**CMake Result:** Never reached (syntax error)
**Error:** `unexpected EOF while looking for matching '"'`
**Root Cause:** Line 119 had three quotes: `===\"` instead of `==="`
**Fix Applied:** Fixed quote syntax - removed extra quote
**Lessons:**
- Should have validated YAML syntax before committing
- Failed to actively monitor - user had to alert me 7 hours later
- Shameful: Wasted CI time on trivial syntax error

---
