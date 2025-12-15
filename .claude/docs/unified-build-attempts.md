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
