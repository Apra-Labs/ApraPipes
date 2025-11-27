# Phase 1 Optimization Plan - Avoid continue-on-error

## Problem Analysis
- Phase 1 uses `continue-on-error: true` on CMake configure
- This hides real errors (libxml2 hash mismatch, glib distutils failure)
- Makes debugging difficult - "success" doesn't mean actual success
- Phase 1 goal: Cache OpenCV and dependencies for phase 2

## Root Constraints (from user)
1. **Time limit**: Public hosted agents have 1-hour max runtime
2. **Disk space**: Limited on public agents, need incremental caching
3. **Split strategy**: Phase 1 = prep/cache, Phase 2 = full build+test

## Current Phase 1 Workflow
```yaml
- Leave only OpenCV from vcpkg during prep phase (onlyOpenCV script)
- CMake configure (continue-on-error: true)  # <-- PROBLEM
- This populates cache with OpenCV build
```

## Proposed Solutions

### Option A: Direct vcpkg install (RECOMMENDED)
**Instead of using CMake to trigger vcpkg, directly call vcpkg install**

```yaml
- name: Install OpenCV to cache (Phase 1)
  run: |
    # Modify vcpkg.json to only include OpenCV
    .\base\fix-vcpkg-json.ps1 -onlyOpenCV

    # Directly install OpenCV using vcpkg
    .\vcpkg\vcpkg.exe install --triplet x64-windows

  # NO continue-on-error needed - will fail cleanly if OpenCV fails
```

**Advantages:**
- No hidden errors - vcpkg install fails loudly
- Cleaner separation: Phase 1 = vcpkg cache, Phase 2 = CMake build
- Easier to debug - vcpkg output is direct
- Still populates vcpkg cache for Phase 2

**Changes needed:**
1. Remove CMake configure from Phase 1
2. Add direct `vcpkg install` command
3. Phase 2 CMake will use cached OpenCV

---

### Option B: Check CMake exit code explicitly
**Keep CMake but validate it actually succeeded on OpenCV**

```yaml
- name: Configure CMake for OpenCV cache
  run: |
    cmake configure...

    # Check if OpenCV was actually installed
    if (!(Test-Path "$env:VCPKG_ROOT/installed/x64-windows/include/opencv2")) {
      Write-Error "OpenCV installation failed in Phase 1"
      exit 1
    }
  continue-on-error: false  # Real failures now caught
```

**Advantages:**
- Keeps current CMake-based approach
- Adds validation

**Disadvantages:**
- Still uses CMake indirection
- Harder to debug vcpkg-specific issues

---

### Option C: Separate vcpkg manifest for Phase 1
**Create a minimal vcpkg.json just for Phase 1**

```yaml
- name: Create Phase 1 minimal manifest
  run: |
    Copy-Item base/vcpkg.json base/vcpkg.json.backup
    # Create minimal vcpkg.json with only opencv4
    @{dependencies = @("opencv4")} | ConvertTo-Json | Out-File base/vcpkg.json

- name: Install Phase 1 dependencies
  run: .\vcpkg\vcpkg.exe install --triplet x64-windows

- name: Restore full manifest for Phase 2
  run: |
    Move-Item base/vcpkg.json.backup base/vcpkg.json -Force
```

**Advantages:**
- Very explicit about Phase 1 scope
- Clean vcpkg install

**Disadvantages:**
- More file manipulation
- Need to ensure manifest restore works

---

## Recommendation: **Option A** - Direct vcpkg install

**Implementation Plan:**
1. Phase 1: Remove CMake configure, add `vcpkg install` with onlyOpenCV manifest
2. Remove `continue-on-error: true`
3. Phase 2: Keep as-is (CMake will use cached OpenCV)
4. Benefit: Real errors (libxml2, glib) will fail Phase 1 immediately

**Testing Strategy:**
- Trigger Phase 1 only, verify it fails on real errors
- Verify cache is populated correctly
- Verify Phase 2 can use Phase 1 cache

---

## Additional Fixes Needed (Regardless of Option)

### 1. libxml2 Hash Mismatch
- **Issue**: GitLab changed libxml2 tarball hash
- **Status**: Should be fixed in microsoft/vcpkg master (be563dfee8)
- **Verify**: Check if our vcpkg submodule has the fix

### 2. Python distutils (glib build)
- **Issue**: glib needs distutils, Python 3.12 removed it
- **Fix Applied**: Downgraded vcpkg Python to 3.10.11 in vcpkg-tools.json
- **Status**: Needs verification in Phase 1 direct install

### 3. vcpkg Submodule Management
- **Current**: Modified vcpkg in detached HEAD, pushed to branch
- **Better**: Consider forking approach or vcpkg overlay ports

---

## Next Steps (Waiting for User Review)
1. User reviews this plan
2. Decide on Option A, B, or C
3. Implement chosen solution
4. Test Phase 1 fails properly on errors
5. Verify Phase 2 uses Phase 1 cache correctly
