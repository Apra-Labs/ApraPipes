# Windows CUDA CI Build Fix Analysis

**Date:** 2025-12-08
**Branch:** fix/ci-additional-workflows
**Target:** CI-Win-CUDA workflow on self-hosted Windows runner

## Current Status

**Latest Run:** 20033688684 (queued - with libarchive 3.5.2 fix)
**Previous Run:** 20033183164 (failed - libarchive 3.7.7)
**Recent Failures:** 20019274614, 20017051688

**Fix Applied:** libarchive downgraded from 3.7.7 to 3.5.2 (commit 4ea16f50)

## Root Cause Analysis

### Primary Issues Identified

1. **Python Installation Failure**
   - Chocolatey fails to install Python 3.10.11
   - Error: `python package files install failed with exit code 1`
   - Causes downstream pip3/ninja/meson failures

2. **CMake Configuration Error**
   - Error: `CMake Error at CMakeLists.txt:1 (cmake_minimum_required)`
   - Indicates CMake version incompatibility
   - Workflow specifies CMake 3.29.6 but may not be installing correctly

3. **Missing vcpkg Dependencies**
   - Missing `pkgconf` package (added in main #437)
   - Version mismatches causing linking errors

## Successful Fixes from Main Branch

**PR #437** (commit 54d51f04) fixed Windows NoCUDA and other builds with:

### vcpkg.json Changes

```json
{
  "dependencies": [
    "pkgconf",  // NEW: Required for build configuration
    // ... other deps
  ],
  "overrides": [
    {
      "name": "openal-soft",
      "version": "1.23.1"  // CHANGED: Pinned from 1.24.x to avoid fmt linking errors
    },
    {
      "name": "libarchive",
      "version": "3.5.2"   // CHANGED: Downgraded from 3.7.7
    },
    {
      "name": "opencv4",
      "version": "4.8.0"    // CHANGED: Downgraded from 4.10.0
    }
  ]
}
```

### Key Learnings

1. **OpenAL-Soft v1.24.x breaks builds** - introduces fmt dependency linking errors
2. **libarchive 3.7.7 has issues** - safer to use 3.5.2
3. **OpenCV 4.8.0 more stable** than 4.10.0 for CUDA builds
4. **pkgconf is essential** for proper dependency resolution

## Current Branch State

### Our vcpkg.json (fix/ci-additional-workflows)

```json
{
  "overrides": [
    {
      "name": "libarchive",
      "version": "3.7.7"  // ❌ Should be 3.5.2
    },
    {
      "name": "opencv4",
      "version": "4.10.0",  // ⚠️  We keep 4.10.0 globally
      "port-version": 6
    },
    {
      "name": "openal-soft",
      "version": "1.23.1"  // ✅ Already correct
    },
    {
      "name": "nu-book-zxing-cpp",
      "version": "2.2.1"  // ✅ Correct for gcc-8
    }
  ],
  "dependencies": [
    "pkgconf",  // ❓ Need to verify if already added
    // ... other deps
  ]
}
```

**Special Case:** We use ARM64 overlay port for OpenCV 4.8.0 on ARM64 while keeping 4.10.0 for other platforms.

## Recommended Fixes

### Priority 1: vcpkg.json Updates

```json
{
  "overrides": [
    {
      "name": "libarchive",
      "version": "3.5.2"  // CHANGE from 3.7.7
    }
  ],
  "dependencies": [
    "pkgconf",  // ENSURE this is present
    // ... rest
  ]
}
```

### Priority 2: Windows Workflow Updates

**File:** `.github/workflows/build-test-win.yml`

**Current prep-cmd:**
```yaml
prep-cmd: 'choco install python --version=3.10.11 --force && refreshenv && pip3 install ninja && pip3 install meson && choco feature enable -n allowEmptyChecksums && choco install pkgconfiglite && choco install cmake --version=3.29.6 --force'
```

**Issues:**
- Python install fails but continues
- No error handling
- Dependencies fail if Python fails

**Proposed fix:**
```yaml
prep-cmd: |
  # Try Python 3.10, fallback to 3.11 if fails
  choco install python --version=3.10.11 --force --ignore-checksums || choco install python3 --force
  refreshenv
  python --version

  # Install build tools
  pip3 install ninja meson || python -m pip install ninja meson

  # Install pkg-config and cmake
  choco feature enable -n allowEmptyChecksums
  choco install pkgconfiglite --force
  choco install cmake --version=3.29.6 --force --ignore-checksums || choco install cmake --force
```

### Priority 3: Add continue-on-error for Prep Phase

Add to workflow:

```yaml
- name: Prepare builder
  continue-on-error: true  # Don't fail if some tools already installed
  run: |
    ${{ inputs.prep-cmd }}
```

Then add validation step:

```yaml
- name: Validate builder prerequisites
  run: |
    ${{ inputs.prep-check-cmd }}
```

## Testing Strategy

### Phase 1: Verify Current Build
1. Monitor run 20033183164
2. Document exact failure point
3. Confirm it matches analysis above

### Phase 2: Apply vcpkg Fixes
1. Update `base/vcpkg.json`:
   - Change libarchive to 3.5.2
   - Verify pkgconf is in dependencies
2. Test locally on Windows CUDA runner if possible
3. Commit and push
4. Monitor build

### Phase 3: Apply Workflow Fixes (if needed)
1. Update prep-cmd with fallbacks
2. Add continue-on-error strategically
3. Test and iterate

## Files to Modify

1. **base/vcpkg.json**
   - Line ~12-14: Change libarchive version
   - Line ~37: Verify pkgconf present

2. **.github/workflows/build-test-win.yml** (if Phase 2 insufficient)
   - Lines ~39-45: Update prep-cmd with error handling
   - Add validation steps

3. **.github/workflows/CI-Win-CUDA.yml**
   - Possibly add force-cache-update option to default input

## SSH Access for Debugging

**Windows CUDA Runner:**
- Host: `utubovyu.users.openrport.io`
- Port: `22179` (varies with tunnel)
- User: `administrator`
- Password: (ask user)

**Connection:**
```bash
sshpass -p '<password>' ssh -o StrictHostKeyChecking=no -p <port> administrator@utubovyu.users.openrport.io
```

**Useful commands:**
```powershell
# Check Python
python --version
where python

# Check CMake
cmake --version
where cmake

# Check vcpkg cache
Get-ChildItem "C:\Users\runneradmin\AppData\Local\vcpkg\archives"

# Check runner status
Get-Service | Where-Object {$_.DisplayName -like "*GitHub Actions*"}

# View GitHub Actions runner logs
Get-ChildItem "C:\_work\_diag" -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

## Expected Outcome

After applying these fixes:

1. ✅ Python installation succeeds or gracefully falls back
2. ✅ CMake 3.29.6 is available
3. ✅ vcpkg dependencies resolve correctly
4. ✅ Build completes without CMake errors
5. ✅ Tests run (may have failures but build succeeds)

## References

- **Main PR #437:** 54d51f04 - "fix(ci): Restore Windows NoCUDA builds with vcpkg dependency fixes"
- **vcpkg baseline:** 4658624c5f19c1b468b62fe13ed202514dfd463e
- **OpenAL-Soft issue:** v1.24.x introduces fmt dependency causing linking errors
- **OpenCV overlay:** thirdparty/custom-overlay/opencv4/ (for ARM64 gcc-8 compatibility)

## Next Steps

1. **Wait for current run** (20033183164) to complete
2. **Apply vcpkg.json fixes** (libarchive downgrade)
3. **Test build**
4. **Apply workflow fixes if needed**
5. **Document results**
6. **Merge to main** when green

---

**Note:** This document should be deleted after fixes are applied and CI is green.
