# Windows Build Troubleshooting

Platform-specific troubleshooting for Windows NoCUDA builds on GitHub-hosted runners.

**Scope**: Windows builds without CUDA, running on `windows-latest` GitHub-hosted runners with two-phase build strategy.

**For Windows CUDA builds**: See `troubleshooting.cuda.md`

---

## Windows-Specific Architecture

### Build Configuration
- **Runner**: `windows-latest` (GitHub-hosted)
- **Strategy**: Two-phase (Phase 1: prep/cache, Phase 2: build/test)
- **Time Limit**: 1 hour per phase
- **Disk Space**: Limited (~14 GB available)
- **Cache**: `C:\Users\runneradmin\AppData\Local\vcpkg\archives`

### Workflow Files
- **Top-level**: `.github/workflows/CI-Win-NoCUDA.yml`
- **Reusable**: `.github/workflows/build-test-win.yml`

### Key Characteristics
- PowerShell-based scripts
- Uses Chocolatey for system tools
- vcpkg downloads tools to `vcpkg/downloads/tools/`
- Two-phase build to work within 1-hour constraint

---

## Issue W1: Python distutils Missing

**Symptom**:
```
ModuleNotFoundError: No module named 'distutils'
```
- Occurs when building glib or similar packages
- Python 3.12+ detected in logs
- Build fails during glib build step

**Root Cause**:
- vcpkg downloads Python from `vcpkg/scripts/vcpkg-tools.json`
- Default vcpkg may specify Python 3.12+ which removed `distutils`
- glib build scripts require `distutils.core` module

**Diagnostic Steps**:
1. Check Python version in logs:
   ```
   grep "python.*3\.12\|python.*3\.13" /tmp/build.log
   ```

2. Check vcpkg-tools.json:
   ```powershell
   cat vcpkg/scripts/vcpkg-tools.json | Select-String "python"
   ```

3. Verify glib build failure:
   ```
   grep -A 10 "Building glib" /tmp/build.log | grep distutils
   ```

**Fix**:

1. Modify `vcpkg/scripts/vcpkg-tools.json` (in vcpkg submodule):
   ```json
   {
     "name": "python3",
     "os": "windows",
     "version": "3.10.11",
     "executable": "python.exe",
     "url": "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip",
     "sha512": "40cbc98137cc7768e3ea498920ddffd0b3b30308bfd7bbab2ed19d93d2e89db6b4430c7b54a0f17a594e8e10599537a643072e08cfd1a38c284f8703879dcc17",
     "archive": "python-3.10.11-embed-amd64.zip"
   }
   ```

2. Clear vcpkg downloads cache:
   - Add to workflow (already present in `build-test-win.yml`):
     ```yaml
     - name: Clear vcpkg downloads
       run: remove-item vcpkg/downloads -Recurse -Force -ErrorAction SilentlyContinue
     ```
   - OR bump cache key version (e.g., v5 → v6)

3. Push vcpkg submodule changes to feature branch (NOT master!)

4. Commit and test:
   ```bash
   cd vcpkg
   git checkout -b fix/python-3.10-downgrade
   git add scripts/vcpkg-tools.json
   git commit -m "Downgrade Python to 3.10.11 for distutils support"
   git push origin fix/python-3.10-downgrade
   cd ..
   git add vcpkg
   git commit -m "Update vcpkg: Downgrade Python to 3.10.11"
   ```

**Verification**:
- Build logs show: `python-3.10.11` being downloaded
- glib builds successfully without distutils errors
- No `ModuleNotFoundError` in logs

---

## Issue W2: continue-on-error Hiding Real Failures

**Symptom**:
- Phase 1 shows "success" but real errors occurred
- Build fails in Phase 2 with errors that should have appeared in Phase 1
- Confusing diagnostic - errors appear unrelated

**Root Cause**:
- Phase 1 uses `continue-on-error: true` on CMake configure step
- This allows partial completion (expected for caching workflow)
- But it also hides real errors (libxml2 hash, distutils failures)

**Diagnostic Approach**:

1. **Don't trust Phase 1 step status** - check actual logs:
   ```bash
   # Phase 1 may show green checkmark but have errors
   grep "error:" /tmp/build.log | grep "prep"
   ```

2. **Distinguish expected vs unexpected failures**:
   - Expected: CMake fails to find packages (only OpenCV installed)
   - Unexpected: vcpkg package build errors, hash mismatches

3. **Search for real errors**:
   ```bash
   # Hash mismatches
   grep "unexpected hash" /tmp/build.log

   # Package build failures
   grep "error:" /tmp/build.log | grep -v "Could NOT find"

   # Python errors
   grep -i "distutils\|ModuleNotFoundError" /tmp/build.log
   ```

4. **Check Phase 2 for inherited failures**:
   - If Phase 1 had hidden vcpkg errors, Phase 2 may fail with same issue
   - Phase 2 does NOT have `continue-on-error`, so it will fail properly

**Pattern Recognition**:

| Error Location | continue-on-error | Action |
|----------------|-------------------|--------|
| Phase 1 - CMake "Could NOT find" | Yes | Ignore (expected) |
| Phase 1 - vcpkg build errors | Yes | **Investigate** (real problem) |
| Phase 2 - Any error | No | Investigate (blocking issue) |

**Better Alternatives**:

1. **Temporarily remove continue-on-error** to see real failures:
   ```yaml
   # In build-test-win.yml
   - name: Configure CMake Common
     run: ...
     continue-on-error: false  # Temporarily change to false
   ```

2. **Add explicit validation after Phase 1**:
   ```yaml
   - name: Validate cache populated
     run: |
       if (!(Test-Path "vcpkg_installed/x64-windows/include/opencv2")) {
         Write-Error "Phase 1 failed to cache OpenCV"
         exit 1
       }
   ```

3. **Use direct vcpkg install**:
   ```yaml
   - name: Install OpenCV to cache
     run: |
       .\base\fix-vcpkg-json.ps1 -onlyOpenCV
       .\vcpkg\vcpkg.exe install --triplet x64-windows
     # No continue-on-error - fails loudly on real errors
   ```

**Key Lesson**: When debugging, check actual error messages in logs, not just workflow step status.

---

## Issue W3: PKG_CONFIG_EXECUTABLE Not Found

**Symptom**:
```
Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)
```
- CMake FindPkgConfig fails
- Occurs in Phase 2 during CMake configure
- May also appear in Phase 1 (ignore if `continue-on-error: true`)

**Root Cause**:
- CMake's FindPkgConfig module can't find compatible pkg-config executable
- System pkg-config (from chocolatey pkgconfiglite) may not be compatible
- vcpkg-provided pkgconf not installed

**Common Misdiagnosis**:
❌ "Need to add chocolatey bin to PATH"
- Wrong because: chocolatey already works (`choco install` succeeds in prep-cmd)
- Wrong because: Adding to PATH doesn't fix compatibility issues
- Wrong because: The tool exists, but CMake can't use it

**Correct Diagnosis**:
✅ CMake FindPkgConfig needs vcpkg's pkgconf package

**Diagnostic Steps**:
1. Check if error occurs in Phase 1 or Phase 2:
   ```bash
   grep -n "PKG_CONFIG_EXECUTABLE" /tmp/build.log
   # Check if line is in prep phase (has continue-on-error) or test phase
   ```

2. Check if pkgconf is in vcpkg.json:
   ```bash
   grep "pkgconf" base/vcpkg.json
   ```

3. If pkgconf installed, check CMake found it:
   ```bash
   grep "Found PkgConfig" /tmp/build.log
   ```

**Fix**:

Add `pkgconf` to vcpkg.json dependencies:

```json
{
  "dependencies": [
    "pkgconf",  // Add as first dependency
    // ... other dependencies
  ]
}
```

**Why This Works**:
- vcpkg's pkgconf package provides CMake integration
- CMake FindPkgConfig automatically finds vcpkg-installed pkgconf
- Located at: `vcpkg_installed/x64-windows/tools/pkgconf/pkgconf.exe`
- vcpkg CMake toolchain ensures FindPkgConfig looks here first

**Verification**:
- Build logs show: `Installing ... pkgconf:x64-windows@...`
- CMake logs show: `Found PkgConfig: D:/a/ApraPipes/ApraPipes/build/vcpkg_installed/x64-windows/tools/pkgconf/pkgconf.exe`
- No PKG_CONFIG_EXECUTABLE errors in Phase 2

**Key Lesson**: When CMake can't find a tool, provide it through vcpkg rather than modifying system PATH.

---

## Issue W4: Package Version Breaking Changes

**Symptom**:
- Build fails after vcpkg baseline update
- API-related errors (e.g., "Unsupported SFML component: system")
- Type mismatch errors (e.g., `sf::Int16` not found)
- CMake errors about missing components

**Example (SFML 3.x)**:
```
CMake Error at .../share/sfml/SFMLConfig.cmake:78 (message):
  Unsupported SFML component: system
Call Stack (most recent call first):
  CMakeLists.txt:51 (find_package)
```

**Root Cause**:
- vcpkg baseline updated to latest
- Package upgraded to new major version (e.g., SFML 2.6.x → 3.0.x)
- New major version has breaking API changes
- Code written for older API

**Diagnostic Steps**:

1. Check which package version was installed:
   ```bash
   grep "Installing.*sfml" /tmp/build.log
   # Output: Installing 140/141 sfml[audio,core,graphics,network,window]:x64-windows@3.0.2
   ```

2. Check CMakeLists.txt for package usage:
   ```bash
   grep -n "find_package.*SFML" base/CMakeLists.txt
   # Shows what components are requested
   ```

3. Check if package has version override:
   ```bash
   grep "sfml" base/vcpkg.json
   ```

**Fix**:

Pin package to compatible major version in `base/vcpkg.json`:

```json
{
  "overrides": [
    {
      "name": "sfml",
      "version": "2.6.2"
    }
  ]
}
```

**Why This Fix**:
- DevOps role: Fix build, not application code
- Code migration (SFML 2.x → 3.x) is developer task
- Pinning maintains build stability

**Verification**:
- Build logs show pinned version: `Installing ... sfml:x64-windows@2.6.2`
- CMake configure succeeds
- No API-related errors

**Prevention**:
Pin all critical dependencies upfront (see `reference.md` → Version Pinning Strategy)

---

## Issue W5: vcpkg Baseline Commit Not Fetchable

**Symptom**:
```
fatal: remote error: upload-pack: not our ref ae8fa5ae5e
error: failed to fetch ref ae8fa5ae5e from repository
```
- Occurs during vcpkg bootstrap or baseline fetch
- Happens after updating `vcpkg-configuration.json` baseline

**Root Cause**:
- Baseline commit exists in git history but not advertised by `git ls-remote`
- Git only advertises branch tips, tags, and HEAD
- Parent commits in middle of history aren't fetchable by default

**Diagnostic Steps**:

1. Check if baseline commit is advertised:
   ```bash
   $baseline = (Get-Content base/vcpkg-configuration.json | ConvertFrom-Json).'default-registry'.baseline
   git ls-remote https://github.com/Apra-Labs/vcpkg.git | Select-String $baseline
   # Empty output = NOT fetchable
   ```

2. Check if commit exists locally:
   ```bash
   cd vcpkg
   git log --oneline $baseline  # Works locally but not remotely
   ```

**Fix Options**:

**Option 1: Use Branch Tip** (Recommended):
```bash
# Find current branch tip
git ls-remote https://github.com/Apra-Labs/vcpkg.git | Select-String "refs/heads/master"
# Output: 3011303ba1... refs/heads/master

# Update baseline to branch tip
# Edit base/vcpkg-configuration.json:
"baseline": "3011303ba1f6586e8558a312d0543271fca072c6"
```

**Option 2: Create Git Tag**:
```bash
cd vcpkg
git tag baseline-2024-11-28 ae8fa5ae5e
git push origin baseline-2024-11-28
# Now ae8fa5ae5e is advertised via refs/tags/baseline-2024-11-28
```

**Option 3: Use Feature Branch** (Best Practice):
```bash
cd vcpkg
git checkout -b fix/update-baseline-2024-11-28 ae8fa5ae5e
git push origin fix/update-baseline-2024-11-28
# Use this branch's HEAD commit as baseline
```

**Verification**:
- vcpkg bootstrap succeeds
- No "not our ref" errors
- Baseline commit appears in `git ls-remote` output

**Prevention**:
Always verify commit is advertised before using as baseline:
```bash
git ls-remote https://github.com/Apra-Labs/vcpkg.git | grep <commit>
```

---

## Issue W6: Disk Space Exhausted

**Symptom**:
```
No space left on device
```
- Build fails during vcpkg install or CMake build
- More common in Phase 2 (full build)
- Windows hosted runners have ~14 GB available

**Root Cause**:
- vcpkg downloads accumulate in `vcpkg/downloads/`
- Build artifacts (*.pdb, *.ilk) are large
- OpenCV + CUDA features + all dependencies exceed available space

**Diagnostic Steps**:

1. Check disk usage in logs:
   ```bash
   grep -i "disk\|space" /tmp/build.log
   ```

2. Estimate package sizes:
   - OpenCV with CUDA: ~3-4 GB
   - Boost: ~2-3 GB
   - FFmpeg: ~1-2 GB

**Fix** (Already implemented in build-test-win.yml):

```yaml
- name: Remove files not needed for the build
  if: ${{!inputs.is-selfhosted}}
  working-directory: ${{github.workspace}}
  run: |
    remove-item vcpkg/downloads -Recurse -Force
    remove-item * -Recurse -Force -Include *.pdb,*.ilk
  shell: pwsh
  continue-on-error: true
```

**When to Run**:
- After Phase 1 completes (before Phase 2)
- After vcpkg install completes (before CMake build)

**What Gets Removed**:
- `vcpkg/downloads/`: Cached package downloads (no longer needed)
- `*.pdb`: Windows debug symbols
- `*.ilk`: Incremental link files

**Verification**:
- Build completes without "No space left" errors
- Disk usage stays under 14 GB limit

---

## Issue W7: Boost filesystem::extension() API Breaking Change

**Symptom**:
```
error C2039: 'extension': is not a member of 'boost::filesystem'
error C3861: 'extension': identifier not found
```

**Files Affected**:
- `base/src/Mp4WriterSinkUtils.cpp` (lines 70, 177)
- `base/src/OrderedCacheOfFiles.cpp` (line 881)
- `base/test/mp4_simul_read_write_tests.cpp` (lines 269, 374, 478, 607, 778, 883, 997, 1097)

**Root Cause**:
- Boost 1.84.0+ removed the free function `boost::filesystem::extension(path)`
- Must use member function `path.extension()` instead
- This affects code written for older Boost versions (< 1.60)

**Diagnostic Steps**:

1. Check Boost version in logs:
   ```bash
   grep "Boost.*version" /tmp/build.log
   ```

2. Search for problematic pattern:
   ```bash
   grep -n "boost::filesystem::extension\|fs::extension" base/src/*.cpp base/test/*.cpp
   ```

**Fix**:

Change from:
```cpp
// Old API (deprecated in Boost 1.60+, removed in 1.84+)
boost::filesystem::extension(path_var)
fs::extension(path_var)
```

To:
```cpp
// New API (available since Boost 1.60)
path_var.extension()
```

**Example Fix**:
```cpp
// Before:
if (boost::filesystem::is_regular_file(dirPath) &&
    boost::filesystem::extension(dirPath) == ".mp4")

// After:
if (boost::filesystem::is_regular_file(dirPath) &&
    dirPath.extension() == ".mp4")
```

**Example Fixes**:
- `Mp4WriterSinkUtils.cpp:70,177` - `boost::filesystem::path(baseFolder).extension()`
- `OrderedCacheOfFiles.cpp:881` - `path.extension()`
- `mp4_simul_read_write_tests.cpp` - `dirPath.extension()` (8 occurrences)

**Verification**:
```bash
# Ensure no old API calls remain
grep -r "boost::filesystem::extension\(" base/
grep -r "[^.]fs::extension\(" base/
# Should return no results
```

---

## Windows-Specific Debugging

### PowerShell Diagnostic Commands

```powershell
# Check Python version
python --version

# List vcpkg installed packages
.\vcpkg\vcpkg.exe list

# Check cache directory size
Get-ChildItem -Recurse "C:\Users\runneradmin\AppData\Local\vcpkg\archives" | Measure-Object -Property Length -Sum

# Check disk space
Get-PSDrive C

# Check environment variables
Get-ChildItem Env: | Where-Object {$_.Name -like "*VCPKG*" -or $_.Name -like "*CMAKE*"}

# Check chocolatey packages
choco list --local-only
```

### Windows-Specific Log Locations

| Log Type | Location |
|----------|----------|
| vcpkg package builds | `vcpkg/buildtrees/<package>/install-x64-windows-*.log` |
| vcpkg package config | `vcpkg/buildtrees/<package>/config-x64-windows-*.log` |
| CMake configure | Build step output in GitHub Actions logs |
| CMake build | Build step output in GitHub Actions logs |
| Test results | `CI_test_result_Win-nocuda.xml` (uploaded artifact) |

### Common Grep Patterns for Windows Logs

```bash
# Find PowerShell errors
grep "At line:" /tmp/build.log

# Find MSBuild errors
grep "error MSB" /tmp/build.log

# Find MSVC compiler errors
grep "error C" /tmp/build.log

# Find link errors
grep "error LNK" /tmp/build.log

# Find specific package failures
grep "Building.*<package>" /tmp/build.log -A 50
```

---

## Windows-Specific Quick Fixes Checklist

Before digging deep, check these common issues:

### Phase 1 (Prep) Checklist
- [ ] Python version is 3.10.x in vcpkg-tools.json
- [ ] vcpkg downloads cleared or cache key bumped
- [ ] Baseline commit is fetchable (git ls-remote)
- [ ] vcpkg-configuration.json has correct baseline
- [ ] Ignore CMake "Could NOT find" errors (expected)
- [ ] Check for real vcpkg build errors (unexpected)

### Phase 2 (Build/Test) Checklist
- [ ] Cache restored successfully from Phase 1
- [ ] pkgconf in vcpkg.json dependencies
- [ ] All pinned versions in overrides section
- [ ] Disk space sufficient (< 14 GB used)
- [ ] CMake finds all required packages
- [ ] No continue-on-error hiding errors

### After Fixing Checklist
- [ ] Commit vcpkg submodule changes to feature branch (not master!)
- [ ] Update vcpkg-configuration.json baseline if needed
- [ ] Bump cache key if vcpkg-tools.json changed
- [ ] Test Phase 1 first, then Phase 2
- [ ] Verify cache saved and restored correctly
- [ ] Document new issue pattern in this guide

---

## Escalation Criteria

### When to Ask for Help

1. **Unknown Error Patterns**: Error doesn't match any documented pattern
2. **Security Decisions**: Accepting packages with known vulnerabilities
3. **Breaking Changes**: Library upgrade requires code changes
4. **Infrastructure Issues**: Runner configuration, credentials, permissions

### Before Escalating

Gather this information:
1. Run ID and workflow name
2. Platform and build type (Win-nocuda, Phase 1 vs 2)
3. Relevant log excerpts (not entire logs)
4. What you tried (commands, fixes attempted)
5. Current hypothesis about root cause

---

**Applies to**: Windows NoCUDA builds on GitHub-hosted runners
**Related Guides**: reference.md, troubleshooting.cuda.md (for Windows CUDA)
