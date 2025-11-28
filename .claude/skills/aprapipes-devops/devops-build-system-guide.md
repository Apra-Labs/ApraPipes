# ApraPipes DevOps Build System Guide

## Overview

ApraPipes uses GitHub Actions for CI/CD across multiple platforms: Windows (with/without CUDA), Linux x64, and Linux ARM64. The build system is designed to work within the constraints of public GitHub-hosted runners while also supporting self-hosted runners.

## Key Constraints

### Public Hosted Runners
1. **Time Limit**: Maximum 1-hour runtime per job
2. **Disk Space**: Limited disk space (requires incremental caching)
3. **Cost**: Build minutes are metered, so efficiency matters

### Self-Hosted Runners
- No time or disk constraints
- CUDA builds require specialized hardware
- Can skip caching mechanisms

## Architecture

### Two-Phase Build Strategy (Public Runners Only)

To work within the 1-hour constraint and disk space limits, builds are split into two phases:

#### Phase 1: Prep/Cache Phase (`is-prep-phase: true`)
- **Goal**: Install heavy dependencies (especially OpenCV) and cache them
- **Trigger**: Manual (workflow_dispatch)
- **Process**:
  1. Modify `vcpkg.json` to only include OpenCV (using `fix-vcpkg-json.ps1 -onlyOpenCV`)
  2. Run CMake configure to trigger vcpkg installation
  3. Cache the vcpkg archives to GitHub Actions cache
  4. **Current Issue**: Uses `continue-on-error: true` which hides real failures
- **Cache Key**: `${{ inputs.flav }}-4-${{ hashFiles('base/vcpkg.json', 'vcpkg/baseline.json', 'submodule_ver.txt') }}`
- **Output**: Populated vcpkg cache for Phase 2

#### Phase 2: Build/Test Phase (`is-prep-phase: false`)
- **Goal**: Full build and test execution
- **Trigger**: Manual (workflow_dispatch)
- **Process**:
  1. Restore vcpkg cache from Phase 1
  2. Run full CMake configure with all dependencies
  3. Build the project
  4. Run unit tests
  5. Upload test results and logs
- **Time Saved**: OpenCV (largest dependency) is already cached

### Single-Phase Build (Self-Hosted Runners)
- No caching needed
- Direct build from checkout to test
- Used for CUDA builds (require specialized hardware)

## Workflow Structure

### Top-Level Workflows
Located in `.github/workflows/`, one per build configuration:
- `CI-Win-NoCUDA.yml` - Windows without CUDA (2 phases)
- `CI-Win-CUDA.yml` - Windows with CUDA (self-hosted, 1 phase)
- `CI-Linux-x64.yml` - Linux x64 (2 phases)
- `CI-Linux-ARM64.yml` - Linux ARM64 (2 phases)
- Plus corresponding cuda variants

Each top-level workflow:
1. Defines the specific runner and parameters
2. Calls the reusable workflow `build-test-win.yml` or `build-test-linux.yml`
3. Can be triggered manually via `workflow_dispatch`

### Reusable Workflows

#### `build-test-win.yml`
Parameterized workflow for all Windows builds.

**Key Inputs:**
- `runner`: Which GitHub runner to use (e.g., `windows-latest`)
- `flav`: Build flavor (e.g., `Windows_NoCUDA`, `Windows_CUDA`)
- `buildConf`: Build configuration (Release/Debug)
- `is-selfhosted`: Boolean - skip caching if true
- `is-prep-phase`: Boolean - if true, only cache dependencies
- `cuda`: ON/OFF - enable/disable CUDA
- `prep-cmd`: Commands to prepare the builder (install tools)
- `bootstrap-cmd`: Commands to bootstrap vcpkg
- `cmake-conf-cmd`: CMake configuration command

**Build Steps:**
1. **Prepare builder**: Install Python, CMake, Ninja, Meson, pkg-config
2. **Check for CUDA**: If CUDA build, verify nvcc and cudnn
3. **Checkout code**: Recursive submodule checkout with LFS
4. **Bootstrap vcpkg**: Run `vcpkg/bootstrap-vcpkg.bat`
5. **Modify vcpkg.json**: Remove CUDA packages if NoCUDA, or keep only OpenCV if prep phase
6. **Cache dependencies**: Restore/save vcpkg archives cache
7. **Configure CMake**: Run CMake with vcpkg toolchain
8. **Build**: Compile the project (Phase 2 only)
9. **Test**: Run unit tests (Phase 2 only)
10. **Upload artifacts**: Test results and build logs

#### `build-test-linux.yml`
Similar structure to Windows workflow but for Linux builds.

## Dependency Management

### vcpkg Package Manager
- **Purpose**: Manages C++ dependencies (OpenCV, Boost, FFmpeg, etc.)
- **Configuration**: `base/vcpkg.json` defines all dependencies
- **Baseline**: Points to specific vcpkg repository commit (in `base/vcpkg-configuration.json`)
- **Toolchain**: `vcpkg/scripts/buildsystems/vcpkg.cmake` integrates with CMake

### vcpkg Tools Configuration
- **File**: `vcpkg/scripts/vcpkg-tools.json`
- **Purpose**: Defines which versions of Python, CMake, etc. vcpkg downloads
- **Critical**: Python version must be 3.10.x (has distutils) not 3.12+ (distutils removed)

### Version Pinning Strategy (Critical for Stability)

**Core Principle**: Pin major versions of all production dependencies to avoid breaking changes.

#### Why Version Pinning Matters
When updating vcpkg baseline without version overrides:
- Any package can jump to a new major version
- Breaking API changes can silently break your build
- Errors appear unrelated to dependency update
- Builds become non-reproducible ("worked yesterday" syndrome)

#### Current Pinned Versions
```json
"overrides": [
  { "name": "ffmpeg", "version": "4.4.3" },      // Pinned to 4.x
  { "name": "libarchive", "version": "3.5.2" },  // Pinned to 3.x
  { "name": "sfml", "version": "2.6.2" }         // Pinned to 2.x (SFML 3.x has breaking changes)
]
```

#### Version Pinning Tiers

**Tier 1 - Critical Dependencies (MUST Pin)**:
- Packages with complex APIs your code directly uses
- Libraries with history of breaking changes between major versions
- Examples: opencv4, boost, gtk3, sfml, ffmpeg

**Tier 2 - Medium Dependencies (Consider Pinning)**:
- Packages used in specific modules
- Rapidly evolving libraries (AI/ML)
- Examples: whisper, nu-book-zxing-cpp

**Tier 3 - Low-Risk Utilities (Can Use Baseline)**:
- Rarely have breaking API changes
- Updated primarily for security/bug fixes
- Examples: pkgconf, zlib, bzip2, liblzma, brotli

#### Version Update Process
When intentionally upgrading a pinned package:
1. Create dedicated branch: `update/opencv-4.8-to-4.9`
2. Update version override in vcpkg.json
3. Review upstream changelog for breaking changes
4. Update code if needed
5. Test thoroughly
6. Document migration in commit message

### Key Dependencies
- **OpenCV**: Largest dependency, includes CUDA/cuDNN features
- **glib**: Requires Python with distutils for build
- **FFmpeg**: Video processing (pinned to 4.4.3)
- **Boost**: C++ utilities
- **SFML**: Audio/graphics (pinned to 2.6.2 - SFML 3.x has breaking API changes)
- **libxml2**: XML parsing (dependency of glib)

## Build Configuration Scripts

### `base/fix-vcpkg-json.ps1`
PowerShell script to modify `vcpkg.json` on-the-fly:
- `-removeCUDA`: Remove CUDA-related packages for NoCUDA builds
- `-onlyOpenCV`: Keep only OpenCV (for Phase 1 prep)

**Usage in workflows:**
```yaml
- name: Remove CUDA from vcpkg if we are in nocuda
  if: ${{ contains(inputs.cuda,'OFF')}}
  working-directory: ${{github.workspace}}/base
  run: .\fix-vcpkg-json.ps1 -removeCUDA
  shell: pwsh

- name: Leave only OpenCV from vcpkg during prep phase
  if: inputs.is-prep-phase
  working-directory: ${{github.workspace}}/base
  run: .\fix-vcpkg-json.ps1 -onlyOpenCV
  shell: pwsh
```

## Caching Strategy

### Cache Path
- **Windows**: `C:\Users\runneradmin\AppData\Local\vcpkg\archives`
- **Linux**: `~/.cache/vcpkg`

### Cache Key Components
```yaml
key: ${{ inputs.flav }}-4-${{ hashFiles('base/vcpkg.json', 'vcpkg/baseline.json', 'submodule_ver.txt') }}
restore-keys: ${{ inputs.flav }}-
```

**Why these files:**
- `base/vcpkg.json`: Changes when dependencies added/removed
- `vcpkg/baseline.json`: Changes when package versions updated
- `submodule_ver.txt`: Changes when vcpkg submodule updated

### Cache Behavior
- **Cache hit**: Phase 2 uses cached OpenCV from Phase 1
- **Cache miss**: Phase 2 must build OpenCV from scratch (may exceed 1 hour)
- **Partial match**: Uses `restore-keys` to get closest previous cache

## Common Issues and Solutions

### Issue 1: Python distutils Missing
**Symptom**: `ModuleNotFoundError: No module named 'distutils'` when building glib
**Root Cause**: vcpkg using Python 3.12+ which removed distutils
**Solution**: Modify `vcpkg/scripts/vcpkg-tools.json` to use Python 3.10.11
```json
{
  "name": "python3",
  "version": "3.10.11",
  "url": "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip",
  "sha512": "40cbc98137cc7768e3ea498920ddffd0b3b30308bfd7bbab2ed19d93d2e89db6b4430c7b54a0f17a594e8e10599537a643072e08cfd1a38c284f8703879dcc17"
}
```

### Issue 2: libxml2 Hash Mismatch
**Symptom**: `error: download from gitlab.gnome.org had an unexpected hash`
**Root Cause**: Stale vcpkg baseline, upstream changed file
**Solution**: Update vcpkg submodule to latest microsoft/vcpkg
```bash
cd vcpkg
git fetch microsoft
git checkout microsoft/master
cd ..
git add vcpkg
git commit -m "Update vcpkg to fix libxml2 hash"
```

### Issue 3: continue-on-error Hides Failures (Phase 1)
**Symptom**: Phase 1 shows "success" even when CMake configure fails
**Root Cause**: `continue-on-error: true` on CMake step
**Current Workaround**: Phase 1 is expected to fail at CMake (missing dependencies)
**Proposed Solution**: Replace CMake with direct vcpkg install (see phase1-optimization-plan.md)

### Issue 4: vcpkg Submodule Commit Not Fetchable
**Symptom**: `fatal: remote error: upload-pack: not our ref <hash>`
**Root Cause**: Modified vcpkg in detached HEAD, not pushed to branch
**Solution**: Create branch in vcpkg submodule and push to remote
```bash
cd vcpkg
git checkout -b <branch-name>
git push origin <branch-name>
cd ..
git add vcpkg
```

### Issue 5: Disk Space Exhausted
**Symptom**: Build fails with "No space left on device"
**Solution**: Clean up unnecessary files during build
```yaml
- name: Remove files not needed for the build
  if: ${{!inputs.is-selfhosted}}
  run: |
    remove-item vcpkg/downloads -Recurse -Force
    remove-item * -Recurse -Force -Include *.pdb,*.ilk
```

### Issue 6: vcpkg Registry Baseline Outdated
**Symptom**: Multiple packages fail with hash mismatches or version issues
**Root Cause**: `base/vcpkg-configuration.json` has stale baseline pointing to old vcpkg fork
**Solution**: Update baseline to latest commit in fork
```bash
# Check current baseline
cat base/vcpkg-configuration.json

# Update vcpkg fork with latest microsoft/vcpkg
cd vcpkg
git fetch microsoft
git checkout -b fix/rebase-on-latest microsoft/master
git cherry-pick <your-custom-changes>
git push origin fix/rebase-on-latest

# Update baseline in configuration
cd ../base
# Edit vcpkg-configuration.json to use new commit hash
git add vcpkg-configuration.json
```

### Issue 7: Python Version Cached by vcpkg
**Symptom**: vcpkg-tools.json changes ignored, still uses old Python version
**Root Cause**: vcpkg caches downloaded tools in `vcpkg/downloads/tools/`
**Solution**: Clear downloads directory or bump cache key
```yaml
# Add workflow step to clear vcpkg downloads
- name: Clear vcpkg downloads to force fresh tool download
  run: remove-item vcpkg/downloads -Recurse -Force -ErrorAction SilentlyContinue

# Or bump cache key version
key: ${{ inputs.flav }}-5-${{ hashFiles(...) }}  # Changed from -4- to -5-
```

### Issue 8: vcpkg Baseline Commit Not Fetchable
**Symptom**: `error: failed to fetch ref <hash> from repository` or "not our ref"
**Root Cause**: The baseline commit exists in git history but isn't advertised by `git ls-remote`
- Git only advertises branch tips, tags, and HEAD
- Commits in the middle of history aren't fetchable by default

**Why This Happens**:
```bash
# This commit is in history but not advertised
ae8fa5ae5e [fontconfig] Update (#48484)  # Parent of branch tip

# Only these are advertised and fetchable:
dfa17587b2  refs/heads/fix/rebase-on-microsoft-master-with-python310  # Branch tip
3011303ba1  refs/heads/master  # Branch tip
```

**Solution**: Always use commits that are advertised
```bash
# Check what commits are fetchable
git ls-remote https://github.com/Apra-Labs/vcpkg.git

# Use branch tip commits or tags as baselines
# Good: HEAD of a branch
"baseline": "3011303ba1f6586e8558a312d0543271fca072c6"  # master HEAD

# Bad: Parent commit in history
"baseline": "ae8fa5ae5e3b4d53d1ef5abdf1b6d0be6d37e806"  # Not advertised
```

**Alternative**: Create a git tag for the baseline
```bash
cd vcpkg
git tag baseline-2025-11-27 ae8fa5ae5e
git push origin baseline-2025-11-27
# Now ae8fa5ae5e is advertised via refs/tags/baseline-2025-11-27
```

### Issue 9: PKG_CONFIG_EXECUTABLE Not Found
**Symptom**: `Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)`
**Root Cause**: CMake's FindPkgConfig module can't find a compatible pkg-config executable
**Common Misdiagnosis**: Trying to add pkg-config directories to PATH

**Why PATH fixes don't work**:
- The system already has pkg-config installed (e.g., via chocolatey on Windows)
- CMake FindPkgConfig is looking for a specific executable format/version
- Adding to PATH doesn't solve incompatibility issues

**Correct Solution**: Add pkgconf to vcpkg dependencies
```json
{
  "dependencies": [
    "pkgconf",  // vcpkg's pkg-config implementation
    // ... other dependencies
  ]
}
```

**Why this works**:
- vcpkg provides pkgconf package with CMake integration
- FindPkgConfig automatically finds vcpkg-installed pkgconf
- Located at: `vcpkg_installed/x64-windows/tools/pkgconf/pkgconf.exe`

**Key Lesson**: When CMake can't find a tool, provide it through vcpkg rather than modifying system PATH.

### Issue 10: Package Version Breaking Changes
**Symptom**: Build fails after vcpkg baseline update with API-related errors
**Root Cause**: Package upgraded to new major version with breaking changes
**Example**: SFML 2.x → 3.x removed "system" component, changed `sf::Int16` to `std::int16_t`

**Solution**: Pin package to compatible major version
```json
{
  "overrides": [
    { "name": "sfml", "version": "2.6.2" }  // Prevents auto-upgrade to 3.x
  ]
}
```

**Prevention Strategy**:
1. Pin major versions of all critical dependencies
2. Test baseline updates in isolated branches first
3. Review vcpkg changelog before updating baseline
4. Use version overrides liberally - unpinned packages are time bombs

**DevOps Principle**: Fix the build, not the code. Code migration (e.g., SFML 2.x → 3.x) is a separate developer task, not a CI fix task.

### Issue 11: "No Version Database Entry" Errors After Baseline Update
**Symptom**: `error: no version database entry for boost-chrono at 1.89.0`
**Root Cause**: The baseline commit doesn't have version database files in `versions/` directory
**Common Causes**:
1. Used wrong commit (not from vcpkg repository)
2. Baseline commit is before version database was created
3. Fork diverged from microsoft/vcpkg without version files

**Solution**: Verify baseline has version database
```bash
# Check if baseline has version files
git ls-tree <baseline-commit> versions/baseline.json
git ls-tree <baseline-commit> versions/b-/boost-chrono.json

# Ensure baseline is from microsoft/vcpkg or a proper fork
git log --oneline <baseline-commit> | head -20

# If missing, use a newer commit from microsoft/vcpkg
```

### Issue 12: continue-on-error Hiding Real Failures
**Symptom**: Workflow shows "success" but actual errors occurred
**Root Cause**: `continue-on-error: true` prevents step failure from stopping workflow
**When it happens**: Commonly used in Phase 1 (prep) to allow partial completion

**Problem**: Makes debugging difficult
- Real errors (libxml2 hash, distutils failures) appear as "success"
- Hard to distinguish expected failures from real problems
- "Success" doesn't mean actual success

**Diagnostic Approach**:
1. **Don't trust step status** - check actual error messages in logs
2. **Search for "error:" in logs** - not just workflow status
3. **Ignore errors from steps with continue-on-error** - focus on steps without it
4. **Look for CMake Error, Build failed, etc.** - actual failure indicators

**Better Alternatives**:
- Use direct vcpkg install instead of CMake for caching phases
- Add explicit validation steps after continue-on-error steps
- Temporarily remove continue-on-error to see real failures

## vcpkg Fork Management Best Practices

### CRITICAL: Never Modify Master Branches
**❌ NEVER DO THIS:**
```bash
# DON'T push to master of forks
git checkout master
git merge fix/my-changes
git push origin master  # DANGEROUS - breaks other builds!
```

**Why This Is Dangerous**:
- Other builds/projects may depend on the fork's master
- Can break production CI pipelines
- Hard to revert without force-push (often blocked)
- Violates protected branch policies

**✅ ALWAYS DO THIS INSTEAD:**
```bash
# Use feature branches
git checkout -b fix/vcpkg-update-2025-11-27
git push origin fix/vcpkg-update-2025-11-27

# Update baseline to point to feature branch HEAD
# In vcpkg-configuration.json:
"baseline": "<commit-from-feature-branch>"
```

### Proper vcpkg Fork Update Workflow

#### Option 1: Feature Branch (Recommended)
```bash
# 1. Create feature branch from microsoft/vcpkg
cd vcpkg
git fetch microsoft
git checkout -b fix/update-2025-11-27 microsoft/master

# 2. Cherry-pick your custom changes
git cherry-pick <your-python-fix-commit>

# 3. Push to fork
git push origin fix/update-2025-11-27

# 4. Get the commit hash
BASELINE=$(git rev-parse HEAD)

# 5. Update ApraPipes configuration
cd ../base
# Edit vcpkg-configuration.json with $BASELINE
git commit -am "ci: Update vcpkg baseline to fix/update-2025-11-27"
```

#### Option 2: Git Tags (For Stable Baselines)
```bash
# Tag a specific commit
cd vcpkg
git tag baseline-2025-11-27-stable dfa17587b2
git push origin baseline-2025-11-27-stable

# Use the tag commit in configuration
"baseline": "dfa17587b27fcb5642e74632e49b3f9775aa1c19"
```

#### Option 3: Overlay Ports (For Custom Packages)
For small modifications, use vcpkg overlay-ports instead of forking:
```yaml
# vcpkg-configuration.json already has:
"overlay-ports": [
  "../thirdparty/custom-overlay"
]

# Put modified portfiles in thirdparty/custom-overlay/
thirdparty/custom-overlay/
  glib/
    portfile.cmake
    vcpkg.json
```

### Mistake Recovery: What If You Already Pushed to Master?

If you accidentally pushed to fork's master branch:

1. **Don't panic** - The damage is done but containable
2. **Create a revert commit** (if master isn't protected):
```bash
git revert HEAD --no-edit
git push origin master
```

3. **Or create a hotfix branch** from the previous good commit:
```bash
git checkout -b hotfix/restore-master <previous-good-commit>
git push origin hotfix/restore-master
# Ask repo owner to reset master to this commit
```

4. **Communicate**: Notify other users of the fork about the change
5. **Use feature branch going forward**: Don't repeat the mistake

### Lessons from Build #6-#9

**What Went Wrong**:
1. Created baseline with commit `ae8fa5ae5e` (microsoft/vcpkg parent)
2. That commit wasn't advertised by `git ls-remote`
3. vcpkg couldn't fetch it → "no version database entry" errors
4. **MISTAKE**: Merged to Apra-Labs/vcpkg master (should have used feature branch)
5. Fixed by using merge commit `3011303ba1` (advertised)

**What Should Have Been Done**:
1. Keep fixes on feature branch: `fix/rebase-on-microsoft-master-with-python310`
2. Use feature branch tip as baseline: `dfa17587b2`
3. Never touch master branch
4. Use git tags for stable baselines

**Key Takeaways**:
- ✅ Always use `git ls-remote` to verify commit is fetchable
- ✅ Use branch tips or tags as baselines, never parent commits
- ✅ Keep all changes on feature branches
- ❌ Never push to master branches of forks
- ❌ Never assume commits in history are fetchable

## Fast-Fail Testing Strategy

### Minimal vcpkg.json for Rapid Testing
When debugging vcpkg or baseline issues, use a minimal dependency set to speed up builds (5-10 minutes vs 60 minutes).

**Use Case**: Test libxml2 hash fix and Python distutils without building entire project.

#### Creating Minimal vcpkg.json
```json
{
  "$schema": "https://raw.githubusercontent.com/microsoft/vcpkg/master/scripts/vcpkg.schema.json",
  "name": "apra-pipes-minimal-test",
  "version": "0.0.1",
  "builtin-baseline": "4658624c5f19c1b468b62fe13ed202514dfd463e",
  "dependencies": [
    {
      "name": "glib",
      "default-features": true,
      "platform": "windows"
    }
  ]
}
```

**Why glib?**:
- Depends on libxml2 (tests hash fix)
- Build scripts require Python distutils (tests Python version)
- Small package, builds quickly

#### Helper Scripts
Provided in `base/`:
- `use-minimal-vcpkg.ps1` (Windows)
- `use-minimal-vcpkg.sh` (Linux/Mac)

**Usage**:
```powershell
# Switch to minimal
.\base\use-minimal-vcpkg.ps1

# Run test build
# ... (trigger CI or build locally)

# Restore full vcpkg.json
Copy-Item base\vcpkg.json.full-backup base\vcpkg.json -Force
```

**Benefits**:
- 5-10 minute builds vs 60 minutes
- Faster iteration on vcpkg/baseline fixes
- Saves CI build minutes
- Reduces disk space usage

**When to Use**:
- Testing vcpkg baseline updates
- Verifying libxml2 or other hash fixes
- Testing Python version changes
- Debugging vcpkg registry issues
- Before triggering full Phase 1+2 builds

## Testing Strategy

### Test Execution
- **Binary**: `build/${{buildConf}}/aprapipesut` (Windows) or `build/aprapipesut` (Linux)
- **Framework**: Boost.Test
- **Output Format**: JUnit XML for CI integration

### Test Commands
```bash
# List all test cases
./aprapipesut --list_content > tests.txt

# Run all tests with JUnit output
./aprapipesut --log_format=JUNIT --log_sink=CI_test_result.xml -p -l all
```

### Test Timeouts
- Default: 20 minutes
- Configurable via `nTestTimeoutMins` input parameter

### Test Artifacts
- JUnit XML results uploaded to GitHub Actions
- Test failure screenshots/data from `data/SaveOrCompareFail/`
- Build logs from `vcpkg/buildtrees/**/*.log`

## Triggering Builds

### Manual Trigger (Recommended During Fixes)
```bash
# Trigger a specific workflow
gh workflow run CI-Win-NoCUDA.yml

# Trigger and monitor
gh workflow run CI-Win-NoCUDA.yml
gh run list --workflow=CI-Win-NoCUDA.yml --limit 1
gh run watch <run-id>
```

### Automatic Triggers (Disabled During Active Development)
```yaml
# Typical setup (currently commented out to avoid wasting build minutes)
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
```

## Monitoring and Debugging

### Check Build Status
```bash
# List recent runs
gh run list --limit 10

# View specific run
gh run view <run-id>

# Download logs
gh run view <run-id> --log > build.log

# Download artifacts
gh run download <run-id>
```

### Key Log Files
- **CMake configure**: `vcpkg/buildtrees/<package>/config-x64-windows-*.log`
- **Package build**: `vcpkg/buildtrees/<package>/install-x64-windows-*.log`
- **vcpkg manifest**: `vcpkg_installed/vcpkg/status`

### Common Log Patterns to Search
```bash
# Find errors
grep -i "error" build.log

# Find package failures
grep "error:" build.log | grep "package"

# Find Python-related issues
grep -i "python\|distutils" build.log

# Find hash mismatches
grep "unexpected hash" build.log
```

## Best Practices

### For Development
1. **Use manual triggers**: Disable automatic triggers during active development
2. **Test Phase 1 first**: Ensure caching works before running Phase 2
3. **Monitor regularly**: Catch failures early
4. **Document major issues**: Add new patterns to this guide for future engineers
5. **Use minimal vcpkg**: Temporarily remove packages to isolate issues

### For Dependency Management
1. **Pin major versions**: Use version overrides for all critical dependencies
2. **Test baseline updates in isolation**: Never update baseline directly in production
3. **Capture version snapshots**: Run `vcpkg list` after successful builds
4. **Review changelogs**: Check for breaking changes before upgrading packages
5. **DevOps role clarity**: Fix builds, don't modify application code to accommodate new library versions

### For Maintenance
1. **Update vcpkg quarterly**: Avoid stale baselines causing hash mismatches, but test first
2. **Pin critical tool versions**: Especially Python (must be 3.10.x for distutils)
3. **Test cache invalidation**: Verify new cache keys work as expected
4. **Monitor disk usage**: Public runners have limited space
5. **Review pinned versions**: Check if pinned packages need security updates

### For Troubleshooting
1. **Download full logs**: Don't rely on GitHub Actions UI truncation
2. **Beware continue-on-error**: Check actual error messages, not just step status
3. **Check vcpkg buildtrees**: Most detailed error information
4. **Verify submodules**: Ensure vcpkg commit is fetchable with `git ls-remote`
5. **Test locally**: Reproduce issues on similar environment when possible
6. **Think logically**: Before adding PATH fixes, verify the tool actually needs to be in PATH

## Future Improvements

### Phase 1 Optimization (Proposed)
Replace CMake configure with direct vcpkg install to eliminate `continue-on-error`:

```yaml
- name: Install OpenCV to cache (Phase 1)
  run: |
    .\base\fix-vcpkg-json.ps1 -onlyOpenCV
    .\vcpkg\vcpkg.exe install --triplet x64-windows
  # No continue-on-error - fails cleanly on real errors
```

**Benefits:**
- Real errors fail immediately (no hidden failures)
- Cleaner separation: Phase 1 = vcpkg cache, Phase 2 = CMake build
- Easier to debug vcpkg-specific issues
- Still populates cache for Phase 2

### Potential Optimizations
1. **Parallel package builds**: vcpkg can build independent packages concurrently
2. **Ccache integration**: Cache compiled objects for faster rebuilds
3. **Incremental builds**: Reuse previous build artifacts when possible
4. **Split test execution**: Run fast tests first, slow tests in parallel jobs

## GitOps Workflow

### Branch Strategy
- `main`: Production-ready code
- `fix/*`: Bug fixes and CI improvements
- Feature branches for new functionality

### CI Fix Workflow
1. Create branch from main: `fix/ci-<issue>-<initials>`
2. Disable automatic triggers on fix branch
3. Make incremental changes
4. Manually trigger builds
5. Monitor and iterate
6. Document attempts in `ci-fix-log.md`
7. Once stable, create PR to main
8. Re-enable automatic triggers after merge

### Commit Messages
Follow conventional commits:
```
ci: Update vcpkg baseline to fix libxml2 hash
fix: Downgrade Python to 3.10.11 for distutils
docs: Add phase 1 optimization plan
```

## Reference Files

### Critical Configuration Files
- `.github/workflows/*.yml`: Workflow definitions
- `base/vcpkg.json`: Dependency manifest
- `vcpkg/baseline.json`: Package version pins
- `vcpkg/scripts/vcpkg-tools.json`: Tool versions
- `base/fix-vcpkg-json.ps1`: vcpkg.json modification script
- `CMakeLists.txt`: Build system configuration

### Documentation Files
- `README.md`: Project overview and setup
- `devops-build-system-guide.md`: This document - comprehensive build system knowledge

## Appendix: Common Commands

### vcpkg Commands
```bash
# Bootstrap vcpkg
./vcpkg/bootstrap-vcpkg.bat  # Windows
./vcpkg/bootstrap-vcpkg.sh   # Linux

# Install packages
./vcpkg/vcpkg install --triplet x64-windows

# List installed packages
./vcpkg/vcpkg list

# Update vcpkg
cd vcpkg
git pull
./bootstrap-vcpkg.bat
```

### CMake Commands
```bash
# Configure
cmake -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build build --config Release -j 6

# Test
cd build
ctest -C Release -V
```

### GitHub CLI Commands
```bash
# Workflow operations
gh workflow list
gh workflow run <name>
gh workflow enable/disable <name>

# Run operations
gh run list
gh run view <id>
gh run watch <id>
gh run download <id>
gh run cancel <id>

# Check PR status
gh pr status
gh pr checks
```
