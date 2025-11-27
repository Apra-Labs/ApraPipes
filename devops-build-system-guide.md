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
- **Baseline**: `vcpkg/baseline.json` pins package versions
- **Toolchain**: `vcpkg/scripts/buildsystems/vcpkg.cmake` integrates with CMake

### vcpkg Tools Configuration
- **File**: `vcpkg/scripts/vcpkg-tools.json`
- **Purpose**: Defines which versions of Python, CMake, etc. vcpkg downloads
- **Critical**: Python version must be 3.10.x (has distutils) not 3.12+ (distutils removed)

### Key Dependencies
- **OpenCV**: Largest dependency, includes CUDA/cuDNN features
- **glib**: Requires Python with distutils for build
- **FFmpeg**: Video processing
- **Boost**: C++ utilities
- **libxml2**: Previously had hash mismatch issues (fixed by vcpkg update)

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
3. **Monitor every 5 minutes**: Catch failures early
4. **Keep fix log**: Document each attempt and findings (see `ci-fix-log.md`)
5. **Use minimal vcpkg**: Temporarily remove packages to isolate issues

### For Maintenance
1. **Update vcpkg regularly**: Avoid stale baselines causing hash mismatches
2. **Pin critical tool versions**: Especially Python (must be 3.10.x)
3. **Test cache invalidation**: Verify new cache keys work as expected
4. **Monitor disk usage**: Public runners have limited space

### For Troubleshooting
1. **Download full logs**: Don't rely on GitHub Actions UI truncation
2. **Check vcpkg buildtrees**: Most detailed error information
3. **Verify submodules**: Ensure vcpkg commit is fetchable
4. **Test locally**: Reproduce issues on similar environment when possible

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
- `ci-fix-log.md`: Build fix attempt history
- `phase1-optimization-plan.md`: Phase 1 improvement proposals
- `devops-build-system-guide.md`: This document

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
