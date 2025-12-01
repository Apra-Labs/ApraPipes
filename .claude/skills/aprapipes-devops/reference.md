# ApraPipes DevOps Reference

Cross-platform reference for vcpkg configuration, GitHub Actions, caching, and version management.

---

## Architecture Overview

### Build Strategies by Platform

| Platform | Runner Type | Build Strategy | Time Limit | Caching |
|----------|-------------|----------------|------------|---------|
| Windows NoCUDA | GitHub-hosted | Two-phase | 1 hour/phase | Yes |
| Windows CUDA | Self-hosted | Single-phase | None | No |
| Linux x64 NoCUDA | GitHub-hosted | Two-phase | 1 hour/phase | Yes |
| Linux x64 CUDA | Self-hosted | Single-phase | None | No |
| Jetson ARM64 | Self-hosted | Single-phase | None | No |
| Docker | Varies | Varies | Varies | Optional |

### Two-Phase Build Strategy (Hosted Runners Only)

**Phase 1: Prep/Cache**
- Goal: Install heavy dependencies (especially OpenCV) and cache them
- Trigger: Manual (`workflow_dispatch`)
- Process:
  1. Modify vcpkg.json to only include OpenCV (`fix-vcpkg-json.ps1 -onlyOpenCV`)
  2. Run CMake configure to trigger vcpkg installation
  3. Cache vcpkg archives to GitHub Actions cache
  4. Note: Uses `continue-on-error: true` (can hide real failures)

**Phase 2: Build/Test**
- Goal: Full build and test execution
- Trigger: Manual (`workflow_dispatch`)
- Process:
  1. Restore vcpkg cache from Phase 1
  2. Run full CMake configure with all dependencies
  3. Build the project
  4. Run unit tests
  5. Upload test results and logs

### Single-Phase Build (Self-Hosted Runners)
- No caching needed (persistent disk)
- Direct build from checkout to test
- Used for CUDA builds (require specialized hardware)
- No time limits, no disk constraints

---

## vcpkg Configuration

### File Structure

```
base/
├── vcpkg.json                 # Dependency manifest
├── vcpkg-configuration.json   # Registry and baseline
└── fix-vcpkg-json.ps1        # Runtime manifest modifier

vcpkg/
├── scripts/
│   └── vcpkg-tools.json      # Tool versions (Python, CMake)
└── scripts/buildsystems/
    └── vcpkg.cmake           # CMake toolchain file
```

### vcpkg.json - Dependency Manifest

**Location**: `base/vcpkg.json`

**Structure**:
```json
{
  "$schema": "https://raw.githubusercontent.com/microsoft/vcpkg/master/scripts/vcpkg.schema.json",
  "name": "apra-pipes-cuda",
  "version": "0.0.1",
  "builtin-baseline": "4658624c5f19c1b468b62fe13ed202514dfd463e",

  "overrides": [
    { "name": "ffmpeg", "version": "4.4.3" },
    { "name": "libarchive", "version": "3.5.2" },
    { "name": "sfml", "version": "2.6.2" }
  ],

  "dependencies": [
    "pkgconf",
    { "name": "opencv4", "features": ["contrib", "cuda", ...] },
    "boost-system",
    // ... more dependencies
  ]
}
```

**Key Sections**:
- `builtin-baseline`: Not used (overridden by vcpkg-configuration.json)
- `overrides`: Version pins for specific packages (CRITICAL for stability)
- `dependencies`: List of required packages with optional features/platforms

### vcpkg-configuration.json - Registry and Baseline

**Location**: `base/vcpkg-configuration.json`

**Structure**:
```json
{
  "$schema": "https://raw.githubusercontent.com/microsoft/vcpkg-tool/main/docs/vcpkg-configuration.schema.json",

  "overlay-ports": [
    "../thirdparty/custom-overlay"
  ],

  "default-registry": {
    "kind": "git",
    "repository": "https://github.com/Apra-Labs/vcpkg.git",
    "baseline": "3011303ba1f6586e8558a312d0543271fca072c6"
  }
}
```

**CRITICAL**: The baseline commit MUST be:
- Advertised by `git ls-remote` (branch tip or tag)
- From a repository with version database (`versions/` directory)
- Accessible without authentication

### vcpkg-tools.json - Tool Versions

**Location**: `vcpkg/scripts/vcpkg-tools.json`

**CRITICAL Python Version**:
```json
{
  "name": "python3",
  "os": "windows",
  "version": "3.10.11",  // MUST be 3.10.x (has distutils)
  "executable": "python.exe",
  "url": "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip",
  "sha512": "40cbc98137cc7768e3ea498920ddffd0b3b30308bfd7bbab2ed19d93d2e89db6b4430c7b54a0f17a594e8e10599537a643072e08cfd1a38c284f8703879dcc17"
}
```

**Why 3.10.x**: Python 3.12+ removed `distutils` module required by glib and other packages.

---

## Version Pinning Strategy

### Core Principle
**Pin major versions of all production dependencies to avoid breaking changes.**

### Why Version Pinning Matters

When updating vcpkg baseline WITHOUT version overrides:
- ❌ Any package can jump to new major version
- ❌ Breaking API changes silently break builds
- ❌ Errors appear unrelated to dependency update
- ❌ "Worked yesterday" syndrome - non-reproducible builds

### Current Pinned Versions (as of 2024-11-28)

```json
"overrides": [
  { "name": "ffmpeg", "version": "4.4.3" },      // Pinned to 4.x
  { "name": "libarchive", "version": "3.5.2" },  // Pinned to 3.x
  { "name": "sfml", "version": "2.6.2" }         // Pinned to 2.x (SFML 3.x has breaking changes)
]
```

### Version Pinning Tiers

**Tier 1 - MUST Pin** (Critical Dependencies):
- Packages with complex APIs your code directly uses
- Libraries with history of breaking changes between major versions
- Current: opencv4, boost, gtk3, sfml, ffmpeg

**Tier 2 - Consider Pinning** (Medium Risk):
- Packages used in specific modules
- Rapidly evolving libraries (AI/ML)
- Current: whisper, nu-book-zxing-cpp

**Tier 3 - Can Use Baseline** (Low Risk):
- Utilities that rarely have breaking API changes
- Updated primarily for security/bug fixes
- Current: pkgconf, zlib, bzip2, liblzma, brotli

### When to Update Pinned Versions

**Process**:
1. Create dedicated branch: `update/opencv-4.8-to-4.9`
2. Update version override in vcpkg.json
3. Review upstream changelog for breaking changes
4. Test build thoroughly
5. If code changes needed → escalate to developers
6. Document migration in commit message
7. Merge only after verification

**DevOps Role**: Fix build config, NOT application code to accommodate new versions.

---

## GitHub Actions Reference

### Workflow Structure

**Top-Level Workflows** (`.github/workflows/`):
- `CI-Win-NoCUDA.yml`
- `CI-Win-CUDA.yml`
- `CI-Linux-x64-NoCUDA.yml`
- `CI-Linux-x64-CUDA.yml`
- `CI-Linux-ARM64.yml`
- `CI-Jetson.yml` (if exists)

**Reusable Workflows**:
- `build-test-win.yml` - Parameterized Windows builds
- `build-test-linux.yml` - Parameterized Linux builds

### Key Workflow Inputs

| Input | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `runner` | string | GitHub runner to use | `windows-latest`, `ubuntu-latest`, `self-hosted` |
| `flav` | string | Build flavor | `Win-nocuda`, `Linux-x64-cuda` |
| `buildConf` | string | Build configuration | `Release`, `Debug` |
| `is-selfhosted` | boolean | Skip caching if true | `true`, `false` |
| `is-prep-phase` | boolean | Phase 1 vs Phase 2 | `true`, `false` |
| `cuda` | string | Enable CUDA | `ON`, `OFF` |
| `prep-cmd` | string | Commands to prep builder | Install tools |
| `cmake-conf-cmd` | string | CMake configuration | CMake flags |

### Workflow Triggers

**Manual Trigger** (recommended during fixes):
```yaml
on:
  workflow_dispatch:
```

**Automatic Triggers** (disabled during development):
```yaml
# on:
#   push:
#     branches: [ main ]
#   pull_request:
#     branches: [ main ]
```

### CI Best Practices - Non-Regression Testing

**CRITICAL RULE**: When making build system or devops changes, always validate that you don't regress other platforms.

**Minimum Required Validation**:
- For vcpkg/dependency changes: Test Windows NoCUDA + Linux NoCUDA (minimum 2 platforms)
- For workflow changes: Test affected workflow only
- For code changes: Test all relevant platforms (Windows + Linux at minimum)

**Cost Management**:
- **During development**: Disable automatic triggers (`workflow_dispatch` only)
- **Test selectively**: Only trigger workflows you need to validate
- **Cancel wasteful runs**: If auto-triggers fire accidentally, immediately cancel unnecessary builds
- **Monitor actively**: Check builds every 5-10 minutes, don't leave builds running unnecessarily

**Example Workflow**:
1. Make vcpkg change (e.g., add dependency, update baseline)
2. Manually trigger Windows NoCUDA build
3. Monitor and fix any errors
4. Manually trigger Linux NoCUDA build to validate cross-platform
5. If Linux fails, fix and re-trigger ONLY Linux (not all 7 workflows)
6. Once both pass, reinstate automatic triggers and merge

**Anti-Pattern** (wasteful):
```
✗ Reinstating auto-triggers before validating
✗ Letting all 7 workflows run when only testing 1 platform
✗ Not canceling unnecessary builds
✗ Triggering builds without monitoring them
```

**Efficient Pattern**:
```
✓ workflow_dispatch only during development
✓ Test minimum required platforms
✓ Cancel unnecessary auto-triggered builds immediately
✓ Monitor builds actively, fix quickly
✓ Reinstate auto-triggers only after validation
```

---

## Cache Configuration

### Cache Paths by Platform

| Platform | Cache Path | Size Typical |
|----------|------------|--------------|
| Windows | `C:\Users\runneradmin\AppData\Local\vcpkg\archives` | 5-10 GB |
| Linux | `~/.cache/vcpkg` or `${HOME}/.cache/vcpkg` | 5-10 GB |
| Self-hosted | Not cached (persistent disk) | N/A |

### Cache Key Structure

**Current** (v5):
```yaml
key: ${{ inputs.flav }}-5-${{ hashFiles('base/vcpkg.json', 'base/vcpkg-configuration.json', 'submodule_ver.txt') }}
restore-keys: ${{ inputs.flav }}-5-
```

**Components**:
- `flav`: Platform identifier (Win-nocuda, Linux-x64-cuda, etc.)
- `5`: Cache version (bump to invalidate all caches)
- `hashFiles(...)`: Hash of files that affect dependencies

**Files Hashed**:
- `base/vcpkg.json`: Changes when dependencies added/removed
- `base/vcpkg-configuration.json`: Changes when baseline updated
- `submodule_ver.txt`: Changes when vcpkg submodule updated

### Cache Invalidation

**When to Bump Cache Version**:
1. vcpkg-tools.json changed (e.g., Python version downgrade)
2. Major vcpkg baseline update
3. Cache corruption suspected
4. Force rebuild of all packages

**How to Bump**:
Change `5` to `6` in cache key definition in `build-test-win.yml` or `build-test-linux.yml`.

### Cache Behavior

| Scenario | Result |
|----------|--------|
| Exact key match | Cache hit - fast restore |
| Prefix match (restore-keys) | Partial hit - some packages cached |
| No match | Cache miss - full build |
| Phase 1 saves, Phase 2 restores | Expected flow |
| Phase 2 before Phase 1 | Cache miss (run Phase 1 first) |

### Cache Optimization Tip - Preserving Cache During Development

**Problem**: Cache invalidates on every vcpkg.json change, causing hours of rebuilding the same packages repeatedly.

**Solution**: During iterative development, remove `hashFiles()` from cache key to preserve cache across dependency changes.

**Standard Cache Key** (production):
```yaml
key: ${{ inputs.flav }}-5-${{ hashFiles('base/vcpkg.json', 'base/vcpkg-configuration.json', 'submodule_ver.txt') }}
restore-keys: ${{ inputs.flav }}-5-
```

**Optimized Cache Key** (development):
```yaml
key: ${{ inputs.flav }}-5
restore-keys: ${{ inputs.flav }}-
```

**Benefits**:
- Cache persists across vcpkg.json modifications
- Adding/removing dependencies doesn't trigger full rebuild
- Packages already built remain cached
- Only new/changed packages rebuild

**Trade-offs**:
- Cache doesn't auto-invalidate on dependency changes
- Must manually bump version number (5 → 6) when needed
- Can accumulate stale packages over time

**When to Use**:
- ✓ During active development with frequent vcpkg.json changes
- ✓ When iteratively fixing build issues across platforms
- ✓ When adding multiple dependencies in sequence

**When NOT to Use**:
- ✗ On main branch (use hash-based invalidation)
- ✗ When cache corruption suspected (bump version instead)
- ✗ For production builds (want deterministic cache behavior)

**Applies to**: All workflows with caching (Windows NoCUDA, Linux NoCUDA, etc.)

---

## Linking Static Dependencies

### vcpkg.json vs CMakeLists.txt

**Two separate requirements for using a package:**

| Action | File | Purpose |
|--------|------|---------|
| Install package | `vcpkg.json` | Makes vcpkg download and build the library |
| Link package | `CMakeLists.txt` | Tells linker to include library in executable |

**Both are required.** Installing alone doesn't link it.

### Transitive Dependencies in Static Builds

**Problem**: vcpkg uses static libraries (`.a`). Static libs don't carry dependency info.

**Example**:
```
Your executable → sfml-audio → libopenal.a → needs fmt
```

**What happens**:
- OpenAL is compiled with fmt and has references to `fmt::vformat`
- When linking statically, linker doesn't know OpenAL needs fmt
- Result: `undefined reference to fmt::vformat`

**Solution**: Explicitly link ALL transitive dependencies in CMakeLists.txt:

```cmake
find_package(SFML REQUIRED)
find_package(fmt CONFIG REQUIRED)  # Even if you don't use fmt directly

target_link_libraries(myapp
  sfml-audio    # Direct dependency
  fmt::fmt      # Transitive dependency (required by OpenAL, used by SFML)
)
```

### Debugging Transitive Dependency Errors

**Symptoms**: `undefined reference` errors during linking

**Diagnosis steps**:
1. Look at undefined symbol: `undefined reference to 'fmt::v12::vformat'`
2. Check which `.a` file references it: `libopenal.a(alc.cpp.o)`
3. Trace dependency chain: executable → sfml-audio → openal → fmt
4. Add missing library to `target_link_libraries`

**Quick check**:
```bash
# Find what libraries reference missing symbol
grep -r "undefined reference" build.log
# Example: /usr/bin/ld: vcpkg_installed/x64-linux/lib/libopenal.a(alc.cpp.o)
```

### Target-Specific Linking

**Common mistake**: Adding library to wrong target

```cmake
# WRONG - linking to 'main' when 'aprapipesut' needs it
target_link_libraries(main PRIVATE fmt::fmt)

# RIGHT - link to the target that actually uses it
target_link_libraries(aprapipesut PRIVATE fmt::fmt)
```

**Rule**: Link libraries to the specific executable/library target that uses them.

---

## Dependency Matrix

### Core Dependencies (All Platforms)

| Package | Purpose | Pinned? | Version |
|---------|---------|---------|---------|
| pkgconf | CMake FindPkgConfig support | No | Latest |
| opencv4 | Computer vision (CUDA features) | No | Latest (4.x) |
| boost-* | C++ utilities | No | Latest (1.x) |
| ffmpeg | Video processing | **Yes** | 4.4.3 |
| sfml | Audio/graphics | **Yes** | 2.6.2 |
| libjpeg-turbo | Image codec | No | Latest |
| zlib, bzip2, liblzma | Compression | No | Latest |

### Platform-Specific Dependencies

**Windows**:
```json
{ "name": "glib", "platform": "windows" }
```

**Linux x64**:
```json
{
  "name": "glib",
  "features": ["libmount"],
  "platform": "(linux & x64)"
}
```

**Excluded on ARM64**:
```json
{ "name": "hiredis", "platform": "!arm64" }
{ "name": "redis-plus-plus", "platform": "!arm64" }
```

**Excluded on Windows**:
```json
{ "name": "gtk3", "platform": "!windows" }
{ "name": "re", "platform": "!windows" }
{ "name": "baresip", "platform": "!windows" }
```

### CUDA-Specific Features

**OpenCV4 with CUDA**:
```json
{
  "name": "opencv4",
  "features": ["contrib", "cuda", "cudnn", "dnn", ...]
}
```

**Whisper with CUDA**:
```json
{
  "name": "whisper",
  "features": ["cuda"]
}
```

**Removed for NoCUDA builds**: `fix-vcpkg-json.ps1 -removeCUDA` strips CUDA features.

---

## vcpkg Fork Management

### Repository Structure

- **Upstream**: `https://github.com/microsoft/vcpkg.git`
- **Fork**: `https://github.com/Apra-Labs/vcpkg.git`
- **Custom Overlay**: `thirdparty/custom-overlay/` (for small modifications)

### CRITICAL: Never Modify Master Branches

**❌ NEVER**:
```bash
git checkout master
git merge fix/my-changes
git push origin master  # BREAKS OTHER BUILDS!
```

**✅ ALWAYS**:
```bash
git checkout -b fix/vcpkg-update-2024-11-28
git push origin fix/vcpkg-update-2024-11-28
# Use feature branch HEAD as baseline in vcpkg-configuration.json
```

### Baseline Commit Requirements

A valid baseline commit MUST:
1. Be advertised by `git ls-remote` (branch tip or tag)
2. Exist in repository with version database (`versions/` directory)
3. Be accessible without authentication

**Verify Before Using**:
```bash
git ls-remote https://github.com/Apra-Labs/vcpkg.git | grep <commit-hash>
```

**Common Mistake**: Using parent commit (not advertised) → "not our ref" error

---

## File Locations Matrix

| File/Directory | Windows (Hosted) | Linux (Hosted) | Self-Hosted |
|----------------|------------------|----------------|-------------|
| Workspace | `D:\a\ApraPipes\ApraPipes` | `/home/runner/work/ApraPipes` | Varies |
| vcpkg cache | `C:\Users\runneradmin\AppData\Local\vcpkg\archives` | `~/.cache/vcpkg` | Not cached |
| Build dir | `D:\a\ApraPipes\ApraPipes\build` | `/home/runner/work/ApraPipes/ApraPipes/build` | `./build` |
| vcpkg installed | `{workspace}/vcpkg_installed` | Same | Same |
| vcpkg buildtrees (logs) | `{workspace}/vcpkg/buildtrees` | Same | Same |
| Temp | `D:\a\_temp` | `/tmp` | Varies |

---

## Common Commands Quick Reference

### GitHub CLI

```bash
# Workflow operations
gh workflow list
gh workflow run <workflow-name> --ref <branch>
gh workflow enable/disable <workflow-name>

# Run operations
gh run list --workflow=<workflow-name> --limit 10
gh run view <run-id>
gh run view <run-id> --log > build.log
gh run watch <run-id>
gh run cancel <run-id>
gh run download <run-id>  # Download artifacts
```

### vcpkg

```bash
# Bootstrap
./vcpkg/bootstrap-vcpkg.bat  # Windows
./vcpkg/bootstrap-vcpkg.sh   # Linux

# Package operations
./vcpkg/vcpkg install --triplet x64-windows
./vcpkg/vcpkg list
./vcpkg/vcpkg remove <package>
./vcpkg/vcpkg update

# Diagnostics
cat vcpkg_installed/vcpkg/status  # Installed packages
ls vcpkg/buildtrees/  # Build logs by package
```

### CMake

```bash
# Configure
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=OFF

# Build
cmake --build build --config Release -j 6

# Test
cd build
ctest -C Release -V
```

### Log Analysis

```bash
# Download logs
gh run view <run-id> --log > /tmp/build.log

# Find errors
grep -i "error:" /tmp/build.log
grep "CMake Error" /tmp/build.log
grep "failed with" /tmp/build.log

# Find specific issues
grep -i "distutils\|python" /tmp/build.log
grep "unexpected hash" /tmp/build.log
grep "PKG_CONFIG" /tmp/build.log
grep "not our ref" /tmp/build.log

# Check Phase 1 vs Phase 2
grep "win-nocuda-build-prep" /tmp/build.log  # Phase 1
grep "win-nocuda-build-test" /tmp/build.log  # Phase 2
```

---

## Environment Variables

### Common Across Platforms

| Variable | Purpose | Example Value |
|----------|---------|---------------|
| `CMAKE_TOOLCHAIN_FILE` | vcpkg CMake integration | `{workspace}/vcpkg/scripts/buildsystems/vcpkg.cmake` |
| `VCPKG_ROOT` | vcpkg installation | `{workspace}/vcpkg` |
| `CMAKE_BUILD_TYPE` | Build configuration | `Release`, `Debug` |

### Windows-Specific

| Variable | Purpose | Example Value |
|----------|---------|---------------|
| `CUDA_HOME` | CUDA toolkit location | `c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8` |
| `CUDA_PATH` | CUDA bin directory | `{CUDA_HOME}\bin` |

### Linux-Specific

| Variable | Purpose | Example Value |
|----------|---------|---------------|
| `PKG_CONFIG_PATH` | pkg-config search path | `/usr/lib/pkgconfig` |

### Jetson-Specific

| Variable | Purpose | Example Value |
|----------|---------|---------------|
| `VCPKG_FORCE_SYSTEM_BINARIES` | Force using system binaries on ARM64 | `1` |

---

## Version Information

**Last Updated**: 2024-11-28
**vcpkg Baseline**: `3011303ba1f6586e8558a312d0543271fca072c6`
**Python Version**: 3.10.11
**CMake Version**: 3.29.6
**Cache Version**: 5
