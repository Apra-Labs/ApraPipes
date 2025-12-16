# vcpkg Troubleshooting Guide

## vcpkg Cache Management

### Issue: GitHub Actions Cache Not Saving on Build Failures

**Symptom**:
- Build fails during CMake configure or compilation
- Next CI run starts vcpkg installation from scratch
- Takes 1-2 hours to rebuild all dependencies every iteration
- Debugging becomes impractical due to long feedback cycles

**Root Cause**:
Traditional GitHub Actions cache patterns only save on success:

```yaml
# ❌ WRONG: Only saves on successful completion
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/vcpkg/archives
    key: vcpkg-${{ hashFiles('vcpkg.json') }}
```

If the build fails after vcpkg installs some packages, those packages are NOT cached.

**Solution: Split Restore and Save with always() Condition**

The key insight: **GitHub Actions automatically replaces the old cache when you save with the same key**. No need to manually delete!

```yaml
# Step 1: Restore previous cache (if exists)
- name: Restore vcpkg cache
  id: cache-restore
  if: ${{ !inputs.is-selfhosted }}
  uses: actions/cache/restore@v3
  with:
      path: |
        ${{ inputs.cache-path }}
      key: ${{ inputs.flav }}-5-${{ hashFiles('base/vcpkg.json', 'base/vcpkg-configuration.json', 'submodule_ver.txt') }}
      restore-keys: ${{ inputs.flav }}-

# Step 2: Run CMake configure (vcpkg install happens here)
- name: Configure CMake Common
  working-directory: ${{github.workspace}}/build
  run: 'cmake -B . ...'

# Step 3: ALWAYS save cache after configure (even if build fails later)
- name: Save vcpkg cache (even if build fails later)
  if: ${{ always() && !inputs.is-selfhosted }}
  uses: actions/cache/save@v3
  with:
    path: |
      ${{ inputs.cache-path }}
    key: ${{ inputs.flav }}-5-${{ hashFiles('base/vcpkg.json', 'base/vcpkg-configuration.json', 'submodule_ver.txt') }}
```

**How It Works**:
1. **Restore**: Tries to restore cache with exact key, falls back to prefix match
2. **Build**: CMake runs vcpkg install, building any missing packages
3. **Save**: ALWAYS saves current state, even if build fails later
4. **Replace**: GitHub Actions automatically replaces old cache (same key)
5. **Next run**: Starts with more packages cached than before

**Benefits**:
- **First build**: Downloads ~50+ packages, takes ~2 hours
- **Second build** (after failure): Reuses cached packages, only builds new ones
- **Incremental progress**: Each run adds more packages to cache
- **Practical debugging**: Iterations take minutes instead of hours

**Cache Key Strategy**:
```yaml
key: ${{ inputs.flav }}-5-${{ hashFiles('base/vcpkg.json', 'base/vcpkg-configuration.json', 'submodule_ver.txt') }}
restore-keys: ${{ inputs.flav }}-
```

- **Exact match**: Same vcpkg.json + baseline + submodule versions
- **Prefix match**: Same platform flavor (MacOSX, Linux-ARM64, etc.)
- **Version bump**: `-5-` allows manual cache invalidation if needed

**Important Notes**:
- Don't use `cache-hit != 'true'` checks - always save after configure
- Don't use combined `actions/cache@v3` - split into restore/save
- Don't try to manually delete cache with `gh cache delete` - GitHub auto-replaces
- Only skip caching for self-hosted runners (they have persistent storage)

**References**:
- Commit: 58e25a2075900a7c0bb8acdf44e669f00fec5052
- Applied in: build-test-macosx.yml, build-test-lin-container.yml

---

## vcpkg Package Version Pinning

### Issue: Specific Package Version Needed Across All Platforms

**Use Case**: Need to pin one package to an older version while keeping others current

**Example**: libjpeg-turbo 3.1.2 breaks on macOS CI, need 3.0.4 specifically

**Solution: vcpkg-configuration.json with Custom Registry**

Create `vcpkg-configuration.json` at repository root:

```json
{
  "$schema": "https://raw.githubusercontent.com/microsoft/vcpkg-tool/main/docs/vcpkg-configuration.schema.json",
  "default-registry": {
    "kind": "git",
    "repository": "https://github.com/microsoft/vcpkg",
    "baseline": "3011303ba1f6586e8558a312d0543271fca072c6"
  },
  "registries": [
    {
      "kind": "git",
      "repository": "https://github.com/microsoft/vcpkg",
      "baseline": "20d1b778772da35c42c7729be82ad8dcf40b1e88",
      "packages": [
        "libjpeg-turbo"
      ]
    }
  ]
}
```

**How It Works**:
- **Default registry**: All packages use this baseline (most recent stable)
- **Custom registry**: Specific packages override with different baseline
- **Package list**: Only listed packages use the custom registry
- **Result**: libjpeg-turbo uses old baseline, everything else uses new baseline

**Finding the Right Baseline**:

```bash
# 1. Clone vcpkg repository
git clone https://github.com/microsoft/vcpkg.git /tmp/vcpkg-research
cd /tmp/vcpkg-research

# 2. Find commits that touched the package port
git log --all --oneline -- ports/libjpeg-turbo/vcpkg.json

# 3. Check version in specific commit
git show <commit-hash>:ports/libjpeg-turbo/vcpkg.json | grep version

# 4. Find the baseline commit (use full commit hash)
git log --all --oneline | grep <short-hash>

# 5. Verify the baseline is in vcpkg's versions database
git log --all --oneline -- versions/baseline.json | grep <commit-hash>
```

**Alternative: Verify via vcpkg versions**:
```bash
# Check available versions
cd /tmp/vcpkg-research
./vcpkg search libjpeg-turbo --x-full-desc

# Find commits with specific version
git log --all --format="%H %s" -- versions/l-/libjpeg-turbo.json | head -20
```

**Common Gotchas**:
- Baseline must be a full commit hash (not short hash)
- Baseline must exist in vcpkg's versions database
- Package name must match exactly (case-sensitive)
- Changes require vcpkg cache invalidation (bump cache key version)

**Testing**:
```bash
# Verify vcpkg recognizes the configuration
./vcpkg/vcpkg install libjpeg-turbo --dry-run

# Check which version will be installed
./vcpkg/vcpkg list | grep libjpeg

# Force clean reinstall to test
rm -rf vcpkg_installed build
cmake -B build ...
```

**References**:
- Commit: ee2455c3b126d4d186c95967e31a41bb14d7570e
- vcpkg docs: https://learn.microsoft.com/en-us/vcpkg/users/versioning

---

## vcpkg Baseline Updates

### Issue: Need to Update to Newer vcpkg Packages

**When to Update**:
- Security vulnerabilities in dependencies
- Need newer features from a package
- Compatibility with newer platform (macOS 15, Ubuntu 24.04, etc.)
- Bug fixes in vcpkg ports

**Risks**:
- Breaking changes in package APIs
- Build failures on specific platforms
- Test regressions
- Incompatibilities between packages at different versions

**Safe Update Process**:

```bash
# 1. Check current baseline
cat vcpkg-configuration.json | grep baseline

# 2. Find latest vcpkg commit
cd vcpkg
git fetch origin
git log --oneline origin/master | head -5

# 3. Test with specific baseline first
# Update vcpkg-configuration.json default-registry baseline to new commit

# 4. Use Fast-Fail Testing (see below)

# 5. If successful, update all platforms

# 6. Bump cache version to invalidate old caches
# Change key from "MacOSX-5-" to "MacOSX-6-" in workflow files
```

**Fast-Fail Testing Strategy**:

Instead of waiting 60+ minutes for full Phase 1+2 builds, use minimal vcpkg.json:

```json
{
  "$schema": "https://raw.githubusercontent.com/microsoft/vcpkg/master/scripts/vcpkg.schema.json",
  "name": "apra-pipes-minimal-test",
  "version": "0.0.1",
  "builtin-baseline": "NEW_BASELINE_COMMIT_HERE",
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
- Depends on libxml2 (tests hash fixes)
- Build scripts require Python distutils (tests Python version)
- Small package, builds in 5-10 minutes vs 60+ minutes

**Fast-Fail Workflow**:
1. Create minimal vcpkg.json in test branch
2. Trigger workflow with `gh workflow run ...`
3. Wait 5-10 minutes instead of 60+ minutes
4. If it fails, fix and iterate quickly
5. Once passing, restore full vcpkg.json
6. Run full build to verify

**Time Savings**: 5-10 minute builds vs 60+ minutes = 50+ minutes saved per iteration

**What to Test**:
- Library hash verification (libxml2 in glib dependency)
- Python version compatibility (distutils in glib build)
- vcpkg registry connectivity
- Baseline validity

**References**:
- Documented in methodology.md lines 102-137

---

## vcpkg Binary Cache Configuration

### Standard Configuration

```yaml
env:
  VCPKG_MAX_CONCURRENCY: ${{ inputs.nProc }}
  VCPKG_DEFAULT_BINARY_CACHE: ${{ github.workspace }}/../.cache/vcpkg
```

**Directory Structure**:
```
workspace/
├── ApraPipes/           (git checkout)
├── .cache/
│   └── vcpkg/
│       └── archives/     (binary packages cached here)
```

**Cache Path Variations by Platform**:
- **macOS**: `~/.cache/vcpkg/archives`
- **Linux**: `~/.cache/vcpkg/archives`
- **Windows**: `%LOCALAPPDATA%\vcpkg\archives`

### Cache Size Management

**Typical sizes**:
- Empty cache: 0 MB
- After Phase 1 (minimal deps): ~500 MB
- After Phase 2 (full deps): ~2-3 GB
- Full build (all platforms): ~10+ GB total

**GitHub Actions Cache Limits**:
- Total cache size per repository: 10 GB
- Least-recently-used caches evicted automatically
- Each platform flavor should have separate cache key

**Monitoring Cache Size**:
```bash
# List all caches for repository
gh cache list --limit 50

# Check specific cache size
gh cache list | grep MacOSX

# Delete old caches manually (if needed)
gh cache delete <cache-key>
```

**Best Practices**:
- Use separate cache keys per platform (`MacOSX-5-`, `Linux-ARM64-5-`)
- Include hash of vcpkg.json + baseline + submodules in key
- Use version prefix (`-5-`) for manual invalidation
- Self-hosted runners should skip caching (persistent disk)

---

## vcpkg Installation Failures

### Network Timeouts

**Symptom**:
```
Failed to download <package> from https://github.com/...
curl: (28) Operation timed out
```

**Solutions**:
1. Retry the build (transient network issues)
2. Check GitHub API rate limits: `curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit`
3. Use vcpkg binary cache to avoid re-downloads
4. For persistent issues, mirror the package source

### Package Hash Mismatches

**Symptom**:
```
Expected hash: abc123...
Actual hash:   def456...
```

**Causes**:
- Upstream package source changed
- vcpkg port updated but not baseline
- Network corruption during download

**Solutions**:
1. Update to newer vcpkg baseline (may have fix)
2. Verify package source URL is still valid
3. Report to vcpkg repository if reproducible
4. Temporarily pin to older baseline with known-good hash

### Missing Platform Support

**Symptom**:
```
Error: Package <name> does not support platform x64-osx
```

**Solutions**:
1. Check if package truly unsupported on platform
2. Update to newer vcpkg baseline (may have added support)
3. Use overlay ports to add platform support
4. Exclude package from vcpkg.json for that platform:
   ```json
   {
     "name": "package-name",
     "platform": "!osx"
   }
   ```

---

## vcpkg Custom Overlays

### When to Use Overlays

**Use overlay ports when**:
- Need to patch a vcpkg package for compatibility
- Package has platform-specific build issues
- Waiting for upstream vcpkg fix to be merged
- Testing patches before contributing to vcpkg

### Overlay Structure

```
thirdparty/custom-overlay/
├── libmp4/
│   ├── portfile.cmake
│   ├── vcpkg.json
│   └── patches/
│       └── fix-macos-build.patch
```

### Configuring Overlays in CMake

```cmake
# In vcpkg triplet or toolchain configuration
set(VCPKG_OVERLAY_PORTS "${CMAKE_SOURCE_DIR}/thirdparty/custom-overlay")
```

Or via command line:
```bash
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_OVERLAY_PORTS=thirdparty/custom-overlay \
  ...
```

### Updating Overlays

When upstream vcpkg fixes the issue:
1. Test without overlay first
2. Remove overlay directory
3. Update vcpkg baseline
4. Verify build succeeds
5. Document the change in commit message

---

## Quick Reference: vcpkg Commands

```bash
# List installed packages
./vcpkg/vcpkg list

# Search for package
./vcpkg/vcpkg search <package-name>

# Install package manually
./vcpkg/vcpkg install <package-name>

# Remove package
./vcpkg/vcpkg remove <package-name>

# Update vcpkg itself
cd vcpkg && git pull && ./bootstrap-vcpkg.sh

# Clear vcpkg cache
rm -rf ~/.cache/vcpkg/archives/*

# Rebuild single package
./vcpkg/vcpkg remove <package-name>
./vcpkg/vcpkg install <package-name> --recurse

# Check package dependencies
./vcpkg/vcpkg depend-info <package-name>

# Export installed packages
./vcpkg/vcpkg export --zip --output=packages.zip

# Verify installation
./vcpkg/vcpkg owns <file-path>
```
