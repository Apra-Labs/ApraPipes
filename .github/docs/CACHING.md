# GitHub Actions Caching Strategy

## Overview

ApraPipes uses GitHub Actions caching to store vcpkg-built dependencies between workflow runs. This dramatically reduces build times from ~4 hours to ~2 hours by avoiding redundant package compilation.

## Cache Architecture

### Cache Key Structure

```
{flavor}-{version}-{hash}
```

- **flavor**: Build variant (e.g., `Win-nocuda`, `Linux`, `Windows-cuda`)
- **version**: Cache schema version (currently `5`)
- **hash**: SHA256 of `vcpkg.json`, `vcpkg/baseline.json`, and `submodule_ver.txt`

**Example**: `Win-nocuda-5-a1b2c3d4e5f6...`

### Cache Locations

| Platform | Path |
|----------|------|
| Windows | `C:\Users\runneradmin\AppData\Local\vcpkg\archives` |
| Linux | `~/.cache/vcpkg/archives` |
| WSL | `~/.cache/vcpkg/archives` |

## Normal Caching Workflow

### Automatic Cache Management

In standard builds, caching uses `actions/cache@v3` which handles both restore and save automatically:

```yaml
- name: Cache dependencies for fast cloud build
  id: cache-all
  if: ${{ !inputs.force-cache-update }}
  uses: actions/cache@v3
  with:
    path: ${{ inputs.cache-path }}
    key: ${{ inputs.flav }}-5-${{ hashFiles('base/vcpkg.json', 'vcpkg/baseline.json', 'submodule_ver.txt') }}
    restore-keys: ${{ inputs.flav }}-
```

### Behavior

**Cache Hit (exact key match)**:
- Restores cached dependencies
- Sets `cache-hit = 'true'`
- Skips save at job end (cache already exists)

**Cache Miss (no exact match)**:
- Attempts partial restore using `restore-keys` prefix
- Restores most recent matching cache
- CMake builds missing/updated packages
- **Automatically saves** new cache at job end with exact key

**First Run (no cache exists)**:
- No restore occurs
- CMake builds all packages from source
- Saves complete cache at job end

## Force Cache Update

### Purpose

Use `force-cache-update` when:

1. **Cache corruption**: Incomplete or broken cached packages
2. **Dependency upgrades**: Forcing rebuild with newer vcpkg package versions
3. **Debugging**: Eliminating cache as source of build issues
4. **Cache bloat**: Resetting to clean state

### Mechanism

When `force-cache-update: true`, the workflow:

1. **Restores cache** (read-only operation)
2. **Deletes the restored cache** from GitHub's cache storage
3. **Builds all packages fresh** via CMake/vcpkg
4. **Saves new cache** after configure completes

```yaml
- name: Force cache update - restore and delete old cache
  id: cache-force-restore
  if: ${{ inputs.force-cache-update }}
  uses: actions/cache/restore@v3
  with:
    path: ${{ inputs.cache-path }}
    key: ${{ inputs.flav }}-5-${{ hashFiles(...) }}
    restore-keys: ${{ inputs.flav }}-

- name: Force cache update - delete old cache
  if: ${{ inputs.force-cache-update && steps.cache-force-restore.outputs.cache-matched-key != '' }}
  env:
    GH_TOKEN: ${{ github.token }}
  run: gh cache delete "${{ steps.cache-force-restore.outputs.cache-matched-key }}"

- name: Force cache update - save updated cache
  if: ${{ always() && inputs.force-cache-update }}
  uses: actions/cache/save@v3
  with:
    path: ${{ inputs.cache-path }}
    key: ${{ inputs.flav }}-5-${{ hashFiles(...) }}
```

### Triggering Force Update

**Via GitHub UI**:
1. Navigate to Actions tab
2. Select workflow (e.g., "CI-Win-NoCUDA")
3. Click "Run workflow"
4. Check "Force cache rebuild" checkbox
5. Click "Run workflow"

**Via workflow_dispatch input**:
```yaml
workflow_dispatch:
  inputs:
    force-cache-update:
      description: 'Force cache rebuild (deletes old cache and builds fresh)'
      required: false
      type: boolean
      default: false
```

## Cache Invalidation

Cache automatically invalidates when:

1. **vcpkg.json changes**: Dependencies added/removed/modified
2. **vcpkg baseline changes**: vcpkg package versions updated
3. **Git submodules change**: Submodule commits updated

The hash in the cache key ensures any of these changes create a new cache entry.

## Multi-Phase Builds

Some workflows split builds into prep + test phases:

### Phase 1: Prep (cache population)
- `is-prep-phase: true`
- Runs CMake configure (builds vcpkg dependencies)
- Saves cache with exact key
- Skips actual compilation and testing

### Phase 2: Test (cache consumption)
- `is-prep-phase: false`
- Restores cache from Phase 1 (exact key match)
- Runs compilation and tests
- No cache save (already exists)

This separation optimizes GitHub-hosted runners with limited storage.

## Best Practices

### When to Use Normal Caching
- ✅ Daily development builds
- ✅ Pull request validation
- ✅ Scheduled CI runs
- ✅ Any build where dependencies haven't changed

### When to Use Force Update
- ⚠️ Build failures suspected due to cache corruption
- ⚠️ After major vcpkg version bump
- ⚠️ Quarterly cache refresh (preventive maintenance)
- ⚠️ Debugging persistent build issues

### Cache Size Management

GitHub has cache limits:
- **10 GB** per repository
- Oldest caches evicted automatically

To prevent bloat:
- Use specific cache keys (not overly broad restore-keys)
- Avoid force-update unless necessary
- Monitor cache usage in repo Settings → Actions → Caches

## Troubleshooting

### Cache Not Restoring
**Symptom**: Every build rebuilds all packages
**Check**:
1. Cache key hash calculation (ensure files exist)
2. Cache branch restrictions (caches are branch-scoped)
3. GitHub cache storage limits reached

### Build Slower After Force Update
**Cause**: Normal - force update rebuilds everything from scratch
**Solution**: Subsequent builds will be fast once new cache is saved

### Inconsistent Cache Behavior
**Symptom**: Same commit shows different cache hits
**Check**:
1. Parallel builds may create duplicate caches
2. Race condition if multiple jobs save same key
3. Cache eviction due to size limits

## Implementation Files

- **Reusable workflows**: `.github/workflows/build-test-{win,lin,lin-wsl}.yml`
- **CI workflows**: `.github/workflows/CI-*.yml`
- **Cache key inputs**: `flav`, `cache-path` in workflow parameters
- **Force update flag**: `force-cache-update` boolean input

## Permissions Required

Force cache update requires:
```yaml
permissions:
  actions: write  # For cache deletion via gh CLI
  contents: read  # For checkout
```

Standard caching only requires `contents: read`.
