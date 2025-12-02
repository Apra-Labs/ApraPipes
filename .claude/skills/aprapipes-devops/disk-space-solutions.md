# Disk Space Solutions for CI Builds

**Problem**: GitHub free runners have ~14GB available. vcpkg builds easily exceed this.

## Quick Fixes (Implement Immediately)

### 1. Build Only Release Variants
Saves ~50% disk space. CI doesn't need Debug builds.

```cmake
# vcpkg/triplets/x64-linux-release.cmake
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_BUILD_TYPE release)  # Only build release
```

Apply to all platforms: `x64-windows-release.cmake`, `arm64-linux-release.cmake`

Update workflows to use release triplet:
```yaml
-DVCPKG_TARGET_TRIPLET=x64-linux-release
```

### 2. Cleanup After prep-cmd
Saves ~500MB. Remove apt cache and temp files after installing dependencies.

```yaml
prep-cmd: |
  apt-get update -qq && apt-get install -y <packages> && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
```

### 3. Clean buildtrees Between Phases
Saves ~4-6GB. Delete vcpkg buildtrees after dependencies install, before building ApraPipes.

```yaml
- name: Install vcpkg dependencies
  run: cmake -B build ...  # Triggers vcpkg install

- name: Clean buildtrees
  run: rm -rf vcpkg/buildtrees/*

- name: Build ApraPipes
  run: cmake --build build
```

## Alternative: Binary Caching
Pre-built dependencies reused across builds. Fastest long-term solution.

```yaml
- name: Setup vcpkg binary cache
  run: |
    mkdir -p vcpkg_binary_cache
    echo "VCPKG_DEFAULT_BINARY_CACHE=${{github.workspace}}/vcpkg_binary_cache" >> $GITHUB_ENV

- uses: actions/cache@v3
  with:
    path: vcpkg_binary_cache
    key: vcpkg-${{ runner.os }}-${{ hashFiles('base/vcpkg.json') }}
```

## Last Resort: Paid Larger Runners
4-core (150GB SSD): $0.008/min. Only if above fixes insufficient.
