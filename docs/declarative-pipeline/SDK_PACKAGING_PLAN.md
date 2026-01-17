# SDK Packaging Improvement Plan

> Created: 2026-01-17

## Goal

Create a consistent, reusable SDK packaging mechanism across all 4 CI workflows that:
1. Packages all artifacts (CLI, Node addon, libraries, examples)
2. Works out of the box for end users
3. Can be published as GitHub Releases

## Versioning Strategy

**Format:** `{major}.{minor}.{patch}-g{short-hash}`

**Examples:**
- `2.0.0-g6146afb` - Development build
- `2.0.0` - Tagged release (no hash suffix)

**Implementation:**
```bash
# Get version string
VERSION=$(git describe --tags --always 2>/dev/null || echo "0.0.0-g$(git rev-parse --short HEAD)")
# Result: "2.0.0-g6146afb" or "v2.0.0" (for tagged commits)
```

## Naming Strategy

### Phase 1: Fixed Names (Now)
Consistent names across all platforms for CI compatibility:

| Platform | Artifact Name |
|----------|---------------|
| Windows x64 | `aprapipes-sdk-windows-x64` |
| Linux x64 | `aprapipes-sdk-linux-x64` |
| macOS ARM64 | `aprapipes-sdk-macos-arm64` |
| Linux ARM64 | `aprapipes-sdk-linux-arm64` |

**VERSION file inside SDK** contains the version string (e.g., `2.0.0-g6146afb`) for programmatic access.

### Phase 2: Release Assets (Deferred)
When release workflow is implemented, assets will be renamed:

| Platform | Release Asset Name |
|----------|-------------------|
| Windows x64 | `aprapipes-sdk-2.0.0-g6146afb-windows-x64.zip` |
| Linux x64 | `aprapipes-sdk-2.0.0-g6146afb-linux-x64.tar.gz` |
| macOS ARM64 | `aprapipes-sdk-2.0.0-g6146afb-macos-arm64.tar.gz` |
| Linux ARM64 | `aprapipes-sdk-2.0.0-g6146afb-linux-arm64.tar.gz` |

## GPU Test Impact Analysis

**Current Flow:**
```
CI-Windows/CI-Linux (build job)
    ↓ uploads artifact: aprapipes-sdk-{platform}
CI-CUDA-Tests.yml (gpu job on self-hosted runner)
    ↓ downloads artifact by fixed name
    ↓ runs aprapipesut with GPU tests
```

**Impact:** ✅ **No breaking changes**

The two-tier naming strategy preserves fixed artifact names for internal CI consumption. GPU tests will continue to work without modification. Versioned names are only used for GitHub Releases (external distribution).

## Current State

| Workflow | SDK Artifact | Contents |
|----------|-------------|----------|
| CI-Windows | ✅ `aprapipes-sdk-windows-x64` | bin/, lib/, include/ |
| CI-Linux | ✅ `aprapipes-sdk-linux-x64` | bin/, lib/, include/ |
| CI-MacOSX | ❌ None | Only test results |
| CI-Linux-ARM64 | ❌ None | Only test results |

**Current SDK contents are minimal:** Only test executable, libraries, and headers.

## Proposed SDK Structure

```
aprapipes-sdk-{platform}/
├── bin/
│   ├── aprapipes_cli              # CLI tool
│   ├── aprapipesut                # Unit tests (optional, for validation)
│   ├── aprapipes.node             # Node.js addon
│   └── *.so / *.dll / *.dylib     # Shared libraries
├── lib/
│   └── *.a / *.lib                # Static libraries
├── include/
│   └── *.h                        # Header files
├── examples/
│   ├── basic/                     # JSON pipeline examples
│   ├── cuda/                      # CUDA examples (if applicable)
│   ├── jetson/                    # Jetson examples (ARM64 only)
│   ├── node/                      # Node.js examples
│   └── README.md                  # Examples documentation
├── data/
│   ├── frame.jpg                  # Sample input files
│   └── faces.jpg                  # For examples to work out of box
├── README.md                      # SDK usage documentation
└── VERSION                        # Version info
```

## Platform Matrix

| Component | Windows | Linux x64 | macOS | ARM64/Jetson |
|-----------|---------|-----------|-------|--------------|
| aprapipes_cli | ✅ | ✅ | ✅ | ✅ |
| aprapipes.node | ✅ | ✅ | ✅ | ✅ |
| libaprapipes | ✅ | ✅ | ✅ | ✅ |
| examples/basic | ✅ | ✅ | ✅ | ✅ |
| examples/cuda | ✅ (CUDA) | ✅ (CUDA) | ❌ | ✅ |
| examples/jetson | ❌ | ❌ | ❌ | ✅ |
| examples/node | ✅ | ✅ | ✅ | ✅ |

## Implementation Plan

### Phase 1: Create Reusable Packaging Workflow

Create `.github/workflows/package-sdk.yml`:

```yaml
name: Package SDK
on:
  workflow_call:
    inputs:
      platform:
        description: 'Platform name (windows-x64, linux-x64, macos-arm64, linux-arm64)'
        required: true
        type: string
      build-dir:
        description: 'Build output directory'
        required: true
        type: string
      has-cuda:
        description: 'Include CUDA examples'
        type: boolean
        default: false
      is-jetson:
        description: 'Include Jetson-specific examples'
        type: boolean
        default: false
    outputs:
      version:
        description: 'SDK version string'
        value: ${{ jobs.package.outputs.version }}
```

**Key Implementation Details:**
```yaml
jobs:
  package:
    runs-on: ubuntu-latest  # Packaging is platform-agnostic
    outputs:
      version: ${{ steps.version.outputs.version }}
    steps:
      - name: Get version
        id: version
        run: |
          VERSION=$(git describe --tags --always 2>/dev/null || echo "0.0.0-g$(git rev-parse --short HEAD)")
          echo "version=$VERSION" >> $GITHUB_OUTPUT
          echo "$VERSION" > VERSION

      - name: Package SDK
        run: |
          # Create SDK structure
          mkdir -p sdk/{bin,lib,include,examples,data}
          # ... copy files ...

      - name: Upload artifact (fixed name for CI)
        uses: actions/upload-artifact@v4
        with:
          name: aprapipes-sdk-${{ inputs.platform }}  # Fixed name
          path: sdk/
```

### Phase 2: SDK Contents by Platform

**All platforms:**
- `bin/aprapipes_cli` (or `.exe` on Windows)
- `bin/aprapipes.node`
- `bin/aprapipesut` (for installation validation)
- `lib/libaprapipes.*` / `aprapipes.lib`
- `include/*`
- `examples/basic/*`
- `examples/node/*`
- `data/frame.jpg` (~110KB)
- `data/faces.jpg` (~92KB)
- `VERSION` (contains version string)
- `README.md` (SDK usage documentation)

**CUDA platforms (Windows-CUDA, Linux-CUDA, ARM64):**
- `examples/cuda/*`

**Jetson only:**
- `examples/jetson/*`

### Phase 3: Update Existing Workflows

Modify each workflow to call `package-sdk.yml`:

```yaml
# In build-test.yml (Windows/Linux), build-test-macosx.yml, build-test-lin.yml
package:
  needs: [build]
  uses: ./.github/workflows/package-sdk.yml
  with:
    platform: linux-x64
    build-dir: build
    has-cuda: ${{ inputs.enable_cuda }}
```

**No changes to GPU test jobs** - they continue downloading `aprapipes-sdk-{platform}`.

### Phase 4: GitHub Releases (Deferred)

**Deferred to Phase 2** - Implement after SDK packaging is stable.

A coordinated release workflow will:
- Run nightly or on-demand (manual trigger)
- Check if a new release is needed (new tag since last release)
- Trigger all 4 CI workflows and wait for completion
- Collect SDK artifacts from all 4 platforms
- Create a **single** GitHub Release with all 4 archives attached

```yaml
# Future: .github/workflows/release.yml
name: Release
on:
  workflow_dispatch:  # Manual trigger
  schedule:
    - cron: '0 0 * * *'  # Nightly check

jobs:
  check-release:
    # Determine if new tag exists since last release

  build-all-platforms:
    needs: [check-release]
    if: needs.check-release.outputs.should_release == 'true'
    # Trigger CI-Windows, CI-Linux, CI-MacOSX, CI-Linux-ARM64

  create-release:
    needs: [build-all-platforms]
    # Download all 4 SDK artifacts
    # Create single GitHub Release with all 4 attached
```

**Not implementing now** - focus on consistent SDK packaging first.

## Data Files for Examples

Minimal data files needed (keep SDK size small):

| File | Size | Used By |
|------|------|---------|
| `data/frame.jpg` | ~110KB | Most basic examples |
| `data/faces.jpg` | ~92KB | Face detection examples |

Total: ~202KB

## Estimated Changes

### Phase 1 (Now) - SDK Packaging

| File | Change |
|------|--------|
| `.github/workflows/package-sdk.yml` | NEW - Reusable packaging workflow |
| `.github/workflows/build-test.yml` | Modify to call package-sdk (after build job) |
| `.github/workflows/build-test-lin.yml` | Modify to call package-sdk |
| `.github/workflows/build-test-macosx.yml` | Modify to call package-sdk |
| `.github/workflows/CI-CUDA-Tests.yml` | No changes (uses fixed artifact names) |
| `docs/SDK_README.md` | NEW - SDK usage documentation |

### Phase 2 (Deferred) - GitHub Releases

| File | Change |
|------|--------|
| `.github/workflows/release.yml` | NEW - Coordinated release workflow |

## Decisions (Approved)

1. **SDK naming convention:** ✅ Fixed names for Phase 1
   - `aprapipes-sdk-{platform}` (e.g., `aprapipes-sdk-linux-x64`)
   - Versioned names deferred to Phase 2 (release workflow)

2. **Include unit tests (aprapipesut)?** ✅ Yes
   - Already included in current SDK, kept for installation validation

3. **Data files:** ✅ Include minimal set
   - `frame.jpg` (~110KB) and `faces.jpg` (~92KB)
   - Total: ~202KB for out-of-box examples

4. **Versioning:** ✅ `{major}.{minor}.{patch}-g{short-hash}`
   - Generated via `git describe --tags --always`
   - VERSION file inside SDK for programmatic access

5. **GPU test impact:** ✅ No breaking changes
   - Fixed artifact names preserved for internal CI consumption
   - GPU tests continue downloading `aprapipes-sdk-{platform}`

6. **GitHub Releases:** ⏳ Deferred to Phase 2
   - Coordinated release workflow (nightly/manual)
   - Single release with all 4 platform SDKs attached

7. **Retention:** 7 days for PR builds, permanent for releases

## Next Steps

### Phase 1: SDK Packaging (Now)

1. ✅ Plan approved (decisions above)
2. ⏳ Create `package-sdk.yml` reusable workflow
3. ⏳ Update `build-test.yml` (Windows/Linux x64) to call package-sdk
4. ⏳ Update `build-test-macosx.yml` to call package-sdk
5. ⏳ Update `build-test-lin.yml` (ARM64) to call package-sdk
6. ⏳ Test all workflows - verify GPU tests still work
7. ⏳ Create `docs/SDK_README.md` - SDK usage documentation

### Phase 2: GitHub Releases (Deferred)

8. ⏳ Create `release.yml` - coordinated nightly/manual release workflow
9. ⏳ Test release workflow creates single release with all 4 platforms
