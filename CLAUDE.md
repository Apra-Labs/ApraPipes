# CLAUDE.md — ApraPipes DevOps Agent

## Mission Statement

### Goal: Unified CI Architecture

Consolidate CI workflows into a single unified reusable workflow (`build-test.yml`) that:
1. Supports all platforms (Linux, Windows, macOS, Docker, Jetson)
2. Builds with CUDA support but runs gracefully on non-GPU systems
3. Uses release-only triplets for ~50% faster builds
4. Shares vcpkg cache across related builds (e.g., Linux cloud -> Docker)

### Completed Work

| Component | Status | Details |
|-----------|--------|---------|
| Phase 1: Unified Builds | DONE | Runtime CUDA detection via Driver API |
| Release-only Triplets | DONE | PR #461 merged - `x64-linux-release`, etc. |
| Unified Reusable Workflow | DONE | `build-test.yml` handles all platforms |
| Cache Key Standardization | DONE | `<os>-vcpkg-<hash>` pattern everywhere |
| Docker Chaining | DONE | Docker build runs after cloud tests pass |

### Platform Coverage

| Platform | Workflow | Tests | Status |
|----------|----------|-------|--------|
| Linux x64 (CUDA) | CI-Linux-Build-Test-v2 | Cloud + Self-hosted GPU | Testing |
| Linux x64 (Docker) | Chained from Linux | Container tests | Testing |
| Windows x64 | CI-Windows-Build-Test-v2 | Cloud + Self-hosted GPU | Testing |
| Linux ARM64 (Jetson) | CI-Linux-ARM64 | Native GPU | Working |
| macOS | CI-MacOSX-NoCUDA | Cloud only | Working |

### Key Design Decisions

1. **Single reusable workflow** - `build-test.yml` replaces per-platform duplicates
2. **Thin caller workflows** - `CI-<platform>-Build-Test-v2.yml` just pass parameters
3. **Chained Docker builds** - Docker only runs after cloud tests pass (saves resources)
4. **Shared caches** - Docker reuses cloud build's vcpkg cache via matching keys
5. **Release-only builds** - No debug symbols = faster builds + smaller artifacts

---

## Operating Instructions

### At Session Start
1. Read `.claude/LEARNINGS.md` — don't repeat past mistakes
2. Read `.claude/CURRENT_STATE.md` — know where we left off
3. Check build status: `gh run list -L3`

### During Work
- After triggering workflows, ALWAYS watch them: `gh run watch <id> --exit-status`
- Update `CURRENT_STATE.md` when significant progress is made
- Add new learnings to `LEARNINGS.md` immediately after discovering them

### Before Stopping
1. Update `CURRENT_STATE.md` with current status
2. Document any new learnings in `LEARNINGS.md`
3. Check for pending workflows: `gh run list -L3 --json status -q '.[] | select(.status=="in_progress")'`

### Slash Commands
- `/project:status` — Quick status check, continue working
- `/project:resume` — Resume from saved state
- `/project:checkpoint` — Force save current state
- `/project:last-run` — Analyze most recent GH Actions run
- `/project:learned <desc>` — Quick-add a learning

---

## Key Files

### State & Memory
- `.claude/CURRENT_STATE.md` — Current task, build status, blockers
- `.claude/LEARNINGS.md` — Institutional memory, past mistakes & fixes

### Documentation
- `.claude/docs/` — Implementation docs, experiment results
- `.claude/skills/aprapipes-devops/` — Troubleshooting guides by platform

### CI Workflows (v2 Architecture)
- `.github/workflows/build-test.yml` — Unified reusable workflow for all platforms
- `.github/workflows/CI-<platform>-Build-Test-v2.yml` — Thin caller workflows
- `.github/workflows/publish-test.yml` — Reusable test result publishing
- `.github/workflows/build-test-lin-container.yml` — Docker container build
- `.github/workflows/build-test-macosx.yml` — macOS-specific build

### Key Config
- `vcpkg/triplets/community/x64-*-release.cmake` — Release-only triplets
- `vcpkg/triplets/community/x64-windows-cuda.cmake` — Windows CUDA (v142 toolset)
- `base/vcpkg.json` — Dependencies with CUDA features
- `thirdparty/custom-overlay/` — Custom vcpkg ports (libpng, sfml, baresip/re)

---

## Technical Details

### Runtime CUDA Detection
Binaries compiled with CUDA can run on systems without GPU by using Driver API:

**Linux:**
```cpp
void* handle = dlopen("libcuda.so.1", RTLD_LAZY);
auto cuInit = (cuInit_t)dlsym(handle, "cuInit");
```

**Windows:**
```cpp
HMODULE handle = LoadLibraryA("nvcuda.dll");
auto cuInit = (cuInit_t)GetProcAddress(handle, "cuInit");
```

### Windows DELAYLOAD
All CUDA DLLs must be delay-loaded for exe to start on non-GPU systems:
- **Toolkit**: nvjpeg64_11, nppig64_11, nppicc64_11, nppidei64_11, nppial64_11, nppc64_11, cublas64_11, cublasLt64_11, cudart64_110
- **Driver**: nvcuvid, nvEncodeAPI64

### Cache Key Pattern
All workflows use consistent hash-based cache keys for automatic invalidation:
```yaml
key: ${{ os }}-vcpkg-${{ hashFiles('base/vcpkg.json', 'vcpkg/baseline.json', 'submodule_ver.txt') }}
restore-keys: ${{ os }}-vcpkg-
```

### JetPack 5.x Compatibility (ARM64)
- Custom `nvbuf_utils.h` compatibility layer for API changes
- Supports both `nvbuf_utils` and `nvbufsurface` library names
- Handles `NVBUFFER_TRANSFORM_*` flag differences
