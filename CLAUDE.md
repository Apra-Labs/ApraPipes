# CLAUDE.md - ApraPipes Declarative Pipeline

> Instructions for Claude Code agents working on the ApraPipes project.

**Branch:** `feat/sdk-packaging`
**Documentation:** `docs/declarative-pipeline/`

---

## Current Phase: Sprint 12 - Windows Integration Test Fix

**Mission:** Fix Windows integration tests that fail with exit code 127.

**Problem:**
- Windows integration tests fail with exit code 127 (CLI fails to launch)
- Linux, macOS, and ARM64 all pass
- Root cause: Git Bash PATH handling for DLL loading is problematic on Windows

**Solution:**
- Use PowerShell (pwsh) for Windows integration tests
- Native Windows PATH handling works correctly
- Linux/macOS continue to use bash (works correctly)

**Status:** Awaiting CI verification (commit c41375381)

---

## SDK Structure (Complete)

```
aprapipes-sdk-{platform}/
├── bin/
│   ├── aprapipes_cli              # CLI tool
│   ├── aprapipesut                # Unit tests
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
│   └── node/                      # Node.js examples
├── data/
│   ├── frame.jpg                  # Sample input files
│   └── faces.jpg                  # For examples to work out of box
├── README.md                      # SDK usage documentation
└── VERSION                        # Version info
```

**Current State:**
| Workflow | SDK Artifact | Status |
|----------|-------------|--------|
| CI-Windows | `aprapipes-sdk-windows-x64` | ✅ Complete (integration tests pending) |
| CI-Linux | `aprapipes-sdk-linux-x64` | ✅ Complete |
| CI-MacOSX | `aprapipes-sdk-macos-arm64` | ✅ Complete |
| CI-Linux-ARM64 | `aprapipes-sdk-linux-arm64` | ✅ Complete |

**Protected Assets (DO NOT BREAK):**
- All 4 CI workflows GREEN
- GPU tests (CI-CUDA-Tests.yml) using fixed artifact names
- Existing test functionality

---

## Critical Rules

### 1. Build and Test Before Commit (MANDATORY)

**NEVER commit code without verifying build and tests pass.**

```bash
# 1. Build must succeed
cmake --build build -j$(nproc)

# 2. Tests must pass
./build/aprapipesut --run_test="<RelevantSuite>/*" --log_level=test_suite

# 3. For CLI changes, smoke test
./build/aprapipes_cli run <example.json>
```

If build/tests fail: fix first, then commit. No exceptions.

### 2. Wait for CI Before Push

Before pushing to this branch, verify all current CI runs are complete:

```bash
gh run list --limit 10 --json status,name,conclusion,headBranch | jq -r '.[] | select(.status != "completed") | "\(.name) (\(.headBranch))"'
```

### 3. Platform Protection

**Keep all 4 CI workflows GREEN:**
- CI-Windows, CI-Linux, CI-Linux-ARM64, CI-MacOSX-NoCUDA

**GPU Test Compatibility:**
- Fixed artifact names: `aprapipes-sdk-{os}-x64`
- CI-CUDA-Tests.yml downloads these artifacts - don't rename!

### 4. Code Review Before Commit

```bash
git diff --staged          # Review ALL changes
git diff --staged --stat   # Check which files changed
```

Check for: debug code, temporary hacks, commented-out code, unrelated changes.

---

## Implementation Tasks

### Sprint 12: Windows Integration Test Fix (Current)

1. [x] Analyze CI failure logs (exit code 127)
2. [x] Identify root cause (Git Bash PATH conversion)
3. [x] Implement PowerShell integration tests for Windows
4. [ ] Verify fix on CI (awaiting run)

### SDK Packaging (Complete)

1. [x] Update `build-test.yml` (Windows/Linux x64) - SDK packaging
2. [x] Update `build-test-macosx.yml` - SDK packaging
3. [x] Update `build-test-lin.yml` (ARM64) - SDK packaging
4. [x] Create `docs/SDK_README.md` - SDK usage documentation
5. [x] Integration tests added (basic, CUDA, Node.js, Jetson)

### Phase 2: GitHub Releases (Deferred)

1. [ ] Create `release.yml` - coordinated release workflow
2. [ ] Test release workflow creates single release with all 4 platforms

---

## Jetson Development

### Device Rules

When working on Jetson (ssh akhil@192.168.1.18):
- **NEVER** modify `/data/action-runner/` (GitHub Actions)
- **NEVER** delete `/data/.cache/` (vcpkg cache shared with CI)
- **ALWAYS** work in `/data/ws/`

### Build Commands

```bash
ssh akhil@192.168.1.18
cd /data/ws/ApraPipes

# Configure
cmake -B _build -S base \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_ARM64=ON \
  -DENABLE_CUDA=ON

# Build (use -j2 to avoid OOM)
TMPDIR=/data/.cache/tmp cmake --build _build -j2

# Test
./_build/aprapipesut --run_test="ModuleRegistryTests/*" --log_level=test_suite
```

---

## Quick Reference

```bash
# Check progress
cat docs/declarative-pipeline/PROGRESS.md

# Check CI status
gh run list --limit 8

# Wait for CI before push
gh run list --json status,name --jq '.[] | select(.status != "completed")'

# Build
cmake --build build -j$(nproc)

# Test specific suite
./build/aprapipesut --run_test="ModuleRegistryTests/*" --log_level=test_suite

# Run CLI
./build/aprapipes_cli list-modules
./build/aprapipes_cli run examples/simple.json
```

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/declarative-pipeline/SDK_PACKAGING_PLAN.md` | SDK packaging plan |
| `docs/declarative-pipeline/PROGRESS.md` | Current status, sprint progress |
| `docs/declarative-pipeline/PROJECT_PLAN.md` | Sprint overview, objectives |
| `.github/workflows/build-test.yml` | Windows/Linux x64 workflow |
| `.github/workflows/build-test-macosx.yml` | macOS workflow |
| `.github/workflows/build-test-lin.yml` | ARM64 workflow |
