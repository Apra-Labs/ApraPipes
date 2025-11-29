# ApraPipes DevOps TODO

Action items and improvements to be implemented when current build fixes are complete.

---

## High Priority

_No high priority items at this time. Current build system is stable._

---

## Medium Priority

### TODO-2: Pin All Critical Dependencies

**Status**: In Progress (SFML, Boost, OpenCV pinned; more needed)
**Priority**: Medium
**Estimated Effort**: 1-2 hours

**Objective**: Pin major versions of all Tier 1 critical dependencies to prevent future breaking changes.

**Packages to Pin** (see reference.md → Version Pinning Tiers):
- [ ] gtk3 (if version changes cause issues)
- [ ] glfw3
- [ ] whisper (rapidly evolving AI library)

**Implementation**:
Add to `base/vcpkg.json` overrides after verifying compatible versions.

---

### TODO-3: Create Version Snapshot Baseline

**Status**: Not Started
**Priority**: Medium
**Estimated Effort**: 30 minutes

**Objective**: Capture current working versions as baseline for future reference.

**Implementation**:
```bash
# After successful build
vcpkg list > .claude/skills/aprapipes-devops/vcpkg-versions-baseline-YYYY-MM-DD.txt
git add .claude/skills/aprapipes-devops/vcpkg-versions-baseline-*.txt
git commit -m "docs: Capture vcpkg version baseline for known working build"
```

**Benefit**: When future builds break, can quickly compare versions to identify what changed.

---

## Low Priority (Future Improvements)

### TODO-4: Optimize vcpkg Build Performance

**Status**: Not Started
**Priority**: Low
**Estimated Effort**: 4-8 hours

**Ideas**:
- Parallel package builds (vcpkg supports this)
- ccache integration for faster rebuilds
- Binary cache optimization

---

### TODO-5: Add Automated Version Pin Checks

**Status**: Not Started
**Priority**: Low
**Estimated Effort**: 2-3 hours

**Objective**: Add CI check to verify pinned package versions haven't drifted.

**Implementation**:
```yaml
- name: Verify version pins haven't changed
  run: |
    vcpkg list > current-versions.txt
    diff known-good-versions.txt current-versions.txt || exit 1
```

---

### TODO-6: Expand Platform-Specific Troubleshooting Guides

**Status**: Outlines created, needs expansion
**Priority**: Low (expand as issues occur)
**Estimated Effort**: Ongoing

**Guides to Expand**:
- [ ] troubleshooting.linux.md - Add Linux-specific patterns as they occur
- [ ] troubleshooting.cuda.md - Document CUDA build issues
- [ ] troubleshooting.jetson.md - Add Jetson ARM64 issues
- [ ] troubleshooting.docker.md - Document Docker/WSL issues

**Approach**: Add to guides as real issues are encountered, not speculatively.

---

## Completed TODOs (Archive)

### ✅ TODO-COMPLETED-1: Fix Windows NoCUDA Build Failures

**Completed**: 2024-11-29 (Build #17)
**Branch**: fix/ci-windows-ak

**Issues Fixed**:
1. **Python distutils missing** - Downgraded vcpkg Python to 3.10.11 in vcpkg-tools.json
2. **PKG_CONFIG_EXECUTABLE not found** - Added pkgconf to vcpkg.json dependencies
3. **SFML 3.x breaking changes** - Pinned SFML to 2.6.2 in vcpkg.json overrides
4. **Boost 1.84.0 API changes** - Fixed 11 occurrences of deprecated `boost::filesystem::extension()` calls:
   - `base/src/Mp4WriterSinkUtils.cpp` (2 locations)
   - `base/src/OrderedCacheOfFiles.cpp` (1 location)
   - `base/test/mp4_simul_read_write_tests.cpp` (8 locations)
5. **OpenCV version override not working** - Fixed override name from `opencv` to `opencv4` in vcpkg.json
6. **vcpkg baseline fetchability** - Used advertised commit (3011303ba1f6586e8558a312d0543271fca072c6)

**Workflow Improvements**:
- Implemented Option B: Phase 1 installs ALL dependencies (not just OpenCV)
- Phase 1 now validates CMake configure (removed continue-on-error)
- Build and test steps still properly skipped in Phase 1 via `if: !inputs.is-prep-phase`
- Fixed cleanup step error with `-ErrorAction SilentlyContinue`
- Added `workflow_dispatch` trigger to Linux NoCUDA workflow

**Build Results**:
- Build #17: ✅ PASSED (Phase 1: 2m39s, Phase 2: 20m13s)
- Linux NoCUDA: ✅ Validated (no regressions from changes)

### ✅ TODO-COMPLETED-2: Create ApraPipes DevOps Skill

**Completed**: 2024-11-28
**Deliverables**:
- SKILL.md orchestrator with platform routing
- reference.md cross-platform reference
- troubleshooting.windows.md (complete with real fixes)
- troubleshooting.linux.md (outline)
- troubleshooting.cuda.md (outline for self-hosted)
- troubleshooting.jetson.md (outline for ARM64)
- troubleshooting.docker.md (outline)
- devops-build-system-guide.md comprehensive guide

---

**Last Updated**: 2024-11-29
**Maintained By**: ApraPipes DevOps team
**Review Frequency**: After each major build system change
