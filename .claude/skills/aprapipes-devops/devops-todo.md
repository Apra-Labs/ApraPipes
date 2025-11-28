# ApraPipes DevOps TODO

Action items and improvements to be implemented when current build fixes are complete.

---

## High Priority (Implement After Build #13 Success)

### TODO-1: Migrate to Windows Server 2022 and Single-Phase Builds

**Status**: Pending Build #13 success
**Priority**: High
**Estimated Effort**: 2-4 hours

**Context**:
- GitHub-hosted runners have **6 hour limit per job**, not 1 hour as previously assumed
- Windows Server 2025 only has ~33 GB disk space (D: drive removed)
- Windows Server 2022 has ~185 GB disk space (38 GB C: + 147 GB D:)
- Current two-phase build strategy was designed for 1-hour constraint that doesn't exist

**Objective**:
1. Switch from `windows-latest` to `windows-2022` for stable disk space
2. Eliminate two-phase build complexity (Phase 1 prep + Phase 2 build/test)
3. Use single-phase build (just Phase 2 equivalent - full build + test)

**Benefits**:
- ✅ Simplified workflow (one job instead of two)
- ✅ No cache invalidation issues
- ✅ Faster overall (no Phase 1 overhead)
- ✅ More disk space available (185 GB vs 33 GB)
- ✅ Easier debugging (one job to check, not two)

**Implementation Steps**:

1. **Update Runner Specification**
   ```yaml
   # .github/workflows/CI-Win-NoCUDA.yml
   # Change from:
   runner: windows-latest

   # To:
   runner: windows-2022
   ```

2. **Simplify Workflow to Single Phase**
   ```yaml
   # Remove is-prep-phase parameter entirely
   # Remove Phase 1 job
   # Keep only Phase 2 (build-test) job

   jobs:
     build-and-test:  # Rename from win-nocuda-build-test
       uses: ./.github/workflows/build-test-win.yml
       with:
         runner: windows-2022
         flav: Win-nocuda
         is-selfhosted: false
         is-prep-phase: false  # Will always be false
         cuda: "OFF"
   ```

3. **Update build-test-win.yml**
   - Remove conditional logic for `is-prep-phase`
   - Remove `fix-vcpkg-json.ps1 -onlyOpenCV` step
   - Keep full vcpkg.json always
   - Simplify cache strategy (optional - may not need cache at all)

4. **Test Migration**
   - Create test branch: `update/windows-2022-single-phase`
   - Trigger manual build
   - Verify:
     - [ ] Build completes within 6 hours
     - [ ] Disk space sufficient (~185 GB available)
     - [ ] Tests pass
     - [ ] No cache-related issues

5. **Update Documentation**
   - Update `.claude/skills/aprapipes-devops/reference.md`:
     - Time Limit: 1 hour → 6 hours
     - Disk Space: 14 GB → 185 GB (Windows 2022)
     - Build Strategy: Two-phase → Single-phase
   - Update `troubleshooting.windows.md`:
     - Remove Phase 1 vs Phase 2 troubleshooting split
     - Simplify to single build phase diagnostics
   - Update `SKILL.md`:
     - Remove Phase 1/Phase 2 routing logic

**Risks & Mitigations**:
- **Risk**: Build might exceed 6 hours
  - **Mitigation**: Monitor first build, implement optimizations if needed
- **Risk**: Disk space still insufficient
  - **Mitigation**: Keep disk cleanup steps, monitor usage
- **Risk**: Breaking existing workflows for other engineers
  - **Mitigation**: Coordinate change, update docs, communicate

**Success Criteria**:
- [ ] Single-phase build completes successfully
- [ ] Build time < 6 hours
- [ ] Disk usage < 185 GB
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Simpler workflow than before

**Assigned To**: TBD (DevOps team)
**Target Completion**: After Build #13 validates current fixes

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

**Completed**: 2024-11-28
**Issues Fixed**:
- Python distutils missing (downgraded to 3.10.11)
- PKG_CONFIG_EXECUTABLE not found (added pkgconf)
- SFML 3.x breaking changes (pinned to 2.6.2)
- Boost 1.89.0 breaking changes (pinned to 1.84.0)
- OpenCV header changes (pinned to 4.8.0)
- vcpkg baseline fetchability (used advertised commit)

### ✅ TODO-COMPLETED-2: Create ApraPipes DevOps Skill

**Completed**: 2024-11-28
**Deliverables**:
- SKILL.md orchestrator
- reference.md cross-platform reference
- troubleshooting.*.md platform guides
- devops-build-system-guide.md comprehensive guide

---

**Last Updated**: 2024-11-28
**Maintained By**: ApraPipes DevOps team
**Review Frequency**: After each major build system change
