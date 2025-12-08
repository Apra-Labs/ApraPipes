# ApraPipes DevOps TODO

Action items and improvements to be implemented when current build fixes are complete.

---

## High Priority

_No high priority items at this time. Current build system is stable._

---

## Medium Priority

### TODO-1: Publish Pre-Built Developer Docker Images with vcpkg Cache

**Status**: Not Started
**Priority**: Medium-High
**Estimated Effort**: 3-4 hours

**Objective**: Create and publish Docker images with ApraPipes development environment pre-installed including vcpkg cache to accelerate developer onboarding.

**Benefits**:
- Dramatically faster developer setup (minutes instead of hours)
- Consistent development environment across team
- Pre-built vcpkg dependencies (saves 30-60 min of build time per developer)
- Easier CI/CD debugging (developers can reproduce exact CI environment locally)
- Reduces "works on my machine" issues

**Implementation Plan**:
1. Create multi-stage Dockerfiles for each platform:
   - `Dockerfile.dev-linux-cuda` - Based on nvidia/cuda:11.8.0-devel-ubuntu20.04
   - `Dockerfile.dev-linux-nocuda` - Based on ubuntu:22.04
   - `Dockerfile.dev-jetson` - Based on nvcr.io/nvidia/l4t-base

2. Include in each image:
   - All system dependencies (libssl-dev, build-essential, cmake, ninja, etc.)
   - vcpkg bootstrapped and ready
   - Pre-built vcpkg cache at `/opt/vcpkg-cache` (copy from successful CI run)
   - ApraPipes repository cloned to `/workspace/aprapipes`
   - Development tools (git, vim, gdb, valgrind, etc.)
   - Documentation on how to mount local code for development

3. Build and tag images:
   ```bash
   docker build -f Dockerfile.dev-linux-cuda -t ghcr.io/apra-labs/aprapipes-dev:linux-cuda-latest .
   docker build -f Dockerfile.dev-linux-nocuda -t ghcr.io/apra-labs/aprapipes-dev:linux-nocuda-latest .
   ```

4. Publish to GitHub Container Registry (ghcr.io):
   ```bash
   docker push ghcr.io/apra-labs/aprapipes-dev:linux-cuda-latest
   docker push ghcr.io/apra-labs/aprapipes-dev:linux-nocuda-latest
   ```

5. Add usage documentation to README.md:
   ```bash
   # Quick start for developers
   docker run -it --rm \
     -v $(pwd):/workspace/aprapipes \
     -v aprapipes-build:/workspace/aprapipes/build \
     ghcr.io/apra-labs/aprapipes-dev:linux-nocuda-latest

   # Inside container
   cd /workspace/aprapipes/build
   cmake ../base -DCMAKE_BUILD_TYPE=RelWithDebInfo
   cmake --build . -j4
   ```

6. Automate image builds in CI:
   - Trigger on successful main branch builds
   - Tag with version (e.g., `v2024.12.01`) and `latest`
   - Store vcpkg cache as artifact, copy into Docker image during build

**Estimated Savings**:
- Developer onboarding: 2-3 hours → 15 minutes
- vcpkg rebuild time: 45 minutes → 0 minutes (pre-cached)
- CI debugging iterations: 30% faster (local reproduction)

**Related**: This aligns with TODO-7 (Autonomous Agent Design) - agent could use these images for testing fixes

---

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

**Completed**: 2024-11-29

**Key Fixes Applied**:
- Python 3.12+ removed distutils → Use Python 3.10.11 in vcpkg-tools.json
- pkgconf missing → Add to vcpkg.json dependencies for PKG_CONFIG_EXECUTABLE
- Version breaking changes → Pin critical packages (SFML, Boost, OpenCV) in vcpkg.json overrides
- Boost filesystem API changes → Change `boost::filesystem::extension(path)` to `path.extension()`
- vcpkg override name mismatch → Override must match dependency name exactly (`opencv4` not `opencv`)
- vcpkg baseline must be advertised → Use branch tips or tags, not parent commits

**Pattern**: When vcpkg baseline updates, newer package versions may have breaking API changes.
Pin known-good versions and update code to match new APIs.

See `troubleshooting.windows.md` for detailed diagnostic steps for each issue type.

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
