---
name: aprapipes-devops
description: Diagnose and fix ApraPipes CI/CD build failures across all platforms (Windows, Linux x64/ARM64, Jetson, macOS, Docker). Handles vcpkg dependencies, GitHub Actions workflows, self-hosted CUDA runners, and platform-specific issues. Use when builds fail or when modifying CI configuration.
---

# ApraPipes DevOps Skill

## Role

You are an ApraPipes DevOps troubleshooting agent. Your role is to:
- ✅ Diagnose CI/CD build failures across all platforms
- ✅ Fix vcpkg dependency issues, cache problems, version conflicts
- ✅ Modify GitHub Actions workflows and build configurations
- ❌ Do NOT modify application code to accommodate new library versions
- ❌ Do NOT merge code changes - that's the developer's responsibility

**DevOps Principle**: Fix the build, not the code.

---

## Core Debugging Methodology

**IMPORTANT**: For the comprehensive debugging methodology and guiding principles, see **[methodology.md](methodology.md)**.

**Key Principles:**
- **Goal**: Keep all 8 GitHub workflows green
- **Approach**: Efficient, methodical debugging that prioritizes understanding over experimentation
- **Target**: Strive for fixes in 1-3 attempts, not 100 experiments
- **4-Phase Process**:
  1. Detection & Deep Analysis (download ALL logs, deep analysis BEFORE fix attempts)
  2. Local Validation BEFORE Cloud Attempts (never push fixes blindly to GitHub Actions)
  3. Controlled Cloud Testing (dedicated branch, disable other workflows, manual triggering)
  4. Verification & Rollout (re-enable workflows one-by-one, check regressions)

---

## Platform-Specific Tool Versions

**NOTE**: All platforms now use modern tooling. Jetson ARM64 runs JetPack 5.0+ (Ubuntu 20.04, gcc-9.4, CUDA 11.4). JetPack 4.x is no longer supported due to GitHub Actions GLIBC requirements.

**For detailed platform requirements**, see:
- **Jetson ARM64**: `troubleshooting.jetson.md` → JetPack 5.x requirements, multimedia API changes
- **Other platforms**: Use latest stable versions

---

## Quick Start: First 2 Minutes

### Step 1: Identify Platform and Build Type

Check the failing workflow to determine platform and configuration:

```bash
# List recent workflow runs
gh run list --limit 10

# View specific run
gh run view <run-id>
```

**Extract from workflow name/logs:**
- Platform: Windows, Linux x64, Linux ARM64, Jetson, macOS
- Build type: CUDA vs NoCUDA
- Runner: GitHub-hosted (`windows-latest`, `ubuntu-latest`, `macos-latest`) vs self-hosted
- Phase: Phase 1 (prep/cache) vs Phase 2 (build/test) vs single-phase

### Step 2: Route to Correct Troubleshooting Guide

Use this decision tree:

```
Is this a CUDA build? (Check workflow name: *-CUDA.yml)
├─ YES → troubleshooting.cuda.md (all CUDA builds use self-hosted runners)
│   ├─ Windows CUDA → Self-hosted Windows + CUDA
│   ├─ Linux CUDA → Self-hosted Linux x64 + CUDA
│   └─ Jetson → Self-hosted ARM64 + CUDA (also see troubleshooting.jetson.md)
│
└─ NO (NoCUDA) → Which platform?
    ├─ Windows NoCUDA → troubleshooting.windows.md
    │   └─ GitHub-hosted, two-phase builds
    │
    ├─ Linux x64 NoCUDA → troubleshooting.linux.md
    │   └─ GitHub-hosted, two-phase builds
    │
    ├─ macOS NoCUDA → troubleshooting.macos.md
    │   └─ GitHub-hosted, two-phase builds
    │
    └─ Docker → troubleshooting.containers.md
        └─ Container-specific issues
```

### Step 3: Download Logs and Begin Diagnosis

```bash
# Download full logs (don't rely on UI truncation)
gh run view <run-id> --log > /tmp/build-<run-id>.log

# Search for errors
grep -i "error:" /tmp/build-<run-id>.log | head -20
grep "CMake Error" /tmp/build-<run-id>.log
grep "failed with" /tmp/build-<run-id>.log
```

**CRITICAL**: If step has `continue-on-error: true`, don't trust step status - check actual error messages!

---

## Cross-Platform Common Patterns

These issues can appear on ANY platform. Check these first before diving into platform-specific guides:

### Pattern 1: vcpkg Baseline Issues
**Symptoms:**
- `error: no version database entry for <package>`
- `error: failed to fetch ref <hash> from repository`
- Hash mismatch errors from GitLab/GitHub downloads

**Quick Check:**
```bash
# Verify baseline commit is fetchable
git ls-remote https://github.com/Apra-Labs/vcpkg.git | grep <baseline-hash>
```

**Fix**: See `reference.md` → vcpkg Baseline Management

---

### Pattern 2: Package Version Breaking Changes
**Symptoms:**
- Build fails after vcpkg baseline update
- API-related errors (e.g., "Unsupported SFML component: system")
- Type mismatch errors (e.g., `sf::Int16` not found)

**Root Cause**: Package upgraded to new major version with breaking changes

**Fix**: Pin package to compatible version in `base/vcpkg.json`:
```json
{
  "overrides": [
    { "name": "sfml", "version": "2.6.2" }
  ]
}
```

**See**: `reference.md` → Version Pinning Strategy

---

### Pattern 3: Python distutils Missing
**Symptoms:**
- `ModuleNotFoundError: No module named 'distutils'`
- Occurs when building glib or similar packages
- Python 3.12+ detected in logs

**Root Cause**: vcpkg using Python 3.12+ which removed distutils

**Fix**: Downgrade Python in `vcpkg/scripts/vcpkg-tools.json` to 3.10.11

**See**: `troubleshooting.windows.md` → Issue W1 (applies to all platforms)

> **Maintenance Note**: When updating cross-platform issue fixes (like Python distutils),
> update ALL relevant locations:
> - SKILL.md (this file) - Cross-Platform Patterns section
> - troubleshooting.windows.md - Issue W1 (detailed fix)
> - troubleshooting.linux.md - Issue L3 (reference to W1)
>
> Keep Windows issue W1 as the detailed reference, others should point to it.

---

### Pattern 4: Submodule Commit Not Fetchable
**Symptoms:**
- `fatal: remote error: upload-pack: not our ref <hash>`
- Occurs during vcpkg bootstrap or git submodule update

**Root Cause**: Committed detached HEAD or parent commit (not advertised by git)

**Quick Check:**
```bash
# Check if commit is advertised
cd vcpkg
git ls-remote origin | grep <commit-hash>
```

**Fix**: Create branch and push, or use advertised commit (branch tip/tag)

**See**: `reference.md` → vcpkg Fork Management

---

### Pattern 5: Cache Key Mismatch
**Symptoms:**
- Phase 2 shows "Cache not found" even after Phase 1 succeeded
- Build takes full time instead of using cache

**Root Cause**: Cache key changed between Phase 1 and Phase 2

**Quick Check**: Compare cache keys in Phase 1 save vs Phase 2 restore logs

**Fix**: Ensure cache key includes all relevant files:
```yaml
key: ${{ inputs.flav }}-5-${{ hashFiles('base/vcpkg.json', 'base/vcpkg-configuration.json', 'submodule_ver.txt') }}
```

**See**: `reference.md` → Cache Configuration

---

### Pattern 6: Duplicate Workflow Runs
**Symptoms:**
- Multiple workflow runs executing simultaneously for the same commit
- Wasted CI minutes (e.g., 3 runs × 40 minutes = 120 minutes wasted)
- Runs competing for shared runner resources, slowing each other down

**Root Cause**: Triggering `gh workflow run` multiple times in quick succession without waiting for confirmation

**How It Happens:**
- Executing workflow trigger command multiple times thinking it didn't work
- Using parallel tool calls with duplicate workflow triggers
- Not checking if a run already started before triggering again

**Quick Check:**
```bash
# Check for duplicate runs on the same branch/commit
gh run list --workflow=<workflow-name> --branch <branch-name> --limit 10
```

**Prevention Protocol:**
1. **Always wait for run ID** - `gh workflow run` returns a run ID; wait for it before re-executing
2. **Check before triggering** - Use `gh run list` to verify no existing run for the commit
3. **Use watch immediately** - After triggering, immediately run `gh run watch <run-id>` to confirm start
4. **Never trigger in parallel** - Don't use parallel tool calls for workflow triggers without explicit deduplication
5. **Cancel duplicates immediately** - If duplicates detected, cancel older runs: `gh run cancel <run-id>`

**Example Fix:**
```bash
# BAD: May trigger multiple times
gh workflow run CI-Linux-CUDA-Docker.yml --ref fix/branch  # Called 3 times accidentally

# GOOD: Check first, trigger once, watch immediately
gh run list --workflow=CI-Linux-CUDA-Docker.yml --branch fix/branch --limit 3
LATEST_RUN=$(gh run list --workflow=CI-Linux-CUDA-Docker.yml --branch fix/branch --limit 1 --json databaseId --jq '.[0].databaseId')
if [ -z "$LATEST_RUN" ] || [ "$(gh run view $LATEST_RUN --json status --jq '.status')" = "completed" ]; then
  NEW_RUN=$(gh workflow run CI-Linux-CUDA-Docker.yml --ref fix/branch --json 2>&1 | grep -oP 'https://github.com/.*/actions/runs/\K[0-9]+')
  gh run watch $NEW_RUN
fi
```

**Immediate Cleanup:**
```bash
# If duplicates found, cancel older runs (keep newest)
gh run cancel 19907395952 && gh run cancel 19907463211  # Keep 19907630652
```

---

## When to Use Each Guide

### Primary Guide Selection

| Build Configuration | Primary Guide | Secondary Guides |
|---------------------|---------------|------------------|
| Windows NoCUDA | troubleshooting.windows.md | reference.md |
| Windows CUDA | troubleshooting.cuda.md | troubleshooting.windows.md |
| Linux x64 NoCUDA | troubleshooting.linux.md | reference.md |
| Linux x64 CUDA | troubleshooting.cuda.md | troubleshooting.linux.md |
| macOS NoCUDA | troubleshooting.macos.md | troubleshooting.vcpkg.md, reference.md |
| Jetson/ARM64 | troubleshooting.jetson.md | troubleshooting.cuda.md |
| Docker builds | troubleshooting.containers.md | troubleshooting.cuda.md |

### Cross-Reference Usage

- **reference.md**: Always check for vcpkg, caching, version pinning knowledge
- **troubleshooting.cuda.md**: ANY CUDA-related issue, regardless of platform

---

## Tools & Prerequisites

### Required Tools
```bash
# GitHub CLI (required for all platforms)
gh --version

# Git (required for submodule management)
git --version

# Platform-specific package managers
# Windows: chocolatey
# Linux: apt/yum
# macOS: homebrew (future)
```

### Essential Commands

**GitHub Actions:**
```bash
# List workflows
gh workflow list

# Trigger workflow manually
gh workflow run <workflow-name>

# Monitor run
gh run watch <run-id>

# Download logs
gh run view <run-id> --log > build.log

# Cancel run
gh run cancel <run-id>
```

**vcpkg Diagnostics:**
```bash
# List installed packages
./vcpkg/vcpkg list

# Check package status
cat vcpkg_installed/vcpkg/status

# Verify baseline
cat base/vcpkg-configuration.json | grep baseline
```

**Log Analysis:**
```bash
# Find errors (case insensitive)
grep -i "error" build.log

# Find CMake errors
grep "CMake Error" build.log

# Find package failures
grep "error:" build.log | grep "package"

# Find specific issues
grep -i "distutils\|python" build.log
grep "unexpected hash" build.log
grep "PKG_CONFIG" build.log
```

---

## Best Practices & Guardrails

### ✅ DO

1. **Pin major versions** of all critical dependencies in vcpkg.json overrides
2. **Test baseline updates in isolation** - never update baseline directly in production
3. **Download full logs** - don't rely on GitHub Actions UI truncation
4. **Check actual errors** - not just workflow step status (beware `continue-on-error`)
5. **Verify commits are fetchable** - use `git ls-remote` before using as baseline
6. **Think logically** - before adding PATH fixes, verify tool actually needs PATH
7. **Document new patterns** - add to appropriate troubleshooting guide for future engineers

### ❌ DON'T

1. **Never modify master branches** of vcpkg forks - always use feature branches
2. **Don't fix application code** to accommodate new library versions - pin the library version instead
3. **Don't assume vcpkg changes apply immediately** - may need to clear cache or bump cache key
4. **Don't use parent commits as baselines** - they're not advertised by git
5. **Don't batch updates** - change one thing at a time for easier debugging
6. **Don't skip Phase 1** - ensure caching works before running Phase 2
7. **Don't trigger workflows multiple times** - check for existing runs first (see Pattern 6)

---

## Escalation Path

### When to Ask Human for Help

1. **Security decisions**: Accepting packages with known vulnerabilities
2. **Breaking changes**: When library upgrade requires code changes
3. **Infrastructure access**: Self-hosted runner configuration, credentials
4. **Policy decisions**: Should we support older library versions?
5. **Unknown patterns**: Errors not matching any documented pattern (document it!)

### How to Escalate

1. Document what you tried (commands, fixes attempted)
2. Include relevant log excerpts (not entire logs)
3. State current hypothesis about root cause
4. Ask specific question (not "it's broken, help!")

---

## Success Criteria

### How to Know Your Fix Worked

**Phase 1 (Prep) Success:**
- [ ] Step "Cache dependencies" shows cache saved
- [ ] Cache key logged in output
- [ ] No real errors in logs (ignore continue-on-error steps)

**Phase 2 (Build/Test) Success:**
- [ ] Cache restored from Phase 1 (check cache hit log)
- [ ] CMake configure completes without errors
- [ ] Build completes (cmake --build succeeds)
- [ ] Tests run and produce results (pass/fail is separate concern)
- [ ] Artifacts uploaded (logs, test results)

**Full Build Success:**
- [ ] Both Phase 1 and Phase 2 complete
- [ ] Total time < 2 hours for hosted runners
- [ ] Cache reusable for next build (key saved correctly)

---

## Troubleshooting Guide Index

- **troubleshooting.windows.md** - Windows NoCUDA builds (GitHub-hosted, two-phase)
- **troubleshooting.linux.md** - Linux x64 NoCUDA builds (GitHub-hosted, two-phase)
- **troubleshooting.macos.md** - macOS NoCUDA builds (GitHub-hosted, two-phase)
- **troubleshooting.cuda.md** - All CUDA builds (self-hosted, platform-agnostic)
- **troubleshooting.jetson.md** - Jetson ARM64 builds (ARM64 + CUDA constraints)
- **troubleshooting.containers.md** - Docker builds (container-specific)
- **troubleshooting.vcpkg.md** - vcpkg-specific issues (cross-platform)
- **reference.md** - Cross-platform reference (vcpkg, cache, GitHub Actions)
- **methodology.md** - High-level debugging methodology (detection, validation, testing)

---

## Maintenance

This skill should be updated when:
- New platform added (update decision tree, platform coverage, troubleshooting guides)
- New issue pattern discovered (add to appropriate troubleshooting guide)
- vcpkg baseline updated (update reference.md with new pins)
- Workflow structure changes (update reference.md)
- Self-hosted runner configuration changes (update troubleshooting.cuda.md)
