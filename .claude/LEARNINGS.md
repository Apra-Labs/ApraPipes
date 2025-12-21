# CI/CD Learnings — Institutional Memory

> This file survives /clear. Read it at session start. Append new learnings.
> Don't repeat mistakes documented here.

## Entry Format
```
### YYYY-MM-DD | workflow-name | PASS/FAIL
**Tried:** What was attempted
**Error:** Error message (if any)
**Root cause:** Why it happened
**Fix:** What resolved it
**Rule:** One-line principle for future
```

---

## Entries

(Append new learnings below this line)

### 2025-12-21 | CI-Linux-ARM64 | FAIL (CRITICAL INFRASTRUCTURE ISSUE)
**Tried:** Run workflow on Jetson ARM64 self-hosted runner
**Error:** `GLIBC_2.28' not found (required by /mnt/disks/actions-runner/externals/node20/bin/node)`
**Root cause:**
- Jetson runs Ubuntu 18.04 (GLIBC 2.27), but GitHub Actions Node 20 requires GLIBC 2.28+
- GitHub Actions Node16 reached END OF LIFE on November 12, 2024
- Runner 2.330.0 no longer includes Node16, only Node20
- Main passed on Dec 16 with runner 2.329.0 (last version with Node16)
- Ubuntu 18.04 is now UNSUPPORTED by GitHub Actions
**Fix options:**
1. **Upgrade Jetson to Ubuntu 20.04+** (recommended - has GLIBC 2.28+)
2. Pin runner to 2.329.0 or earlier (not recommended - no security updates)
3. Use Docker container for build (wraps workflow in container with newer GLIBC)
**Rule:** Ubuntu 18.04 self-hosted runners are no longer supported by GitHub Actions as of Nov 2024

### 2025-12-21 | ACTIONS_RUNNER_FORCED_INTERNAL_NODE_VERSION | FAIL
**Tried:** Set `ACTIONS_RUNNER_FORCED_INTERNAL_NODE_VERSION=node16` as job env var
**Error:** Runner still uses Node 20, ignores env var
**Root cause:** This env var must be set at the RUNNER SERVICE level, not workflow level. Also, Node16 was removed from runner 2.330.0 entirely.
**Fix:** This workaround no longer works as of runner 2.330.0
**Rule:** Workflow-level env vars cannot override runner's internal Node version

### 2025-12-21 | Release-only triplets | LESSON
**Tried:** Implement release-only vcpkg triplets to reduce build times
**Error:** CMake couldn't find baresip library - `find_library(BARESIP_LIB NAMES libbaresip.so ...)` failed
**Root cause:** Release-only triplets build static libraries (`.a`) by default, not shared (`.so`)
**Fix:** Updated find_library to accept both: `NAMES libbaresip.so libbaresip.a baresip`
**Rule:** When using release-only triplets, expect static libraries - update find_library calls accordingly

---