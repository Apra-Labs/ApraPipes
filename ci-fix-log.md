# CI Fix Log - Windows NoCUDA Build

## Objective
Fix the Windows NoCUDA build that has been failing for months. Multiple engineers and agents have attempted fixes.

## Strategy
1. Fast-fail approach: Reduce vcpkg dependencies to minimal set to identify root cause quickly
2. Monitor builds every 5 minutes
3. Iteratively fix errors and re-trigger
4. Document each attempt with findings

---

## Attempt #1 - 2025-11-27 06:24 UTC
**Branch:** fix/ci-windows-ak
**Run ID:** 19727203910
**Changes:** Added Python 3.10.11 installation to prep-cmd, added python --version verification
**Status:** Running (monitoring...)
**Expected:** Will likely fail with same libxml2 hash mismatch (old branch, Python fix not applied yet)

## Root Cause Analysis
**Primary Issue:** libxml2 download hash mismatch from GitLab (vcpkg baseline issue)
- Error: `libxml2:x64-windows` fails with unexpected hash from gitlab.gnome.org
- Expected hash: 289d8e30...
- Actual hash: 4b3d7af2...
- This is NOT a Python issue - it's a stale vcpkg baseline

**Strategy:**
1. Update vcpkg submodule to latest to get fresh baseline
2. Implement fast-fail: temporarily remove glib dependency (pulls in libxml2) to test other deps
3. Once core build works, add glib back with updated vcpkg

---

## Attempt #2 - 2025-11-27 06:34 UTC
**Branch:** fix/ci-windows-ak
**Run ID:** 19727403084
**Changes:** Updated vcpkg submodule from 6ba64191 (May 14) to be563dfee8 (Nov 27 latest)
**Status:** ✅ Prep phase SUCCESS! → Now in build-test phase
**Result:** libxml2 issue FIXED by vcpkg update. Both builds (old Python & new Python) passed prep phase.

## Critical Discovery - Python Version Conflict
vcpkg downloads its own Python 3.12.7 (defined in vcpkg-tools.json). This conflicts with our Python 3.10 installation attempt. Python 3.12 removed distutils, which explains all the previous distutils-related fix attempts in git history.

**New Strategy Needed:**
- Option A: Let vcpkg use Python 3.12.7 and fix distutils compatibility (install setuptools)
- Option B: Force vcpkg to use system Python 3.10 (may require vcpkg-tools.json modification)
- Option C: Use vcpkg's Python 3.12.7 but ensure setuptools is installed for it

## Attempt #1 Result - 2025-11-27 07:22 UTC (FAILED)
**Run ID:** 19727203910
**Result:** ❌ FAILED in build-test phase during glib build
**Error:** `ModuleNotFoundError: No module named 'distutils'` when Python 3.12.7 tries to build glib
**Root Cause:** vcpkg's Python 3.12.7 lacks distutils (removed in Python 3.12). glib build scripts require it.
**Fix Needed:** Install setuptools to provide distutils shim for vcpkg's Python 3.12.7

## Attempt #3 - 2025-11-27 07:25 UTC (FAILED)
**Branch:** fix/ci-windows-ak
**Run ID:** 19728449975
**Changes:** Added workflow step to install setuptools for vcpkg's Python 3.12.7
**Result:** ❌ FAILED - vcpkg fetch python3 didn't download Python, returned system py.exe instead
**Lesson:** Wrong approach - patching Python 3.12 is harder than using older Python with distutils

## Attempt #2 Result - 2025-11-27 07:34 UTC (FAILED)
**Run ID:** 19727403084
**Result:** ❌ FAILED after 1 hour - same distutils error as build #1
**Error:** Same glib build failure with Python 3.12.7 distutils issue

## Attempt #4 - 2025-11-27 15:12 UTC (FAILED)
**Branch:** fix/ci-windows-ak
**Run ID:** 19740723976
**Result:** ❌ FAILED - vcpkg submodule commit not fetchable (detached HEAD not pushed)
**Error:** `fatal: remote error: upload-pack: not our ref bf0c600c6a`
**Fix:** Created branch `python-3.10-downgrade` in Apra-Labs/vcpkg and pushed

## Attempt #5 - 2025-11-27 15:18 UTC (FAILED)
**Branch:** fix/ci-windows-ak
**Run ID:** 19740916828
**Changes:** Same Python 3.10.11 changes in vcpkg-tools.json, vcpkg commit now on pushed branch
**Result:** ❌ FAILED - TWO critical issues found:

### Issue 1: Python 3.10 downgrade DID NOT WORK
- vcpkg-tools.json was modified to specify Python 3.10.11
- Build STILL downloaded and used Python 3.12.7: `python-3.12.7-x64-1\python.exe`
- Same distutils error: `ModuleNotFoundError: No module named 'distutils'`
- **Root Cause**: vcpkg might be caching the tools.json or pulling from a different source

### Issue 2: libxml2 hash mismatch STILL FAILING
- Phase 1 (prep): libxml2 download from gitlab.gnome.org still has wrong hash
- Expected: 289d8e30a894a3efde78e06d1cedadc1491f4abdc2c0b653bb5410be48338aacec29e0ca23e1f1cac2725fd4e2114a8d20dcdaa80cf7f5b34302342d6f5efe10
- Actual: eeb5e896c76f7a72c84a7afa31eff70effd39b9091e0c662539d994885f4ad24fafab6e1dfdcad39ae124c8c7e8abb11732628fba31b0b45bce2d0d7afbb4dc0
- **Root Cause**: vcpkg registry still pointing to Apra-Labs/vcpkg fork with old libxml2 portfile
- Build logs show: `git+https://github.com/Apra-Labs/vcpkg.git@caa5f663ba4c26ac2402c6aaa56781bd262fc05e`
- This commit (caa5f663) is in Apra-Labs fork, NOT microsoft/vcpkg

### Analysis:
1. The vcpkg submodule update to microsoft/vcpkg master didn't help because vcpkg registry is configured to use Apra-Labs fork
2. The Python downgrade in vcpkg-tools.json was ignored - vcpkg still downloaded Python 3.12.7
3. Need to investigate why vcpkg-tools.json changes aren't being applied

---

## Attempt #10 - 2025-11-28 00:48 UTC (FAILED)
**Branch:** fix/ci-windows-ak
**Run ID:** 19750941275
**Changes:** Added pkgconf to minimal vcpkg.json (only pkgconf + glib)
**Result:** ❌ FAILED after 28 minutes
**Error:** `Could NOT find Boost (missing: Boost_INCLUDE_DIR system thread filesystem serialization log chrono unit_test_framework)`

### Root Cause Analysis:
The "minimal" vcpkg.json was TOO minimal. It only included:
- pkgconf (for CMake FindPkgConfig) ✅
- glib (to test libxml2 hash + Python distutils) ✅

However, base/CMakeLists.txt line 42 requires ALL of these packages with REQUIRED flag:
- PkgConfig
- Boost (system, thread, filesystem, serialization, log, chrono, unit_test_framework)
- JPEG
- OpenCV
- BZip2, ZLIB, LibLZMA
- FFMPEG
- ZXing
- bigint

### Key Lesson - continue-on-error Flag Confusion:
The PKG_CONFIG_EXECUTABLE error appeared in BOTH phases:
1. **Prep phase**: Error shown but step has `continue-on-error: true` (expected to fail)
2. **Test phase**: Would fail here if CMake config failed

The ACTUAL blocking error was NOT PKG_CONFIG - it was the missing Boost dependency. The prep phase PKG_CONFIG error was a red herring because that step is designed to fail partially.

### Fix Applied:
Restored full vcpkg.json.full-backup and added pkgconf as first dependency. This manifest includes all required packages that CMakeLists.txt needs.

### Status:
Ready to test with full manifest + pkgconf addition. This should properly test:
1. libxml2 hash fix (via glib dependency)
2. Python 3.10 distutils (via glib build)
3. PKG_CONFIG_EXECUTABLE (via pkgconf package)
4. All other required dependencies for complete build

---
