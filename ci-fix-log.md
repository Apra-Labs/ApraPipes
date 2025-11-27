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

## Attempt #3 - 2025-11-27 07:25 UTC
**Branch:** fix/ci-windows-ak
**Run ID:** 19728449975
**Changes:** Added workflow step to install setuptools for vcpkg's Python 3.12.7
- Use `vcpkg fetch python3` to trigger Python download
- Install pip using get-pip.py
- Install setuptools to provide distutils shim
- Verify distutils import before CMake configure
**Status:** Queued → Monitoring
**Expected:** glib build should succeed with distutils available

---
