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
