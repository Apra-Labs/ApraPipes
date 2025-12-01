# Linux Build Troubleshooting

Platform-specific troubleshooting for Linux x64 NoCUDA builds on GitHub-hosted runners.

**Scope**: Linux x64 builds without CUDA, running on `ubuntu-latest` GitHub-hosted runners with two-phase build strategy.

**For Linux CUDA builds**: See `troubleshooting.cuda.md`

---

## Linux-Specific Architecture

### Build Configuration
- **Runner**: `ubuntu-latest` (GitHub-hosted)
- **Strategy**: Two-phase (Phase 1: prep/cache, Phase 2: build/test)
- **Time Limit**: 1 hour per phase
- **Disk Space**: Limited (~14 GB available)
- **Cache**: `~/.cache/vcpkg` or `${HOME}/.cache/vcpkg`

### Workflow Files
- **Top-level**: `.github/workflows/CI-Linux-x64-NoCUDA.yml`
- **Reusable**: `.github/workflows/build-test-linux.yml`

### Key Characteristics
- Bash-based scripts
- Uses apt/yum for system tools
- Similar two-phase architecture to Windows
- Platform-specific dependencies (GTK3, glib with libmount)

---

## Issue L1: pkg-config vs pkgconf Compatibility

**Symptom**:
```
Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)
```
- Similar to Windows Issue W2
- System pkg-config may not be compatible with CMake expectations

**Root Cause**:
- Linux may have system pkg-config installed
- CMake FindPkgConfig may expect different version/behavior
- vcpkg-provided pkgconf is more reliable

**Fix**:
Add `pkgconf` to vcpkg.json dependencies (same as Windows):
```json
{
  "dependencies": [
    "pkgconf",
    // ... other dependencies
  ]
}
```

---

## Issue L2: GTK3/glib Platform-Specific Dependencies

**Symptom**:
```
error: Package gtk3 is not available for platform x64-windows
error: glib requires feature 'libmount' on linux
```

**Root Cause**:
- GTK3 is Linux-only (excluded on Windows)
- glib has different feature requirements per platform

**Platform Filters**:
```json
{
  "name": "gtk3",
  "platform": "!windows"
},
{
  "name": "glib",
  "features": ["libmount"],
  "platform": "(linux & x64)"
},
{
  "name": "glib",
  "default-features": true,
  "platform": "windows"
}
```

**Fix**:
Ensure platform filters are correct in vcpkg.json.

---

## Issue L3: Python distutils Missing

**Symptom**:
Same as Windows Issue W1 - `ModuleNotFoundError: No module named 'distutils'`

**Fix**:
Same solution - downgrade Python to 3.10.11 in `vcpkg/scripts/vcpkg-tools.json`

**Note**: This is a cross-platform issue, see `troubleshooting.windows.md` Issue W1 for full details.

---

## Linux-Specific Quick Fixes Checklist

### Phase 1 (Prep) Checklist
- [ ] Python version is 3.10.x in vcpkg-tools.json
- [ ] pkgconf in vcpkg.json dependencies
- [ ] GTK3 platform filter: `"platform": "!windows"`
- [ ] glib platform filter correct for Linux x64
- [ ] Baseline commit is fetchable

### Phase 2 (Build/Test) Checklist
- [ ] Cache restored from Phase 1
- [ ] CMake finds all required packages
- [ ] Platform-specific dependencies installed
- [ ] Test execution completes

---

## To Be Expanded

This guide will be expanded as Linux-specific issues are encountered. Common patterns to document:
- System library conflicts (system OpenCV vs vcpkg OpenCV)
- Linux-specific compiler issues
- Library path issues (LD_LIBRARY_PATH)
- Bash script errors specific to Linux workflows

**Cross-Platform Issues**: See `troubleshooting.windows.md` for issues that apply to both platforms (Python distutils, vcpkg baseline, version breaking changes, etc.)

---

**Last Updated**: 2024-11-28
**Status**: Outline - expand as issues occur
**Applies to**: Linux x64 NoCUDA builds on GitHub-hosted runners
**Related Guides**: reference.md, troubleshooting.windows.md (cross-platform patterns)
