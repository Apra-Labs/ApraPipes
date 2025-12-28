# Linux Build Troubleshooting

Platform-specific troubleshooting for Linux x64 builds on GitHub-hosted runners.

**Scope**: Linux x64 builds on cloud runners (CI-Linux build job), running on `ubuntu-22.04` with CUDA toolkit installed.

**For GPU-specific tests**: See `troubleshooting.cuda.md` (CI-Linux cuda job on self-hosted runners)
**For Docker builds**: See `troubleshooting.containers.md` (CI-Linux docker job)

---

## Linux-Specific Architecture

### Build Configuration
- **Workflow**: CI-Linux.yml
- **Job**: build (runs on cloud runners)
- **Runner**: `ubuntu-22.04` (GitHub-hosted)
- **Strategy**: Single-phase build with CUDA toolkit installed
- **Time Limit**: 6 hours
- **Disk Space**: Limited (~14 GB available after cleanup)
- **Cache**: `/mnt/runner-work/.cache/vcpkg`

### Workflow Files
- **Top-level**: `.github/workflows/CI-Linux.yml`
- **Reusable**: `.github/workflows/build-test.yml` (unified with Windows)

### Key Characteristics
- PowerShell-based scripts (consistent with Windows)
- Uses apt for system tools
- CUDA toolkit installed for compilation (GPU tests run separately)
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

### Build Job Checklist
- [ ] Disk cleanup succeeded (check available space)
- [ ] Python version is 3.10.x in vcpkg-tools.json
- [ ] pkgconf in vcpkg.json dependencies
- [ ] GTK3 platform filter: `"platform": "!windows"`
- [ ] glib platform filter correct for Linux x64
- [ ] Baseline commit is fetchable
- [ ] Cache restored from previous build
- [ ] CMake finds all required packages
- [ ] Platform-specific dependencies installed
- [ ] Cloud tests execution completes (GPU tests run separately)

---

## To Be Expanded

This guide will be expanded as Linux-specific issues are encountered. Common patterns to document:
- System library conflicts (system OpenCV vs vcpkg OpenCV)
- Linux-specific compiler issues
- Library path issues (LD_LIBRARY_PATH)
- Bash script errors specific to Linux workflows

**Cross-Platform Issues**: See `troubleshooting.windows.md` for issues that apply to both platforms (Python distutils, vcpkg baseline, version breaking changes, etc.)

---

**Applies to**: CI-Linux build job (cloud runners with CUDA toolkit)
**Related Guides**: reference.md, troubleshooting.windows.md (cross-platform patterns), troubleshooting.cuda.md (GPU tests), troubleshooting.containers.md (Docker builds)
