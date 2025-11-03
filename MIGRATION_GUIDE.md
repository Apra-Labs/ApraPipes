# Migration Guide: Component-Based Build System

This guide helps existing ApraPipes users migrate to the new component-based build system introduced in October 2025.

---

## What Changed?

ApraPipes now supports **selective component building** - you can build only the modules you need instead of always building everything. This reduces build times by 60-80% and simplifies dependency management.

### Before (Old System)
```bash
# Always built ALL components (~60-90 min)
build_windows_cuda.bat
./build_linux_cuda.sh
```

### After (New System)
```bash
# Choose what to build with presets
build_windows_cuda.bat --preset minimal    # ~5-10 min
build_windows_cuda.bat --preset video      # ~15-25 min
build_windows_cuda.bat --preset cuda       # ~30-40 min
build_windows_cuda.bat --preset full       # ~60-90 min (same as before)
```

---

## Backward Compatibility

**Good News:** The new system is **100% backward compatible**.

### Default Behavior (No Changes Required)

If you run build scripts **without any arguments**, you get the **same full build as before**:

```bash
# These commands still work exactly as before
build_windows_cuda.bat              # Builds ALL components
build_windows_no_cuda.bat           # Builds ALL non-CUDA components
./build_linux_cuda.sh               # Builds ALL components
./build_jetson.sh                   # Builds ALL Jetson components
```

**Result:** Your existing build processes, CI/CD pipelines, and scripts continue to work without modification.

---

## Migration Steps

### Step 1: Update Your Repository

```bash
git pull origin main  # or your branch name
git submodule update --init --recursive
```

### Step 2: Clean Previous Builds (Recommended)

```bash
# Windows
rmdir /s /q _build _debugbuild vcpkg\buildtrees vcpkg\packages vcpkg\installed

# Linux/Jetson
rm -rf _build _debugbuild vcpkg/buildtrees vcpkg/packages vcpkg/installed
```

**Why?** Component-based builds use different vcpkg features. Cleaning ensures a fresh start.

### Step 3: Choose Your Build Strategy

#### Option A: Continue with Full Builds (No Migration Needed)

```bash
# Keep using the same commands - no changes needed
build_windows_cuda.bat
./build_linux_cuda.sh
```

**When to use:** Production builds, comprehensive testing, backward compatibility.

#### Option B: Migrate to Component-Based Builds

```bash
# Use presets for faster builds
build_windows_cuda.bat --preset minimal   # Development: pipeline only
build_windows_cuda.bat --preset video     # Video processing projects
build_windows_cuda.bat --preset cuda      # GPU-accelerated projects
```

**When to use:** Development, testing specific features, faster iteration.

---

## Common Migration Scenarios

### Scenario 1: CI/CD Pipeline

**Before:**
```yaml
- name: Build ApraPipes
  run: ./build_linux_cuda.sh
```

**After (backward compatible):**
```yaml
- name: Build ApraPipes
  run: ./build_linux_cuda.sh  # Still works - builds ALL
```

**After (optimized for specific tests):**
```yaml
- name: Build Core Tests
  run: ./build_linux_cuda.sh --preset minimal

- name: Build Video Tests
  run: ./build_linux_cuda.sh --preset video
```

---

### Scenario 2: Development Workflow

**Before:**
```bash
# Had to wait 60-90 min for full build
build_windows_cuda.bat
```

**After:**
```bash
# Work on pipeline infrastructure? Build CORE only (~5-10 min)
build_windows_cuda.bat --preset minimal

# Work on video processing? Build VIDEO preset (~15-25 min)
build_windows_cuda.bat --preset video

# Need GPU features? Build CUDA preset (~30-40 min)
build_windows_cuda.bat --preset cuda

# Full regression testing? Build ALL (same as before)
build_windows_cuda.bat --preset full
```

---

### Scenario 3: Specialized Projects

**Before:**
```bash
# Had to build everything, even if only using specific modules
build_windows_cuda.bat  # 60-90 min, lots of unused dependencies
```

**After:**
```bash
# Face detection project
build_windows_cuda.bat --components "CORE;IMAGE_PROCESSING;WEBCAM;FACE_DETECTION"

# Audio transcription project
build_windows_cuda.bat --components "CORE;AUDIO"

# QR code reader
build_windows_no_cuda.bat --components "CORE;IMAGE_PROCESSING;QR"
```

---

## Understanding Components

The framework is now organized into 12 components:

| Component | Description | Typical Use Case |
|-----------|-------------|------------------|
| **CORE** | Pipeline infrastructure (always required) | All projects |
| **VIDEO** | Mp4, H264, RTSP | Video processing |
| **IMAGE_PROCESSING** | OpenCV CPU processing | Image manipulation |
| **CUDA_COMPONENT** | GPU acceleration | High-performance processing |
| **ARM64_COMPONENT** | Jetson hardware support | Jetson projects |
| **WEBCAM** | Camera capture | Live video applications |
| **QR** | QR code reading | QR code scanning |
| **AUDIO** | Audio capture & transcription | Audio applications |
| **FACE_DETECTION** | Face detection & landmarks | Computer vision |
| **GTK_RENDERING** | Linux GUI rendering | Linux visualization |
| **THUMBNAIL** | Thumbnail generation | Media previews |
| **IMAGE_VIEWER** | Image viewing GUI | Debugging/visualization |

See [COMPONENTS_GUIDE.md](COMPONENTS_GUIDE.md) for detailed module lists.

---

## Preset Reference

| Preset | Components | Build Time | Use Case |
|--------|-----------|------------|----------|
| `minimal` | CORE | ~5-10 min | Pipeline development, unit tests |
| `video` | CORE + VIDEO + IMAGE_PROCESSING | ~15-25 min | Video processing projects |
| `cuda` | video + CUDA_COMPONENT | ~30-40 min | GPU-accelerated projects |
| `full` | ALL | ~60-90 min | Production, comprehensive testing |

---

## Troubleshooting Migration Issues

### Issue 1: Build Fails with Missing Modules

**Symptom:**
```
error: undefined reference to `ImageViewerModule::ImageViewerModule`
```

**Cause:** You're using a module that's not in your selected components.

**Solution:**
```bash
# Option 1: Add the required component
build_windows_cuda.bat --components "CORE;IMAGE_PROCESSING;IMAGE_VIEWER"

# Option 2: Use a more comprehensive preset
build_windows_cuda.bat --preset full
```

### Issue 2: Tests Fail After Migration

**Symptom:** Some tests don't run or fail to compile.

**Cause:** Tests are now organized by component - only enabled components have their tests compiled.

**Solution:**
```bash
# Run tests for specific components
_build/RelWithDebInfo/aprapipesut.exe --run_test=core_tests/*

# Run ALL tests (build with --preset full first)
build_windows_cuda.bat --preset full
_build/RelWithDebInfo/aprapipesut.exe --run_test=*
```

### Issue 3: vcpkg Cache Issues

**Symptom:**
```
error: package conflicts or version mismatches
```

**Cause:** Old vcpkg cache from pre-component builds.

**Solution:**
```bash
# Windows
rmdir /s /q vcpkg\buildtrees vcpkg\packages vcpkg\installed

# Linux/Jetson
rm -rf vcpkg/buildtrees vcpkg/packages vcpkg/installed

# Rebuild
build_windows_cuda.bat --preset <your-preset>
```

### Issue 4: CUDA Build Requires Visual Studio 2019

**Symptom:**
```
error: unsupported Microsoft Visual Studio version
```

**Cause:** CUDA 11.8 requires Visual Studio 2019 (or VS 2022 v17.0-v17.3).

**Solution:**
```bash
# Use the VS 2019-specific script
.\build_windows_cuda_vs19.ps1 -Preset cuda

# Or install Visual Studio 2019
# https://visualstudio.microsoft.com/vs/older-downloads/
```

---

## Best Practices

### 1. Development: Use Minimal Builds

```bash
# Fast iteration during development
build_windows_cuda.bat --preset minimal
```

### 2. Testing: Match Component to Test Scope

```bash
# Testing video features? Use video preset
build_windows_cuda.bat --preset video

# Testing GPU features? Use cuda preset
build_windows_cuda.bat --preset cuda
```

### 3. Production: Use Full Builds

```bash
# Production releases - build everything
build_windows_cuda.bat --preset full
```

### 4. CI/CD: Matrix Testing

```yaml
strategy:
  matrix:
    preset: [minimal, video, cuda, full]
steps:
  - run: build_windows_cuda.bat --preset ${{ matrix.preset }}
```

---

## Getting Help

### View Available Options

```bash
# Windows
build_windows_cuda.bat --help
build_windows_no_cuda.bat --help
.\build_windows_cuda_vs19.ps1 -Help

# Linux/Jetson
./build_linux_cuda.sh --help
./build_linux_no_cuda.sh --help
./build_jetson.sh --help
```

### Documentation

- **Component Guide:** [COMPONENTS_GUIDE.md](COMPONENTS_GUIDE.md)
- **Dependency Diagram:** [COMPONENT_DEPENDENCY_DIAGRAM.md](COMPONENT_DEPENDENCY_DIAGRAM.md)
- **Build Instructions:** [README.md](README.md#quick-start-build-scripts-overview)
- **Refactoring Log:** [COMPONENT_REFACTORING_LOG.md](COMPONENT_REFACTORING_LOG.md)

### Report Issues

If you encounter migration issues:
1. Check [COMPONENTS_GUIDE.md](COMPONENTS_GUIDE.md) troubleshooting section
2. Review [COMPONENT_DEPENDENCY_DIAGRAM.md](COMPONENT_DEPENDENCY_DIAGRAM.md) for dependencies
3. Report issues on GitHub: [ApraPipes Issues](https://github.com/Apra-Labs/ApraPipes/issues)

---

## Summary

✅ **Backward Compatible:** Existing builds work without changes
✅ **Opt-In:** Use component builds when you want faster iterations
✅ **Flexible:** Choose from presets or custom component combinations
✅ **Documented:** Comprehensive guides and troubleshooting help

**Recommendation:** Start with your existing build process, then gradually adopt component-based builds for development workflows where faster iteration matters.
