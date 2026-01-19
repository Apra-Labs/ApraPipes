# Declarative Pipeline - Progress Tracker

> Last Updated: 2026-01-19

**Branch:** `feat/sdk-packaging`

---

## Current Status

| Component | Status |
|-----------|--------|
| Core Infrastructure | ✅ Complete (Metadata, Registry, Factory, Validator, CLI) |
| JSON Parser | ✅ Complete (TOML removed) |
| Cross-platform Modules | ✅ 37 modules |
| CUDA Modules | ✅ 15 modules (NPP + NVCodec) |
| Jetson Modules | ✅ 8 modules (L4TM working via dlopen wrapper) |
| Node.js Addon | ✅ Complete (including Jetson) |
| Auto-Bridging | ✅ Complete (memory + pixel format) |
| SDK Packaging | ✅ Complete (all 4 platforms) |
| Path Types | ✅ Complete (first-class path type system) |
| Integration Tests | 🔄 In Progress (Windows fix pending CI verification) |

---

## Sprint 12: Windows Integration Test Fix (In Progress)

> Started: 2026-01-19

**Goal:** Fix Windows integration tests that fail with exit code 127.

### Problem Analysis

Windows integration tests fail with exit code 127 (CLI fails to launch) while Linux, macOS, and ARM64 all pass. Root cause analysis:

1. **Symptom**: CLI fails to execute with exit code 127 despite file existing
2. **Root Cause**: Git Bash PATH handling for DLL loading is problematic on Windows
3. **Why bash works on Linux/macOS**: Unix shells handle shared library paths natively
4. **Why bash fails on Windows**: PATH conversion from Unix-style to Windows-style doesn't always work correctly for DLL search paths

### Solution

Use PowerShell (pwsh) for Windows integration tests instead of bash:
- PowerShell uses native Windows PATH handling
- Properly sets up SDK bin and CUDA bin directories
- Includes debug output for diagnostics
- Linux/macOS continue to use bash (works correctly)

### Tasks

| Task | Status | Notes |
|------|--------|-------|
| Analyze CI failure logs | ✅ Complete | Exit code 127, DLL loading issue |
| Identify root cause | ✅ Complete | Git Bash PATH conversion |
| Implement PowerShell integration tests | ✅ Complete | In build-test.yml |
| Verify fix on CI | ⏳ Pending | Awaiting CI run results |
| Update documentation | ✅ Complete | This file |

---

## Sprint 11: Path Types Enhancement (Complete)

> Started: 2026-01-18 | Completed: 2026-01-18

**Goal:** Introduce first-class path types for file/directory path properties.

### Completed Tasks

| Task | Status | Notes |
|------|--------|-------|
| Add PathType enum | ✅ Complete | FilePath, DirectoryPath, FilePattern, GlobPattern, DevicePath, NetworkURL |
| Add PathRequirement enum | ✅ Complete | MustExist, MayExist, MustNotExist, ParentMustExist, WillBeCreated |
| Add PropDef path factories | ✅ Complete | FilePath(), DirectoryPath(), FilePattern(), etc. |
| Create PathUtils.h/.cpp | ✅ Complete | Validation, normalization, pattern matching |
| Update PipelineValidator | ✅ Complete | validatePaths() phase with warnings |
| Update ModuleFactory | ✅ Complete | Path normalization, directory creation |
| Update ModuleRegistrationBuilder | ✅ Complete | filePathProp(), directoryPathProp(), etc. |
| Update 12 module properties | ✅ Complete | See list below |

### Updated Module Properties

| Module | Property | Path Type | Requirement |
|--------|----------|-----------|-------------|
| FileReaderModule | strFullFileNameWithPattern | FilePattern | MustExist |
| FileWriterModule | strFullFileNameWithPattern | FilePattern | WillBeCreated |
| Mp4ReaderSource | videoPath | FilePath | MustExist |
| Mp4WriterSink | baseFolder | DirectoryPath | WillBeCreated |
| RTSPClientSrc | rtspURL | NetworkURL | None (no validation) |
| ThumbnailListGenerator | fileToStore | FilePath | WillBeCreated |
| FacialLandmarkCV | faceDetectionConfig | FilePath | MustExist |
| FacialLandmarkCV | faceDetectionWeights | FilePath | MustExist |
| FacialLandmarkCV | landmarksModel | FilePath | MustExist |
| FacialLandmarkCV | haarCascadeModel | FilePath | MustExist |
| ArchiveSpaceManager | pathToWatch | DirectoryPath | MustExist |
| AudioToTextXForm | modelPath | FilePath | MustExist |

### Key Features

1. **Path Types**: Semantic classification (FilePath, DirectoryPath, FilePattern, etc.)
2. **Path Requirements**: Existence and access expectations (MustExist, WillBeCreated, etc.)
3. **Early Validation**: Path issues detected at pipeline build time, not runtime
4. **Path Normalization**: Cross-platform separator handling via boost::filesystem
5. **Auto Directory Creation**: For `WillBeCreated` paths, parent directories are created
6. **Validation Warnings**: For readers with no matching files (not errors)
7. **Write Permission Checks**: Ensures directories are writable for writers

---

## Sprint 10: SDK Packaging (Complete)

> Started: 2026-01-17 | Completed: 2026-01-17

**Goal:** Create consistent SDK packaging across all 4 CI workflows.

### Completed Tasks

| Task | Status | Notes |
|------|--------|-------|
| Update CLAUDE.md | ✅ Complete | New mission |
| Reboot PROGRESS.md | ✅ Complete | Sprint 10 tracking |
| Reboot PROJECT_PLAN.md | ✅ Complete | Updated for SDK packaging |
| Enhance build-test.yml | ✅ Complete | Windows/Linux x64 SDK |
| Add SDK to build-test-macosx.yml | ✅ Complete | macOS SDK |
| Add SDK to build-test-lin.yml | ✅ Complete | ARM64 SDK + Jetson examples |
| Create docs/SDK_README.md | ✅ Complete | SDK usage documentation |

### SDK Structure (All Platforms)

```
aprapipes-sdk-{platform}/
├── bin/
│   ├── aprapipes_cli              # CLI tool
│   ├── aprapipesut                # Unit tests
│   ├── aprapipes.node             # Node.js addon
│   └── *.so / *.dll / *.dylib     # Shared libraries
├── lib/
│   └── *.a / *.lib                # Static libraries
├── include/
│   └── *.h                        # Header files
├── examples/
│   ├── basic/                     # JSON pipeline examples
│   ├── cuda/                      # CUDA examples (if applicable)
│   ├── jetson/                    # Jetson examples (ARM64 only)
│   └── node/                      # Node.js examples
├── data/
│   ├── frame.jpg                  # Sample input files
│   └── faces.jpg                  # For examples to work out of box
├── README.md                      # SDK usage documentation
└── VERSION                        # Version info
```

### SDK Artifacts by Platform

| Workflow | Artifact Name | Contents |
|----------|---------------|----------|
| CI-Windows | `aprapipes-sdk-windows-x64` | bin/, lib/, include/, examples/, data/, VERSION |
| CI-Linux | `aprapipes-sdk-linux-x64` | bin/, lib/, include/, examples/, data/, VERSION |
| CI-MacOSX | `aprapipes-sdk-macos-arm64` | bin/, lib/, include/, examples/, data/, VERSION |
| CI-Linux-ARM64 | `aprapipes-sdk-linux-arm64` | bin/, lib/, include/, examples/, data/, VERSION + jetson/ |

### Phase 2: GitHub Releases (Deferred)

| Task | Status | Notes |
|------|--------|-------|
| Create release.yml | ⏳ Deferred | Coordinated release workflow |
| Test release workflow | ⏳ Deferred | All 4 platforms |

---

## Completed Sprints

| Sprint | Theme | Key Deliverables |
|--------|-------|------------------|
| 11 | Path Types | First-class path type system, early validation |
| 10 | SDK Packaging | Consistent SDK across all 4 platforms |
| 9 | Node.js on Jetson | GCC 9 workaround, J2 resolved |
| 8 | Jetson Integration | 8 modules, L4TM dlopen wrapper |
| 7 | Auto-Bridging | PipelineAnalyzer, auto-insert CudaMemCopy/ColorConversion |
| 6 | DRY Refactoring | Fix defaults, type validation |
| 5 | CUDA | 15 modules, shared cudastream_sp |
| 4 | Node.js | @apralabs/aprapipes, event system |
| 1-3 | Core | Registry, Factory, Validator, CLI, 37 modules |

---

## Build Status

| Platform | Build | Node Addon | SDK Artifact |
|----------|-------|------------|--------------|
| macOS | ✅ | ✅ | ✅ aprapipes-sdk-macos-arm64 |
| Windows | ✅ | ✅ | ✅ aprapipes-sdk-windows-x64 |
| Linux x64 | ✅ | ✅ | ✅ aprapipes-sdk-linux-x64 |
| Linux x64 CUDA | ✅ | ✅ | ✅ aprapipes-sdk-linux-x64 |
| Jetson ARM64 | ✅ | ✅ | ✅ aprapipes-sdk-linux-arm64 |

---

## Documentation

| Document | Purpose |
|----------|---------|
| [SDK_README.md](../SDK_README.md) | SDK usage documentation |
| [SDK_PACKAGING_PLAN.md](./SDK_PACKAGING_PLAN.md) | SDK packaging plan |
| [PROJECT_PLAN.md](./PROJECT_PLAN.md) | Sprint overview |
| [JETSON_KNOWN_ISSUES.md](./JETSON_KNOWN_ISSUES.md) | Jetson platform issues |
| [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) | Module registration |
| [PIPELINE_AUTHOR_GUIDE.md](./PIPELINE_AUTHOR_GUIDE.md) | JSON authoring |
