# Path Types Enhancement Plan

> RFC for introducing first-class path types in the declarative pipeline framework

## Executive Summary

Currently, file and directory paths in module properties are typed as plain `string`, providing no semantic information about:
- Whether the path is a file, directory, or pattern
- Whether the path must exist (readers) or will be created (writers)
- How to validate and normalize the path

This plan introduces a **Path Type System** that enables:
1. Early validation of path existence at pipeline build time
2. Automatic path normalization (cross-platform separator handling)
3. Clear documentation of path expectations in module schemas
4. Better error messages for path-related issues

---

## Problem Statement

### Current State

```cpp
// FileWriterModule registration (current)
PropDef::string_("strFullFileNameWithPattern", PropMutability::Immutable)
    .required()
    .description("Output file path pattern with ???? wildcards")
```

**Issues:**
1. No way to know this is a path (not just any string)
2. No validation that parent directory exists
3. Path separator issues (`./data/testOutput\\file.bmp` on Windows)
4. Runtime failures instead of validation-time errors
5. Each module handles path normalization differently

### Desired State

```cpp
// FileWriterModule registration (proposed)
PropDef::filePattern("strFullFileNameWithPattern", PathRequirement::ParentMustExist)
    .required()
    .description("Output file path pattern with ???? wildcards")
```

**Benefits:**
1. Framework knows this is a file pattern
2. Validates parent directory exists at build time
3. Automatically normalizes path separators
4. Clear error: "Parent directory './data/testOutput' does not exist"
5. Centralized path handling in the framework

---

## Inventory: Modules with Path Properties

| Module | Property | Path Type | Requirement | Access |
|--------|----------|-----------|-------------|--------|
| FileReaderModule | strFullFileNameWithPattern | FilePattern | MustExist | Read |
| FileWriterModule | strFullFileNameWithPattern | FilePattern | ParentMustExist | Write |
| Mp4ReaderSource | videoPath | FilePath | MustExist | Read |
| Mp4WriterSink | baseFolder | DirectoryPath | WillBeCreated | Write |
| ThumbnailListGenerator | fileToStore | FilePath | ParentMustExist | Write |
| FacialLandmarkCV | faceDetectionConfig | FilePath | MustExist | Read |
| FacialLandmarkCV | faceDetectionWeights | FilePath | MustExist | Read |
| FacialLandmarkCV | landmarksModel | FilePath | MustExist | Read |
| FacialLandmarkCV | haarCascadeModel | FilePath | MustExist | Read |
| ArchiveSpaceManager | pathToWatch | DirectoryPath | MustExist | Read |
| AudioToTextXForm | modelPath | FilePath | MustExist | Read |

**Special cases (not filesystem paths):**
- RTSPClientSrc.rtspURL - Network URL, not a path
- VirtualCameraSink.device - Device path (special validation)

---

## Proposed Type System

### 1. Path Type Enum

```cpp
enum class PathType {
    NotAPath,        // Regular string, not a path
    FilePath,        // Single file: /path/to/file.mp4
    DirectoryPath,   // Directory: /path/to/folder/
    FilePattern,     // File with wildcards: frame_????.jpg
    GlobPattern,     // Glob pattern: *.mp4
    DevicePath,      // Device file: /dev/video0
    NetworkURL       // Network URL: rtsp://host/stream
};
```

### 2. Path Requirement Enum

```cpp
enum class PathRequirement {
    None,            // No validation (for NotAPath)
    MustExist,       // Path must exist at pipeline start
    MayExist,        // Path may or may not exist
    MustNotExist,    // Path must NOT exist (strict mode)
    ParentMustExist, // Parent directory must exist, file may not
    WillBeCreated    // Framework creates parent directories if needed
};
```

### 3. Extended PropDef

```cpp
struct PropDef {
    std::string name;
    std::string type;           // "string", "int", "double", "bool"
    std::string mutability;
    std::string default_value;

    // NEW: Path metadata
    PathType path_type = PathType::NotAPath;
    PathRequirement path_requirement = PathRequirement::None;

    // Factory methods for paths
    static PropDef filePath(const std::string& name, PathRequirement req);
    static PropDef directoryPath(const std::string& name, PathRequirement req);
    static PropDef filePattern(const std::string& name, PathRequirement req);
    // ... etc
};
```

---

## Implementation Plan

### Phase 1: Core Type System (Metadata.h)

**Files to modify:**
- `base/include/declarative/Metadata.h`

**Changes:**
1. Add `PathType` enum
2. Add `PathRequirement` enum
3. Add path metadata fields to `PropDef`
4. Add factory methods for path properties
5. Maintain backward compatibility (existing `string_()` still works)

**Example:**
```cpp
// New factory methods
static PropDef filePath(const std::string& name,
                        PathRequirement requirement = PathRequirement::MustExist) {
    PropDef def;
    def.name = name;
    def.type = "string";  // Still string at JSON level
    def.path_type = PathType::FilePath;
    def.path_requirement = requirement;
    return def;
}

static PropDef filePattern(const std::string& name,
                           PathRequirement requirement = PathRequirement::ParentMustExist) {
    PropDef def;
    def.name = name;
    def.type = "string";
    def.path_type = PathType::FilePattern;
    def.path_requirement = requirement;
    return def;
}
```

### Phase 2: Path Utilities

**Files to create:**
- `base/include/declarative/PathUtils.h`
- `base/src/declarative/PathUtils.cpp`

**Functions:**
```cpp
namespace apra {
namespace path_utils {

// Normalize path separators to platform-native format
std::string normalizePath(const std::string& path);

// Check if path exists (file or directory)
bool pathExists(const std::string& path);

// Check if path is a file
bool isFile(const std::string& path);

// Check if path is a directory
bool isDirectory(const std::string& path);

// Get parent directory of a path
std::string parentPath(const std::string& path);

// Create directory (and parents) if needed
bool createDirectories(const std::string& path);

// Expand pattern to check if any matching files exist
bool patternHasMatches(const std::string& pattern);

// Validate path based on requirement
struct PathValidationResult {
    bool valid;
    std::string error;
    std::string normalized_path;
};

PathValidationResult validatePath(
    const std::string& path,
    PathType type,
    PathRequirement requirement
);

} // namespace path_utils
} // namespace apra
```

### Phase 3: Update PipelineValidator

**Files to modify:**
- `base/src/declarative/PipelineValidator.cpp`

**New validation pass: Path Validation**

```cpp
void PipelineValidator::validatePaths(const PipelineDescription& desc) {
    for (const auto& [id, inst] : desc.modules) {
        auto* info = registry_.getModule(inst.type);
        if (!info) continue;

        for (const auto& propDef : info->properties) {
            if (propDef.path_type == PathType::NotAPath) continue;

            // Get property value
            auto it = inst.properties.find(propDef.name);
            if (it == inst.properties.end()) {
                // Use default if available
                if (propDef.default_value.empty()) continue;
                // ... handle default
            }

            std::string pathValue = /* extract from variant */;

            // Validate based on path type and requirement
            auto result = path_utils::validatePath(
                pathValue,
                propDef.path_type,
                propDef.path_requirement
            );

            if (!result.valid) {
                issues_.push_back(BuildIssue{
                    BuildIssue::Level::Error,
                    "PATH_" + pathRequirementCode(propDef.path_requirement),
                    id + "." + propDef.name,
                    result.error,
                    suggestPathFix(pathValue, propDef)
                });
            }
        }
    }
}
```

**Error codes:**
- `PATH_NOT_FOUND` - File/directory does not exist
- `PATH_NOT_FILE` - Expected file, found directory
- `PATH_NOT_DIR` - Expected directory, found file
- `PATH_PARENT_NOT_FOUND` - Parent directory does not exist
- `PATH_ALREADY_EXISTS` - File exists but MustNotExist
- `PATH_NO_PATTERN_MATCHES` - No files match pattern

### Phase 4: Update ModuleFactory

**Files to modify:**
- `base/src/declarative/ModuleFactory.cpp`

**Path normalization in property processing:**

```cpp
PropertyValue ModuleFactory::processProperty(
    const std::string& moduleId,
    const PropDef& propDef,
    const PropertyValue& value
) {
    // If it's a path property, normalize it
    if (propDef.path_type != PathType::NotAPath) {
        if (auto* strVal = std::get_if<std::string>(&value)) {
            std::string normalized = path_utils::normalizePath(*strVal);

            // For WillBeCreated, create parent directories
            if (propDef.path_requirement == PathRequirement::WillBeCreated) {
                std::string parent = path_utils::parentPath(normalized);
                if (!parent.empty() && !path_utils::pathExists(parent)) {
                    path_utils::createDirectories(parent);
                }
            }

            return normalized;
        }
    }
    return value;
}
```

### Phase 5: Update Module Registrations

**Files to modify:**
- `base/src/declarative/ModuleRegistrations.cpp`
- `base/include/declarative/modules/*.h` (Jetson modules)

**Example changes:**

```cpp
// BEFORE
REGISTER_MODULE(FileReaderModule)
    .category(ModuleCategory::Source)
    .prop(PropDef::string_("strFullFileNameWithPattern", PropMutability::Immutable)
        .required()
        .description("File path pattern with ???? wildcards"))
    // ...

// AFTER
REGISTER_MODULE(FileReaderModule)
    .category(ModuleCategory::Source)
    .prop(PropDef::filePattern("strFullFileNameWithPattern", PathRequirement::MustExist)
        .required()
        .description("File path pattern with ???? wildcards"))
    // ...
```

**All modules to update:**
1. FileReaderModule - `filePattern(..., MustExist)`
2. FileWriterModule - `filePattern(..., ParentMustExist)`
3. Mp4ReaderSource - `filePath(..., MustExist)`
4. Mp4WriterSink - `directoryPath(..., WillBeCreated)`
5. ThumbnailListGenerator - `filePath(..., ParentMustExist)`
6. FacialLandmarkCV (4 properties) - `filePath(..., MustExist)`
7. ArchiveSpaceManager - `directoryPath(..., MustExist)`
8. AudioToTextXForm - `filePath(..., MustExist)`

### Phase 6: Schema Export Update

**Files to modify:**
- `base/tools/schema_generator.cpp` (if exists)
- CLI `describe` command

**Enhanced schema output:**

```json
{
  "name": "FileWriterModule",
  "properties": [
    {
      "name": "strFullFileNameWithPattern",
      "type": "string",
      "pathType": "filePattern",
      "pathRequirement": "parentMustExist",
      "description": "Output file path pattern with ???? wildcards"
    }
  ]
}
```

---

## Backward Compatibility

1. **JSON format unchanged** - Paths are still strings in JSON
2. **Existing `PropDef::string_()` works** - Modules not yet updated continue to work
3. **Gradual migration** - Modules can be updated one at a time
4. **Validation opt-in** - Path validation only runs for properties with `path_type != NotAPath`

---

## Testing Strategy

### Unit Tests

```cpp
BOOST_AUTO_TEST_SUITE(PathUtilsTests)

BOOST_AUTO_TEST_CASE(NormalizePath_ForwardSlashes_Linux) {
    auto result = path_utils::normalizePath("./data/output/file.txt");
    // On Linux: "./data/output/file.txt"
    // On Windows: ".\\data\\output\\file.txt"
    BOOST_CHECK(/* platform appropriate */);
}

BOOST_AUTO_TEST_CASE(ValidatePath_MustExist_NotFound) {
    auto result = path_utils::validatePath(
        "/nonexistent/file.txt",
        PathType::FilePath,
        PathRequirement::MustExist
    );
    BOOST_CHECK(!result.valid);
    BOOST_CHECK(result.error.find("not found") != std::string::npos);
}

BOOST_AUTO_TEST_SUITE_END()
```

### Integration Tests

```cpp
BOOST_AUTO_TEST_CASE(Pipeline_PathValidation_MissingInput) {
    std::string json = R"({
        "modules": {
            "reader": {
                "type": "FileReaderModule",
                "props": {
                    "strFullFileNameWithPattern": "/nonexistent/????.raw"
                }
            }
        }
    })";

    auto result = JsonParser::parse(json);
    BOOST_CHECK(result.success);

    ModuleFactory factory;
    auto buildResult = factory.build(result.description);

    BOOST_CHECK(buildResult.hasErrors());
    BOOST_CHECK(buildResult.issues[0].code == "PATH_NOT_FOUND" ||
                buildResult.issues[0].code == "PATH_NO_PATTERN_MATCHES");
}
```

---

## Rollout Plan

1. **Phase 1-2**: Core types and utilities (no behavior change)
2. **Phase 3**: Validator with path checks (validation only, warnings first)
3. **Phase 4**: Factory path normalization (fixes Windows issue)
4. **Phase 5**: Update module registrations (gradual, one module at a time)
5. **Phase 6**: Schema export updates

---

## Open Questions

1. **Should path validation be strict or warn-only by default?**
   - Recommend: Error by default, with `--skip-path-validation` CLI flag

2. **How to handle relative vs absolute paths?**
   - Recommend: Relative paths resolved from working directory
   - Document that SDK examples use `./data/` relative to SDK root

3. **Should we auto-create directories for `WillBeCreated`?**
   - Recommend: Yes, with INFO-level log message

4. **How to handle network paths (UNC on Windows, SMB mounts)?**
   - Recommend: Treat as regular paths, let OS handle

5. **Pattern validation - check if ANY files match, or exact count?**
   - Recommend: For readers, at least one file must match
   - For writers, no validation (files don't exist yet)

---

## Success Criteria

1. **Windows FileWriterModule bug fixed** - Paths normalized correctly
2. **Clear error messages** - "File not found: /path/to/video.mp4" at validation
3. **No breaking changes** - Existing JSON pipelines work unchanged
4. **All 11 path properties updated** - With appropriate types and requirements
5. **Tests pass** - Unit and integration tests for path validation
6. **Documentation** - Module schemas show path type information
