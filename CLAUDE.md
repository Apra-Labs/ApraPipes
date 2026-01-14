# CLAUDE.md - Declarative Pipeline Construction

> Instructions for Claude Code agents working on the ApraPipes Declarative Pipeline feature.

**Git Branch:** `feat-declarative-pipeline-v2`

---

## MANDATORY: Build and Test Before Commit

**This is non-negotiable. NEVER commit code without verifying build and tests pass.**

```
1. Write code
2. BUILD - must succeed
3. TEST - must pass
4. ONLY THEN commit
```

**If build directory doesn't exist or build fails:**
- Configure and build first: `cmake -B build -S . && cmake --build build -j$(nproc)`
- If build cannot be completed (disk full, missing deps, etc.), DO NOT COMMIT
- Tell the user the code is ready but untested, and wait for them to build/test

**If tests fail:**
- Fix the code
- Re-run tests
- DO NOT COMMIT until tests pass

**Verification commands before ANY commit:**
```bash
# 1. Build must succeed
cmake --build build -j$(nproc)

# 2. Tests must pass (at minimum, run relevant test suite)
./build/aprapipesut --run_test="<RelevantSuite>/*" --log_level=test_suite

# 3. For CLI/runtime changes, also run a quick smoke test
./build/aprapipes_cli run <some_example.json>
```

**NO EXCEPTIONS. Untested commits waste everyone's time.**

---

## 🚨 Jetson Device Rules (CRITICAL)

When working with the Jetson device (ssh akhil@192.168.1.18):
- **NEVER** delete or modify `/data/action-runner/` - used by GitHub Actions
- **NEVER** delete `/data/.cache/` en-mass - vcpkg cache shared with CI
- **ALWAYS** work in `/data/ws/` for development
- Disable CI-Linux-ARM64.yml before pushing to avoid resource competition

---

## 🔍 Code Review Before Every Commit (MANDATORY)

Before EVERY git commit (whether on local machine or Jetson device):

1. **Run `git diff --staged`** and review EVERY changed line
2. **Justify each change** - ask "why is this line changing?"
3. **Remove unintended changes** with `git checkout -- <file>` or `git reset HEAD <file>`
4. **Check for forbidden content:**
   - No debug code (console.log, printf debugging, etc.)
   - No temporary hacks or workarounds
   - No commented-out code
   - No unrelated files staged
5. **Only commit when 100% confident** all changes are intentional and correct

```bash
# Code review checklist before commit
git diff --staged                    # Review all changes
git diff --staged --stat             # Check which files changed
git status                           # Verify nothing unexpected staged
```

**If you find unwanted changes:**
```bash
git reset HEAD <file>                # Unstage entire file
git checkout -- <file>               # Discard file changes entirely
git add -p <file>                    # Stage only specific hunks
```

---

## 🎯 Project Goal

Transform ApraPipes from imperative C++ construction to declarative JSON configuration. Users should be able to write:

```json
{
  "modules": {
    "source": {
      "type": "FileReaderModule",
      "props": {
        "path": "/video.mp4"
      }
    }
  }
}
```

Instead of:
```cpp
auto source = boost::shared_ptr<Module>(new FileReaderModule(props));
```

---

## 📍 First Things First - Check Current State

**Before doing anything else, run these commands to understand where the project stands:**

```bash
# 1. Check progress tracking file
cat docs/declarative-pipeline/PROGRESS.md

# 2. Check git log for recent work
git log --oneline -20 --all | head -20

# 3. Check what's implemented in declarative/
ls -la base/include/declarative/ 2>/dev/null || echo "Directory not created yet"
ls -la base/src/declarative/ 2>/dev/null || echo "Directory not created yet"

# 4. Check test status (if build exists) - uses Boost.Test
./build/aprapipesut --run_test="ModuleRegistryTests/*" --log_level=test_suite

# 5. Check GitHub issues (if gh cli available)
gh issue list --label "declarative-pipeline" --state open
```

---

## 📋 Task Specifications

All task specs are in `docs/declarative-pipeline/tasks/`. **Always read the full spec before starting.**

| Task | Spec File | Status Check |
|------|-----------|--------------|
| A1 | `A1-core-metadata-types.md` | `ls base/include/declarative/Metadata.h` |
| A2 | `A2-module-registry.md` | `ls base/include/declarative/ModuleRegistry.h` |
| A3 | `A3-frame-type-registry.md` | `ls base/include/declarative/FrameTypeRegistry.h` |
| B1 | `B1-pipeline-description-ir.md` | `ls base/include/declarative/PipelineDescription.h` |
| J1 | `J1-json-parser.md` | `ls base/include/declarative/JsonParser.h` |
| C1 | `C1-validator-shell.md` | `ls base/include/declarative/PipelineValidator.h` |
| D1 | `D1-module-factory.md` | `ls base/include/declarative/ModuleFactory.h` |
| E1 | `E1-cli-tool.md` | `ls base/tools/aprapipes_cli.cpp` |
| E2 | `E2-schema-generator.md` | `ls base/tools/schema_generator.cpp` |

---

## 🔄 Standard Workflow

### 1. Determine Next Task

```bash
# Read progress file
cat docs/declarative-pipeline/PROGRESS.md

# Check dependency graph in task index
cat docs/declarative-pipeline/tasks/README.md
```

**Critical Path:** A1 → A2 → D1 → E1 (with B1 → J1 in parallel)

Pick the **first incomplete task** whose dependencies are complete.

### 2. Read the Full Specification

```bash
# Example for task A1
cat docs/declarative-pipeline/tasks/A1-core-metadata-types.md
```

Note these sections:
- **Files** - exact paths to create
- **Implementation Notes** - code patterns, gotchas
- **Acceptance Criteria** - must all pass before marking complete
- **Definition of Done** - checklist

### 3. Create/Modify Files

Follow the spec exactly. Use the file paths specified.

```bash
# Example structure for core infrastructure
base/
├── include/
│   └── core/
│       ├── Metadata.h           # A1
│       ├── PipelineDescription.h # B1
│       ├── ModuleRegistry.h     # A2
│       ├── FrameTypeRegistry.h  # A3
│       ├── PipelineValidator.h  # C1
│       └── ModuleFactory.h      # D1
├── src/
│   └── core/
│       ├── PipelineDescription.cpp
│       ├── ModuleRegistry.cpp
│       ├── FrameTypeRegistry.cpp
│       ├── PipelineValidator.cpp
│       └── ModuleFactory.cpp
└── test/
    ├── metadata_tests.cpp
    ├── pipeline_description_tests.cpp
    ├── module_registry_tests.cpp
    ├── frame_type_registry_tests.cpp
    ├── pipeline_validator_tests.cpp
    └── module_factory_tests.cpp
```

### 4. Write Unit Tests FIRST (TDD Encouraged)

```bash
# Create test file if it doesn't exist
# Tests go in base/test/<feature>_tests.cpp
```

**Test file template:**
```cpp
#include <boost/test/unit_test.hpp>
#include "core/Metadata.h"  // or whatever you're testing

BOOST_AUTO_TEST_SUITE(MetadataTests)

BOOST_AUTO_TEST_CASE(PinDef_Construction) {
    // Arrange
    // Act
    // Assert
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END()
```

### 5. Build and Test

```bash
# Configure if needed (first time or CMakeLists.txt changed)
cmake -B build -S . -DENABLE_CUDA=OFF  # adjust flags as needed

# Build
cmake --build build --parallel

# Run specific test suite (Boost.Test)
./build/aprapipesut --run_test="MetadataTests/*" --log_level=test_suite

# Run all tests
./build/aprapipesut --log_level=test_suite
```

### 6. Update Progress Tracking

**CRITICAL: Update these before ending your session!**

```bash
# Update progress file
# Mark task complete, note any issues, update timestamp
```

### 7. Commit with Conventional Message

**STOP! Before committing, verify:**
1. `cmake --build build` - Did it succeed?
2. `./build/aprapipesut --run_test="<Suite>/*"` - Did tests pass?
3. If NO to either, DO NOT COMMIT. Fix the issues first.

```bash
# Only after build and tests pass:
git add -A
git commit -m "feat(declarative): implement A1 Core Metadata Types

- Add PinDef, PropDef, AttrDef structs
- Add ModuleCategory enum
- Add factory methods for property types
- Add unit tests

Task: A1
Status: Complete
Refs: #<issue_number>"

git push
```

### 8. Update GitHub Issue (if gh cli available)

```bash
# Add comment with progress
gh issue comment <number> --body "Task A1 completed. All acceptance criteria met. See commit <sha>"

# Close if fully complete
gh issue close <number>
```

---

## 📊 Progress Tracking

### Primary: PROGRESS.md File

Location: `docs/declarative-pipeline/PROGRESS.md`

**Create this file if it doesn't exist:**

```markdown
# Declarative Pipeline - Progress Tracker

Last Updated: YYYY-MM-DD HH:MM by Claude Code Agent

## Current Sprint: 1

## Task Status

| Task | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| A1 | ✅ Complete | 2024-01-15 | 2024-01-15 | All tests pass |
| B1 | ✅ Complete | 2024-01-15 | 2024-01-16 | |
| A2 | 🔄 In Progress | 2024-01-16 | - | REGISTER_MODULE macro WIP |
| B2 | ⏳ Blocked | - | - | Waiting on B1 |
| A3 | 📋 Ready | - | - | Can start (A1 done) |

## Legend
- ✅ Complete
- 🔄 In Progress  
- ⏳ Blocked (waiting on dependency)
- 📋 Ready (dependencies met, not started)
- ❌ Blocked (other issue)

## Current Work

### Active Task: A2 - Module Registry
- [x] Created ModuleRegistry.h
- [x] Created ModuleRegistry.cpp
- [x] Implemented singleton
- [ ] Implemented REGISTER_MODULE macro
- [ ] Added query methods
- [ ] Unit tests

### Blockers
None currently.

### Notes for Next Session
- REGISTER_MODULE macro needs static initialization trick
- See existing pattern in Logger.h for singleton

## Build Status
- Last successful build: 2024-01-16 14:30
- Test results: 45 passed, 0 failed

## Files Modified This Session
- base/include/core/ModuleRegistry.h (created)
- base/src/core/ModuleRegistry.cpp (created)
- base/test/module_registry_tests.cpp (created)
- base/CMakeLists.txt (modified - added new files)
```

### Secondary: Git Commits

Use structured commit messages:
```
feat(declarative): <short description>

<detailed description>

Task: <task_id>
Status: <Complete|In Progress|Blocked>
Refs: #<issue_number>
```

### Tertiary: GitHub Issues

If `gh` CLI is available:
```bash
# Check issue status
gh issue view <number>

# Add progress comment
gh issue comment <number> --body "Progress update:
- Completed: X, Y, Z
- Remaining: A, B
- Blockers: None"

# Update labels
gh issue edit <number> --add-label "in-progress"
gh issue edit <number> --remove-label "ready"
```

---

## 🏗️ Project Structure

```
ApraPipes/
├── CLAUDE.md                  # THIS FILE - Agent instructions
├── .claude/
│   └── settings.json          # Claude Code hooks & tasks (auto-loaded)
├── base/
│   ├── include/
│   │   ├── Module.h           # Existing
│   │   ├── FrameMetadata.h    # Existing
│   │   └── declarative/       # ★ NEW - All declarative pipeline headers
│   │       ├── Metadata.h           # A1
│   │       ├── PipelineDescription.h # B1
│   │       ├── ModuleRegistry.h     # A2
│   │       ├── FrameTypeRegistry.h  # A3
│   │       ├── PipelineValidator.h  # C1
│   │       ├── ModuleFactory.h      # D1
│   │       └── JsonParser.h         # J1
│   ├── src/
│   │   ├── Module.cpp         # Existing
│   │   └── declarative/       # ★ NEW - All declarative pipeline sources
│   │       ├── PipelineDescription.cpp
│   │       ├── ModuleRegistry.cpp
│   │       ├── FrameTypeRegistry.cpp
│   │       ├── PipelineValidator.cpp
│   │       ├── ModuleFactory.cpp
│   │       └── JsonParser.cpp
│   ├── test/
│   │   ├── existing_tests.cpp # Existing
│   │   └── declarative/       # ★ NEW - All declarative pipeline tests
│   │       ├── metadata_tests.cpp
│   │       ├── pipeline_description_tests.cpp
│   │       ├── module_registry_tests.cpp
│   │       ├── frame_type_registry_tests.cpp
│   │       ├── json_parser_tests.cpp
│   │       ├── pipeline_validator_tests.cpp
│   │       └── module_factory_tests.cpp
│   ├── tools/
│   │   ├── aprapipes_cli.cpp        # E1 - CLI tool
│   │   └── schema_generator.cpp     # E2 - Schema export
│   ├── vcpkg.json             # Add: nlohmann-json
│   └── CMakeLists.txt         # Add declarative sources
├── docs/
│   └── declarative-pipeline/
│       ├── README.md
│       ├── RFC.md
│       ├── PROJECT_PLAN.md
│       ├── PROGRESS.md        # ★ UPDATE THIS EVERY SESSION
│       └── tasks/
│           └── *.md           # Full task specifications
└── setup_github_project.sh    # Creates GitHub issues
```

### Include Pattern

```cpp
// From existing modules (when adding REGISTER_MODULE):
#include "declarative/Metadata.h"
#include "declarative/ModuleRegistry.h"

// Inside declarative/ code:
#include "Metadata.h"  // Relative include within declarative/
#include "ModuleRegistry.h"
```

---

## 🔧 CMake Integration

When adding new files, update `base/CMakeLists.txt`:

```cmake
# ═══════════════════════════════════════════════════════════════════
# Declarative Pipeline Sources
# ═══════════════════════════════════════════════════════════════════

# Create the declarative subdirectories if first time
# mkdir -p base/include/declarative base/src/declarative base/test/declarative

set(DECLARATIVE_HEADERS
    include/declarative/Metadata.h
    include/declarative/PipelineDescription.h
    include/declarative/ModuleRegistry.h
    include/declarative/FrameTypeRegistry.h
    include/declarative/PipelineValidator.h
    include/declarative/ModuleFactory.h
    include/declarative/JsonParser.h
)

set(DECLARATIVE_SOURCES
    src/declarative/PipelineDescription.cpp
    src/declarative/ModuleRegistry.cpp
    src/declarative/FrameTypeRegistry.cpp
    src/declarative/PipelineValidator.cpp
    src/declarative/ModuleFactory.cpp
    src/declarative/JsonParser.cpp
)

# Add to main library sources
target_sources(aprapipes PRIVATE ${DECLARATIVE_SOURCES})

# Include directories - both base/include and base/include/declarative
target_include_directories(aprapipes PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/declarative
)

# ═══════════════════════════════════════════════════════════════════
# Declarative Pipeline Tests
# ═══════════════════════════════════════════════════════════════════

set(DECLARATIVE_TESTS
    test/declarative/metadata_tests.cpp
    test/declarative/pipeline_description_tests.cpp
    test/declarative/module_registry_tests.cpp
    test/declarative/frame_type_registry_tests.cpp
    test/declarative/json_parser_tests.cpp
    test/declarative/pipeline_validator_tests.cpp
    test/declarative/module_factory_tests.cpp
)

# Add test executable (or add to existing test target)
add_executable(declarative_tests ${DECLARATIVE_TESTS})
target_link_libraries(declarative_tests aprapipes Boost::unit_test_framework)
add_test(NAME declarative_tests COMMAND declarative_tests)
```

For vcpkg dependencies, add to `base/vcpkg.json`:
```json
{
  "dependencies": [
    "nlohmann-json"
  ]
}
```

### Creating Directories (First Time Setup)

```bash
# Run once when starting the project
mkdir -p base/include/declarative
mkdir -p base/src/declarative
mkdir -p base/test/declarative
```



---

## 🤖 Claude Code Integration

The `.claude/settings.json` provides permissions and hooks for Claude Code agents.

### Configured Permissions

Common build/dev commands are pre-approved:
- `git`, `cmake`, `make`, `ninja`
- `gh issue`, `gh pr`, `gh project`
- File operations: `mkdir`, `cp`, `mv`, `rm`, `touch`, `diff`
- `vcpkg` for dependencies

Dangerous commands are blocked:
- `sudo`, `rm -rf /`, `chmod 777`

### Post-Edit Hook

When editing files in `declarative/` directories, you'll see a reminder to update CMakeLists.txt.

### Quick Commands

```bash
# Check progress
cat docs/declarative-pipeline/PROGRESS.md

# List task specs
ls docs/declarative-pipeline/tasks/

# Check declarative files
find base -path '*/declarative/*' -type f | sort

# Build
cmake --build build --parallel

# Test declarative suites (Boost.Test)
./build/aprapipesut --run_test="ModuleRegistryTests/*" --log_level=test_suite
./build/aprapipesut --run_test="PipelineValidatorTests/*" --log_level=test_suite

# Test all (Boost.Test)
./build/aprapipesut --log_level=test_suite

# Configure CMake
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Create directories (first time)
mkdir -p base/include/declarative base/src/declarative base/test/declarative

# GitHub issues
gh issue list --label declarative-pipeline --state open
```

---

## ⚠️ Common Gotchas

### 1. Constexpr String Handling
```cpp
// WRONG - std::string not constexpr
static constexpr std::string name = "Module";

// CORRECT - use string_view
static constexpr std::string_view name = "Module";
```

### 2. Static Initialization Order
```cpp
// REGISTER_MODULE uses static init - be careful with dependencies
// Use function-local statics for singleton:
ModuleRegistry& ModuleRegistry::instance() {
    static ModuleRegistry inst;
    return inst;
}
```

### 3. Boost.Test Integration
```cpp
// Tests must be in BOOST_AUTO_TEST_SUITE
// Link against Boost::unit_test_framework
```

### 4. ApraPipes Module Connection
```cpp
// Existing API uses setNext() for connections
source->setNext(decoder);
// NOT source.connect(decoder) or similar
```

### 5. Shared Pointers
```cpp
// ApraPipes uses boost::shared_ptr, not std::shared_ptr
boost::shared_ptr<Module> module = ...;
```

---

## 🧪 Testing Requirements

**Every task requires:**

1. **Unit tests** in `base/test/<feature>_tests.cpp`
2. **All tests must pass** before marking complete
3. **Test edge cases** mentioned in spec

**Test naming convention:**
```cpp
BOOST_AUTO_TEST_CASE(ClassName_MethodName_Scenario) {
    // ...
}
```

**Run tests (Boost.Test):**
```bash
# All tests
./build/aprapipesut --log_level=test_suite

# Specific test suite
./build/aprapipesut --run_test="MetadataTests/*" --log_level=test_suite

# List available test suites
./build/aprapipesut --list_content 2>&1 | grep -E "^\w+Tests"
```

---

## 📝 Session Checklist

### Starting a Session

- [ ] Read `docs/declarative-pipeline/PROGRESS.md`
- [ ] Check `git log --oneline -10`
- [ ] Run `./build/aprapipesut` to verify current state
- [ ] Identify next task from progress file
- [ ] Read full task spec

### During Session

- [ ] Follow TDD: write tests first
- [ ] Build frequently: `cmake --build build`
- [ ] Run tests frequently: `./build/aprapipesut --run_test="SuiteName/*"`
- [ ] Commit working increments

### Ending a Session

- [ ] All tests pass
- [ ] Update `PROGRESS.md` with:
  - Current task status
  - What was completed
  - What remains
  - Any blockers
  - Notes for next session
- [ ] Commit all changes
- [ ] Push to remote
- [ ] Update GitHub issue (if applicable)

---

## 🚨 If You Get Stuck

1. **Build errors**: Check CMakeLists.txt includes new files
2. **Test failures**: Read test output carefully, check spec
3. **Unclear requirements**: Re-read the task spec, check RFC.md
4. **Dependency issues**: Check vcpkg.json, run `vcpkg install`
5. **Design questions**: Check RFC.md Section 2 (Core Infrastructure)

**Leave notes in PROGRESS.md for the next agent if you can't resolve.**

---

## 📚 Key References

| Document | Purpose |
|----------|---------|
| `docs/declarative-pipeline/RFC.md` | Full design document |
| `docs/declarative-pipeline/tasks/*.md` | Task specifications |
| `docs/declarative-pipeline/PROGRESS.md` | Current status |
| `base/include/Module.h` | Existing Module base class |
| `base/include/FrameMetadata.h` | Existing frame metadata |
| `base/test/*_tests.cpp` | Test patterns |

---

## 🎯 Definition of Done (Global)

A task is complete when:

1. ✅ All files created per spec
2. ✅ All unit tests written and passing
3. ✅ Code compiles on all platforms (check CI if available)
4. ✅ PROGRESS.md updated
5. ✅ Git commit with proper message
6. ✅ GitHub issue updated/closed

---

## Quick Commands Reference

```bash
# Check progress
cat docs/declarative-pipeline/PROGRESS.md

# Read task spec
cat docs/declarative-pipeline/tasks/A1-core-metadata-types.md

# Build
cmake --build build --parallel

# Test (Boost.Test)
./build/aprapipesut --run_test="MetadataTests/*" --log_level=test_suite

# Commit
git add -A && git commit -m "feat(declarative): ..."

# Update GitHub
gh issue comment <num> --body "..."
gh issue close <num>
```
