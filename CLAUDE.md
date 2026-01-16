# CLAUDE.md - ApraPipes Declarative Pipeline

> Instructions for Claude Code agents working on the ApraPipes project.

**Branch:** `feat-declarative-pipeline-v2`
**Documentation:** `docs/declarative-pipeline/PROGRESS.md`

---

## 🚨 Critical Rules

### 1. Build and Test Before Commit (MANDATORY)

**NEVER commit code without verifying build and tests pass.**

```bash
# 1. Build must succeed
cmake --build build -j$(nproc)

# 2. Tests must pass
./build/aprapipesut --run_test="<RelevantSuite>/*" --log_level=test_suite

# 3. For CLI changes, smoke test
./build/aprapipes_cli run <example.json>
```

If build/tests fail: fix first, then commit. No exceptions.

### 2. Platform Protection (MANDATORY)

**Keep all 4 CI workflows GREEN:**
- CI-Windows, CI-Linux, CI-Linux-ARM64, CI-MacOSX-NoCUDA

**Isolate platform-specific code with guards:**

```cpp
#ifdef ENABLE_ARM64
    // Jetson-only code
#endif

#ifdef ENABLE_CUDA
    // CUDA-only code
#endif

#ifdef ENABLE_LINUX
    // Linux-only code
#endif
```

**Before committing:**
- Verify changes are properly guarded for the target platform
- Check for cross-platform impact (Windows? macOS? Linux x64?)
- Never modify `base/vcpkg.json` or core headers for platform-specific fixes

**If you break another platform:** Revert immediately, fix with proper guards.

### 3. Code Review Before Commit

```bash
git diff --staged          # Review ALL changes
git diff --staged --stat   # Check which files changed
```

Check for: debug code, temporary hacks, commented-out code, unrelated changes.

---

## 🔧 Jetson Development

### Device Rules

When working on Jetson (ssh akhil@192.168.1.18):
- **NEVER** modify `/data/action-runner/` (GitHub Actions)
- **NEVER** delete `/data/.cache/` (vcpkg cache shared with CI)
- **ALWAYS** work in `/data/ws/`

### Workspace Layout

```
/data/ws/ApraPipes/          # Main workspace (NVMe)
├── _build/                  # Build directory
├── base/                    # Source code
└── vcpkg/                   # vcpkg submodule

/data/.cache/                # Shared cache
├── vcpkg_installed/         # vcpkg packages
└── tmp/                     # Temp for builds (TMPDIR)
```

### Build Commands

```bash
ssh akhil@192.168.1.18
cd /data/ws/ApraPipes

# Configure
cmake -B _build -S base \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_ARM64=ON \
  -DENABLE_CUDA=ON

# Build (use -j2 to avoid OOM)
TMPDIR=/data/.cache/tmp cmake --build _build -j2

# Test
./_build/aprapipesut --run_test="ModuleRegistryTests/*" --log_level=test_suite
```

### Known Issues

See `docs/declarative-pipeline/JETSON_KNOWN_ISSUES.md` for:
- J1: libjpeg version conflict (L4TM modules)
- J2: Node.js addon Boost.Serialization linking
- J3: H264EncoderV4L2 not registered on ARM64

---

## 📍 Project Context

### Goal

Transform ApraPipes from imperative C++ to declarative JSON:

```json
{
  "modules": {
    "source": { "type": "FileReaderModule", "props": { "path": "/video.mp4" } }
  }
}
```

### Current Status

Check `docs/declarative-pipeline/PROGRESS.md` for:
- Sprint status (currently Sprint 8: Jetson Integration)
- Registered modules (50+ modules)
- Known issues and workarounds

### Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/declarative-pipeline/PROGRESS.md` | Current status, sprint progress |
| `docs/declarative-pipeline/PROJECT_PLAN.md` | Sprint overview, objectives |
| `docs/declarative-pipeline/JETSON_KNOWN_ISSUES.md` | Jetson platform issues |
| `docs/declarative-pipeline/DEVELOPER_GUIDE.md` | Module registration guide |
| `docs/declarative-pipeline/PIPELINE_AUTHOR_GUIDE.md` | JSON pipeline authoring |

---

## ⚠️ Common Gotchas

### C++ Patterns

```cpp
// Use string_view for constexpr strings
static constexpr std::string_view name = "Module";  // CORRECT
static constexpr std::string name = "Module";       // WRONG

// ApraPipes uses boost::shared_ptr
boost::shared_ptr<Module> module = ...;  // CORRECT
std::shared_ptr<Module> module = ...;    // WRONG

// Module connections use setNext()
source->setNext(decoder);  // CORRECT
```

### Boost.Test

```cpp
BOOST_AUTO_TEST_SUITE(MyTests)
BOOST_AUTO_TEST_CASE(Test_Something) {
    BOOST_CHECK(condition);
}
BOOST_AUTO_TEST_SUITE_END()
```

### Static Initialization

```cpp
// Use function-local statics for singletons
ModuleRegistry& ModuleRegistry::instance() {
    static ModuleRegistry inst;
    return inst;
}
```

---

## 🛠️ Quick Reference

```bash
# Check progress
cat docs/declarative-pipeline/PROGRESS.md

# Build
cmake --build build -j$(nproc)

# Test specific suite
./build/aprapipesut --run_test="ModuleRegistryTests/*" --log_level=test_suite

# Test all
./build/aprapipesut --log_level=test_suite

# Run CLI
./build/aprapipes_cli list-modules
./build/aprapipes_cli run examples/simple.json

# Monitor CI after push
gh run list --limit 8
```
