# CLAUDE.md - ApraPipes Declarative Pipeline

> Instructions for Claude Code agents working on the ApraPipes project.

**Branch:** `feat-declarative-pipeline-v2`
**Documentation:** `docs/declarative-pipeline/PROGRESS.md`

---

## 🎯 Current Phase: Node.js Addon on Jetson (J2)

**Mission:** Fix the Node.js addon (`aprapipes.node`) build and test on Jetson ARM64.

**Protected Assets (DO NOT BREAK):**
- ✅ 7 L4TM tests passing in CI-Linux-ARM64
- ✅ All 4 CI workflows GREEN
- ✅ L4TM CLI pipelines working on Jetson

**Current Issue:**
Node.js addon fails to load due to missing Boost.Serialization RTTI symbols:
```
undefined symbol: _ZTIN5boost7archive6detail17basic_iserializerE
```

**Root Cause (from JETSON_KNOWN_ISSUES.md):**
- `--whole-archive` flag only applies to `aprapipes` library, not Boost libraries
- GCC 9.4 on Jetson has stricter symbol resolution than newer GCC
- Typeinfo symbols for polymorphic classes get discarded during linking

**Potential Solutions:**
1. **Option A:** Extend `--whole-archive` to include `Boost_SERIALIZATION_LIBRARY`
2. **Option B:** Use `--no-as-needed` flag for Boost libraries
3. **Option C:** Build Boost as shared libraries on ARM64
4. **Option E:** Remove Boost.Serialization dependency from declarative code path

**Reference:** See `docs/declarative-pipeline/JETSON_KNOWN_ISSUES.md` → Issue J2

---

## 📚 Learnings (Document for Future Reference)

### L4TM CLI Pipeline Debugging (2026-01-16)

**Problem 1: Frame type mismatch**
- Error: `input frameType is expected to be ENCODED_IMAGE. Actual<0>`
- Fix: Add `"outputFrameType": "EncodedImage"` to FileReaderModule props
- Learning: FileReaderModule defaults to generic "Frame" type; must explicitly specify for encoded content

**Problem 2: No Pins to connect**
- Error: `No Pins to connect. JPEGDecoderL4TM_1 -> JPEGEncoderL4TM_2`
- Fix: Remove `selfManagedOutputPins = true` from L4TM module registrations
- Learning: Modules that expect factory to create output pins must have `selfManagedOutputPins = false` (the default)

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

### 4. Mission Alignment Checkpoint (MANDATORY)

**Before EVERY commit, answer these questions:**

1. **What is the mission?** (Re-read the plan file or task description)
2. **Does this commit move toward the mission?**
3. **Am I disabling/removing/hiding something instead of fixing it?**

#### Red Flag Actions - STOP and Reflect

If you're about to do any of these, STOP and ask "Is this actually achieving the mission?":

- Disabling tests to make CI green
- Removing functionality to avoid fixing it
- Adding `#if 0` / `#ifdef DISABLED` around problematic code
- Marking something as "TODO later" when it's the actual task
- Celebrating "green CI" when tests are skipped/disabled

**These are symptoms of metric gaming, not problem solving.**

#### The Test

> If someone asked "Did you fix the problem?", would your answer be:
> - ✅ "Yes, it works now" → Commit OK
> - ❌ "No, but CI is green because I disabled/skipped it" → DO NOT COMMIT

**Never optimize for intermediate metrics (CI green, build passing) at the expense of the actual mission. If you find yourself disabling something to make a dashboard green, you've lost the plot.**

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
- ~~J1: libjpeg version conflict~~ ✅ RESOLVED (dlopen wrapper isolates symbols)
- J2: Node.js addon Boost.Serialization linking (use CLI as workaround)
- J3: H264EncoderV4L2 not registered on ARM64 (use H264EncoderNVCodec)

### L4TM Test Protection

**Before ANY changes to module registrations or Jetson code:**
```bash
ssh akhil@192.168.1.18
cd /data/ws/ApraPipes
./_build/aprapipesut --run_test="jpegencoderl4tm_tests/*,jpegdecoderl4tm_tests/*" --log_level=test_suite
```

**Expected:** 7 tests pass, 7 tests skipped (disabled). If any previously passing test fails, STOP and investigate before committing.

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
- Sprint 8 (Jetson Integration) ✅ COMPLETE
- Current focus: Module registration improvements for CLI
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
