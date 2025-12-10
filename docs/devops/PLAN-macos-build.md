# macOS Build Implementation Plan

**Goal**: Get ApraPipes building and testing on macOS (x86_64) with all unit tests passing

**Timeline**: Overnight (by morning)

**Test Environment**: Local macOS 15.7.2 (x86_64)

**Target Environment**: GitHub-hosted `macos-latest` runners

---

## Phase 1: Environment Assessment & Dependencies (30 minutes)

### Current State Analysis

**Local Machine**:
- macOS 15.7.2 (x86_64)
- CMake 4.1.2 (NEWER than Linux!)
- Ninja 1.13.1 ✓
- Homebrew 5.0.5 ✓
- vcpkg binary EXISTS but is Linux ELF (needs re-bootstrap)

**Critical Discovery**:
- vcpkg binary is Linux ELF, won't run on macOS
- Need fresh vcpkg bootstrap for macOS

**Repository State**:
- CUDA-specific vcpkg.json (name: "apra-pipes-cuda")
- Dependencies include CUDA packages (whisper[cuda], opencv4[cuda,cudnn])
- Platform filters exist for !windows, linux-specific
- NO macOS-specific filters yet
- No ENABLE_MACOS option in CMakeLists.txt

### Action Items

1. **Bootstrap vcpkg for macOS**
   ```bash
   cd vcpkg
   ./bootstrap-vcpkg.sh
   # This will create macOS binary
   ```

2. **Check Homebrew dependencies needed**
   - Likely need: pkg-config, autoconf, automake, libtool
   - Check what Linux prep-cmd installs that we need

3. **Create macOS vcpkg.json variant**
   - Remove CUDA dependencies (whisper cuda feature, opencv cuda features)
   - Keep core dependencies
   - Add platform filters where needed

---

## Phase 2: CMake Configuration for macOS (1 hour)

### CMake Changes Needed

1. **Add ENABLE_MACOS option** in base/CMakeLists.txt
   ```cmake
   OPTION(ENABLE_MACOS "Use this switch to enable MACOS" OFF)

   IF(ENABLE_MACOS)
       add_compile_definitions(MACOS)
   ENDIF(ENABLE_MACOS)
   ```

2. **Handle macOS-specific packages**
   - Linux uses GTK3, GDK3, GIO, GOBJECT via pkg-config
   - macOS doesn't have native GTK3 (can install via Homebrew but complex)
   - May need to disable GTK3 on macOS OR install via Homebrew
   - Check if code actually requires GTK3 or if it's optional

3. **Configure command for macOS**
   ```bash
   cmake -B build -G Ninja \
     -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
     -DENABLE_WINDOWS=OFF \
     -DENABLE_LINUX=OFF \
     -DENABLE_MACOS=ON \
     -DENABLE_CUDA=OFF \
     ../base
   ```

### Potential Issues to Investigate

- **OpenGL**: macOS has native OpenGL, but deprecated. May need special handling
- **X11 packages**: libx11, libxcb, etc. don't exist on macOS (use native frameworks)
- **GLEW/GLUT**: FreeGLUT might not work on macOS (use native GLUT or GLFW)
- **Sound**: OpenAL works on macOS
- **GTK3**: Either skip or install via Homebrew

---

## Phase 3: Dependency Resolution (2-3 hours)

### Strategy: Iterative CMake Configure

1. **First attempt**: Configure with current vcpkg.json
   - Will likely fail on Linux-specific packages
   - Document each failure

2. **For each failure**:
   - Identify if package is:
     a) Not needed on macOS (platform-specific UI, X11, etc.)
     b) Needs macOS alternative (e.g., native frameworks instead of X11)
     c) Available but needs different triplet/variant

3. **Expected Problem Packages**:

   **Linux-specific (likely skip on macOS)**:
   - gtk3 (line 89-90 in vcpkg.json: `platform: "!windows"`)
   - glib with libmount (line 93-99: `platform: "(linux & x64)"`)
   - X11 libraries (libx11, libxcb, etc.)
   - baresip/re (line 115-121: `platform: "!windows"`)

   **May need investigation**:
   - freeglut (macOS has native GLUT or use GLFW)
   - glew (might work via Homebrew)
   - hiredis/redis-plus-plus (line 107-113: `platform: "!arm64"`)

4. **Create vcpkg-macos.json or platform filters**

   Option A: Separate vcpkg-macos.json
   ```json
   {
     "name": "apra-pipes-macos",
     "builtin-baseline": "...",
     "dependencies": [
       // Core deps without CUDA, Linux-specific packages
     ]
   }
   ```

   Option B: Add platform filters to existing vcpkg.json
   ```json
   {
     "name": "gtk3",
     "platform": "linux"
   },
   {
     "name": "glib",
     "platform": "linux & x64"
   }
   ```

   **Recommendation**: Option B (platform filters) to keep single vcpkg.json

---

## Phase 4: Build Iteration (2 hours)

### Build Loop

```bash
# 1. Configure
cmake -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DENABLE_MACOS=ON -DENABLE_CUDA=OFF \
  -DENABLE_LINUX=OFF -DENABLE_WINDOWS=OFF \
  ../base

# 2. Check vcpkg install log if configure fails
cat build/vcpkg-manifest-install.log

# 3. Fix dependency issues

# 4. Build
cmake --build build -j $(sysctl -n hw.ncpu)

# 5. Document errors, fix, repeat
```

### Expected Build Errors

1. **Missing system frameworks on macOS**
   - May need to link: Cocoa, OpenGL, CoreAudio, etc.
   - Add to CMakeLists.txt: `target_link_libraries(... "-framework Cocoa")`

2. **Code using Linux-specific APIs**
   - #ifdef LINUX blocks that need macOS equivalents
   - File paths (/ vs \, but macOS uses / like Linux)
   - Threading (should work, POSIX)

3. **Symbol conflicts**
   - macOS uses different C++ stdlib than Linux
   - May need -std=c++17 flags

### Strategy for Code Issues

- **Minimal changes**: Add `#ifdef MACOS` where absolutely needed
- **Prefer runtime detection** over compile-time where possible
- **Document workarounds** in docs/devops/mac-build.md

---

## Phase 5: Test Execution (1 hour)

### Test Strategy

```bash
# Run unit tests
./build/aprapipesut

# If failures:
# 1. Check if test is platform-specific (e.g., CUDA tests)
# 2. Check if test uses Linux-specific features
# 3. Fix or skip test with platform guard
```

### Expected Test Issues

1. **File path tests**: May hardcode Linux paths
2. **Performance tests**: Different timing on macOS
3. **Hardware tests**: No CUDA on macOS
4. **UI tests**: Different windowing system

### Success Criteria

- All non-CUDA tests pass
- No segfaults or crashes
- Test output is readable and makes sense

---

## Phase 6: GitHub Actions Workflow (1 hour)

### Create CI-MacOS-NoCUDA.yml

Based on CI-Linux-NoCUDA.yml pattern:

```yaml
name: CI-MacOS-NoCUDA

on:
  workflow_dispatch:
  # push: (disable initially, enable after testing)

jobs:
  macos-nocuda-build-test:
    uses: ./.github/workflows/build-test-mac.yml
    with:
      runner: 'macos-latest'  # or macos-14, macos-13
      flav: 'MacOS'
      cuda: 'OFF'
      is-selfhosted: false
      nProc: 3
```

### Create build-test-mac.yml

Based on build-test-lin.yml but macOS-specific:

```yaml
on:
  workflow_call:
    inputs:
      runner: { type: string, required: true }
      flav: { type: string, required: true }
      cuda: { type: string, required: true }
      is-selfhosted: { type: boolean, required: true }
      nProc: { type: number, default: 3 }

jobs:
  build:
    runs-on: ${{ inputs.runner }}
    steps:
      - name: Prepare builder
        run: |
          brew install pkg-config autoconf automake libtool ninja
          cmake --version

      - name: Checkout code
        uses: actions/checkout@v3
        with:
          submodules: recursive

      - name: Bootstrap vcpkg
        run: ./vcpkg/bootstrap-vcpkg.sh

      - name: Configure CMake
        run: |
          cmake -B build -G Ninja \
            -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
            -DENABLE_MACOS=ON -DENABLE_CUDA=OFF \
            ../base

      - name: Build
        run: cmake --build build -j ${{ inputs.nProc }}

      - name: Test
        run: ./build/aprapipesut
```

---

## Phase 7: Documentation (30 minutes)

### docs/devops/mac-build.md Structure

```markdown
# macOS Build Guide

## Prerequisites
- macOS 10.15+ (tested on 15.7.2)
- Homebrew
- Xcode Command Line Tools

## Local Development Setup
[Step by step instructions]

## Known Issues
[Document each issue encountered and solution]

## Differences from Linux Build
[Platform-specific quirks]

## GitHub Actions
[CI workflow explanation]

## Troubleshooting
[Common errors and fixes]
```

---

## Risk Assessment & Mitigation

### High Risk Areas

1. **GTK3 Dependencies**
   - **Risk**: Core UI code might depend heavily on GTK3
   - **Mitigation**: Check if GTK3 is actually used in tests, may be optional
   - **Fallback**: Install GTK3 via Homebrew (adds complexity)

2. **OpenGL Deprecation on macOS**
   - **Risk**: Apple deprecated OpenGL, may have warnings/issues
   - **Mitigation**: Code likely uses basic OpenGL, should still work
   - **Fallback**: Plan Metal migration (future, not for this phase)

3. **vcpkg Package Availability**
   - **Risk**: Some packages may not have macOS ports
   - **Mitigation**: Check vcpkg registry, use alternatives
   - **Fallback**: Build from source or skip feature

### Medium Risk Areas

1. **CMake version mismatch** (local has 4.1.2, CI uses 3.29)
   - **Mitigation**: Downgrade local CMake to match CI
   - **Or**: Update CI to use newer CMake

2. **File path handling**
   - **Risk**: Hardcoded Linux paths
   - **Mitigation**: macOS uses / like Linux, should mostly work

3. **Library linking order**
   - **Risk**: macOS linker is stricter than Linux
   - **Mitigation**: Fix link order if errors occur

---

## Success Metrics

### Must Have (for "make me clap")
- ✅ vcpkg successfully installs all required packages on macOS
- ✅ CMake configures without errors
- ✅ Project builds without errors
- ✅ Unit tests run (even if some fail initially)
- ✅ At least 80% of tests pass
- ✅ No crashes or segfaults in passing tests

### Nice to Have
- ✅ 100% test pass rate
- ✅ CI workflow ready and tested
- ✅ Comprehensive documentation of issues/solutions
- ✅ Clean build (no warnings)

---

## Execution Order

1. **Bootstrap vcpkg** (5 min)
2. **Add ENABLE_MACOS to CMake** (10 min)
3. **First configure attempt** - document failures (15 min)
4. **Add platform filters to vcpkg.json** (30 min)
5. **Iterate configure until success** (1 hour)
6. **Fix CMake issues** (1 hour)
7. **Build** (30 min)
8. **Fix build errors** (1 hour)
9. **Run tests** (30 min)
10. **Fix test failures** (1 hour)
11. **Create GitHub Actions workflows** (1 hour)
12. **Document everything** (30 min)

**Total Estimated Time**: 6-8 hours

---

## Plan Critique

### Strengths
1. ✅ Methodical approach - each phase builds on previous
2. ✅ Clear success criteria
3. ✅ Risk assessment with mitigations
4. ✅ Realistic time estimates
5. ✅ Leverages existing Linux/Windows patterns

### Weaknesses & Improvements

1. **⚠️ Assumption: GTK3 is optional**
   - **Improvement**: Should check source code for GTK3 usage first
   - **Action**: Add quick grep for gtk usage before Phase 3

2. **⚠️ No rollback plan if blockers found**
   - **Improvement**: Define blockers that would stop progress
   - **Action**: If GTK3 is required and won't build, consider this a blocker

3. **⚠️ Testing might take longer than estimated**
   - **Improvement**: Be realistic - debugging test failures can be time-consuming
   - **Action**: Prioritize getting SOME tests passing over ALL tests

4. **⚠️ CMake version discrepancy not addressed upfront**
   - **Improvement**: Decide on CMake version strategy immediately
   - **Action**: Check if local CMake 4.1.2 causes issues, downgrade if needed

5. **⚠️ No plan for OpenCV CUDA features**
   - **Improvement**: May need separate vcpkg.json or extensive filters
   - **Action**: Consider creating macos-overlay similar to arm64-overlay

### Revised Approach

**Pre-Phase 0: Quick Reconnaissance (15 min)**
1. Check source code for GTK3 usage - is it critical?
2. Check if any code has `#ifdef __APPLE__` or `#ifdef __MACH__`
3. Verify CMake 4.1.2 compatibility, downgrade if needed
4. List all packages in vcpkg.json, mark which are definitely Linux-only

**This gives us critical info before diving in!**

---

## Monitoring & Adjustment

Throughout execution:
- **Document every error** immediately in mac-build.md
- **Update plan** if blockers found
- **Timebox each phase** - if stuck > 30 min on one issue, document and move on
- **Checkpoint progress** every hour - commit working state

---

## Final Checklist Before Execution

- [ ] Branch created: feat/ci-macos-nocuda
- [ ] All other workflows disabled on this branch
- [ ] Plan reviewed and approved
- [ ] Ready to start with Pre-Phase 0 reconnaissance

---

**READY TO EXECUTE**
