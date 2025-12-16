# Test Execution Plan - Unified CUDA Build

**Status:** DRAFT - Ready for implementation after Attempt 7 succeeds
**Last Updated:** 2025-12-15

---

## Overview

After the unified Linux CUDA build succeeds on GitHub-hosted runner, we need a two-phase test strategy:

1. **Phase 1:** Run non-CUDA tests on GitHub-hosted runner (no GPU)
2. **Phase 2:** Package build artifacts and run CUDA tests on self-hosted runner (with GPU)

---

## Phase 1: Non-CUDA Tests on GitHub Runner

### What Happens
- Build completes on ubuntu-22.04 GitHub-hosted runner
- Binary is CUDA-enabled but detects no GPU at runtime
- CUDA tests should SKIP gracefully (via runtime detection)
- Non-CUDA tests should PASS

### Current Workflow Steps (Already in CI-Linux-CUDA-Unified.yml)
```yaml
- name: List Test cases
  run: |
    ldd ${{env.TEST_EXE}} | tee >(grep 'not found') || true
    ${{env.TEST_EXE}} --list_content > tests.txt || true
  timeout-minutes: 1

- name: Run Tests
  run: |
    ${{env.TEST_EXE}} --log_format=JUNIT --log_sink=CI_test_result_Linux-CUDA-Unified.xml -p -l all || echo 'test execution returned error'
  timeout-minutes: 20
```

### Expected Behavior
- CUDA modules detect no GPU and skip their tests
- Non-CUDA tests execute normally
- Some tests may fail/skip - we need to verify this is acceptable

### Success Criteria
- `aprapipesut` executable runs without crashing
- Test results XML is generated
- No segfaults or runtime linking errors

---

## Phase 2: CUDA Tests on Self-Hosted Runner

### Critical Question: What Artifacts to Publish?

Based on old CI-Linux-CUDA.yml workflow pattern, we need to publish:

#### Artifacts to Upload (from GitHub runner build)
1. **Build executable:** `build/aprapipesut`
2. **Shared libraries:** All .so files from build directory
3. ~~**Test data:** `data/` directory~~ - NOT NEEDED (self-hosted checkout has this)
4. **vcpkg installed libraries:** May need runtime dependencies

#### Where Old Workflow Published
Old workflow used **build-test-lin.yml** which:
- Built on self-hosted runner directly (no artifact transfer needed)
- Had all dependencies locally
- Published test results via **publish-test.yml**

#### New Challenge: Cross-Runner Artifact Transfer
We're building on **GitHub-hosted** but testing CUDA on **self-hosted**.

### Option A: Upload Build Binaries Only
```yaml
- name: Upload build artifacts
  uses: actions/upload-artifact@v4
  with:
    name: Linux-CUDA-Unified-Build
    path: |
      build/aprapipesut
      build/**/*.so
      build/**/*.so.*
      vcpkg_installed/x64-linux/lib/**/*.so*
```

### Option B: Create Portable Package (Minimal)
```yaml
- name: Package for CUDA testing
  run: |
    mkdir -p cuda-test-package/build
    cp build/aprapipesut cuda-test-package/build/
    cp -r build/*.so* cuda-test-package/build/ || true
    # Copy runtime dependencies
    ldd build/aprapipesut | grep "=> /" | awk '{print $3}' | xargs -I {} cp {} cuda-test-package/build/ || true
    tar czf cuda-test-package.tar.gz cuda-test-package/

- name: Upload CUDA test package
  uses: actions/upload-artifact@v4
  with:
    name: cuda-test-package
    path: cuda-test-package.tar.gz
```

### Recommended Approach
**Option A** - Upload build binaries only:
- Preserves build directory structure
- Self-hosted runner checks out code (gets data/ folder)
- Downloads artifacts into build/ directory
- Minimal artifact size (no duplicate data files)

---

## Phase 3: Self-Hosted CUDA Test Job

### New Job to Add (After Build Succeeds)
```yaml
jobs:
  linux-cuda-unified-build:
    runs-on: ubuntu-22.04
    # ... existing build steps ...

  linux-cuda-test:
    needs: linux-cuda-unified-build
    runs-on: linux-cuda  # Self-hosted runner with GPU
    if: ${{ always() }}  # Run even if GitHub runner tests had issues

    steps:
    - name: Checkout code (for data/ directory and source)
      uses: actions/checkout@v3
      with:
        lfs: true  # Get LFS files if needed

    - name: Download build artifacts
      uses: actions/download-artifact@v4
      with:
        name: Linux-CUDA-Unified-Build
        path: ./  # Downloads into build/ directory structure

    - name: Set executable permissions
      run: chmod +x build/aprapipesut

    - name: Verify CUDA availability
      run: |
        nvidia-smi
        build/aprapipesut --list_content | grep -i cuda || echo "No CUDA tests found"

    - name: Run CUDA tests
      run: |
        cd build
        ./aprapipesut --log_format=JUNIT --log_sink=../CI_test_result_Linux-CUDA-Tests.xml -p -l all || echo 'test execution returned error'
      timeout-minutes: 30

    - name: Upload CUDA test results
      if: ${{ always() }}
      uses: actions/upload-artifact@v4
      with:
        name: TestResults_Linux-CUDA-GPU
        path: CI_test_result_Linux-CUDA-Tests.xml

  linux-cuda-publish:
    needs: [linux-cuda-unified-build, linux-cuda-test]
    if: ${{ always() }}
    permissions:
      checks: write
      pull-requests: write
    uses: ./.github/workflows/publish-test.yml
    with:
      flav: Linux-CUDA-Unified
    secrets:
      GIST_TOKEN: ${{ secrets.GIST_TOKEN }}
```

---

## Critical Items to Verify BEFORE Adding CUDA Test Job

### 1. What libraries does aprapipesut depend on?
```bash
ldd build/aprapipesut
```
Need to ensure all runtime dependencies are:
- Either included in artifacts
- Or available on self-hosted runner

### 2. What does self-hosted runner have pre-installed?
- CUDA toolkit version
- cuDNN version
- System libraries
- vcpkg dependencies?

### 3. Can we run tests from artifact without rebuild?
- Test executable location
- Relative paths to data files
- Library search paths (LD_LIBRARY_PATH)

### 4. Do CUDA tests actually exist?
```bash
# After build succeeds, check:
./aprapipesut --list_content | grep -i cuda
./aprapipesut --list_content | grep -i gpu
./aprapipesut --list_content | grep -i nvenc
```

---

## Immediate Next Steps (After Attempt 7 Succeeds)

1. ✅ **Verify build succeeds**
2. ✅ **Check test execution on GitHub runner** - do tests skip/pass without GPU?
3. ✅ **Analyze ldd output** - what runtime dependencies exist?
4. ✅ **Check test list** - which tests are CUDA-specific?
5. ✅ **Design artifact package** - what exactly to upload?
6. ⏳ **Test artifact download** - can self-hosted runner use it?
7. ⏳ **Add CUDA test job** - implement Phase 2
8. ⏳ **Validate end-to-end** - GitHub build → self-hosted CUDA test → publish

---

## Questions to Answer

1. **Does aprapipesut have separate test suites?**
   - Can we run only non-CUDA tests with a flag?
   - Or does runtime detection handle this automatically?

2. **What test data is needed?**
   - Is `data/` directory required?
   - Are paths hardcoded or relative?

3. **Self-hosted runner environment:**
   - Does it have vcpkg?
   - Does it have all system libraries?
   - Can we rely on pre-installed dependencies?

4. **Artifact size:**
   - How big is the full build directory?
   - Do we need to optimize what we upload?

---

## Success Metrics

### Phase 1 (GitHub Runner) - DONE when:
- ✅ Build completes successfully
- ✅ Tests run without crashes
- ✅ Test results uploaded

### Phase 2 (Self-Hosted CUDA) - DONE when:
- ✅ Artifacts download successfully
- ✅ CUDA tests execute on GPU
- ✅ Test results published
- ✅ No regression from old CI-Linux-CUDA workflow

---

**READY TO IMPLEMENT:** As soon as Attempt 7 build succeeds, we can validate Phase 1 and design Phase 2 artifacts.
