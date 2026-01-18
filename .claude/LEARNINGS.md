# Learnings

## CMake/ARM64

### GTK3 must be explicitly linked on ARM64
When adding GTK-dependent code to ARM64/Jetson builds, you must explicitly call `pkg_check_modules(GTK3 REQUIRED gtk+-3.0)` AND link the libraries. The CMakeLists.txt had ARM64-specific include directories but was missing the library linking.

```cmake
# For ARM64/Jetson, need BOTH:
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)  # Define GTK3_LIBRARIES
target_include_directories(target PRIVATE ${VCPKG_GTK_INCLUDE_DIRS})
target_link_libraries(target ${GTK3_LIBRARIES})  # Don't forget this!
```

Error symptom: `undefined reference to 'gtk_gl_area_get_error'`

### ARM64 test files shouldn't use nv_test_utils.h symbols
The `nv_test_utils.h` header (which contains `utf` namespace alias and `if_h264_encoder_supported` precondition) is only included for non-ARM64 builds. Don't use NVENC-specific preconditions inside `#ifdef ARM64` blocks.

```cpp
// Bad - nv_test_utils.h not included for ARM64
#ifdef ARM64
BOOST_AUTO_TEST_CASE(test, *utf::precondition(if_h264_encoder_supported()))  // ERROR!
#endif

// Good - no NVENC precondition for ARM64 tests
#ifdef ARM64
BOOST_AUTO_TEST_CASE(test)  // Works
#endif
```

## GitHub CLI

### gh run watch interval
Never run `gh run watch` with default 3 second interval. Always use `-i 120` (2 mins) or more to avoid excessive API calls and rate limiting.

```bash
# Bad - polls every 3 seconds
gh run watch 12345

# Good - polls every 120 seconds
gh run watch 12345 -i 120 --exit-status
```

### NEVER cancel workflows on other branches
When cancelling workflow runs, ALWAYS filter by the current branch. Cancelling runs on other branches is destructive and affects other developers' work.

```bash
# Bad - cancels all matching runs regardless of branch
gh run list -w CI-MacOSX-NoCUDA --json databaseId,status --jq '...'

# Good - filter by current branch before cancelling
gh run list -w CI-MacOSX-NoCUDA -b feature/get-rid-of-nocuda-builds --json databaseId,status --jq '...'
```

## GitHub Actions Workflows

### Runner parameter must be JSON for container workflows
When calling `build-test-lin-container.yml` which uses `fromJson(inputs.runner)`, the runner parameter MUST be a JSON-formatted string, not a plain string.

```yaml
# Bad - plain string causes silent job failure
runner: ubuntu-22.04

# Good - JSON array format
runner: '["ubuntu-22.04"]'

# Good - multiple labels for self-hosted
runner: '["self-hosted", "Linux", "ARM64"]'
```

**Symptom:** Job silently doesn't run (not even shown as skipped), dependent jobs fail trying to download non-existent artifacts.

**Reference:** `CI-Linux-CUDA-Docker.yml.disabled` line 36 shows correct format.

### Cross-workflow check runs cause confusion
`EnricoMi/publish-unit-test-result-action` creates GitHub check runs that are visible across ALL workflows for the same commit. A check named `Test Results Linux_ARM64` created by CI-Linux-ARM64 will appear in CI-Linux's check list.

**Impact:** When CI-Linux shows "failure" with `Test Results Linux_ARM64` failing, it's actually a failure from CI-Linux-ARM64 workflow, not CI-Linux.

**Solution options:**
1. Prefix check names with workflow name: `CI-Linux: Test Results` vs `CI-ARM64: Test Results`
2. Use `check_run_annotations` parameter to control visibility
3. Accept the behavior and train team to check actual workflow run

### Verify CI status claims before accepting
Never trust "all passed" claims from previous sessions without verification. Always:
1. Run `gh run view <id> --json jobs` to see actual job status
2. Check for jobs that didn't run (missing from list = potential silent failure)
3. Look at actual test result annotations, not just job conclusions

### Job naming convention for reusable workflows
When using reusable workflows, the job names appear as `{caller-job} / {reusable-job}`. Use short, meaningful names:

**Caller workflow (e.g., CI-Linux.yml):**
```yaml
jobs:
  ci:  # Short top-level name
    uses: ./.github/workflows/build-test.yml
    with:
      check_prefix: CI-Lin  # For check run naming
```

**Reusable workflow (e.g., build-test.yml):**
```yaml
jobs:
  build:      # ci / build
  report:     # ci / report
  cuda:       # ci / cuda (calls another workflow)
  docker:     # ci / docker
  docker-report:  # ci / docker-report
```

**Result in UI:**
```
ci
├── build
├── report
├── cuda / setup
├── cuda / gpu-test
├── cuda / report
├── docker / build
└── docker-report
```

### Check run naming with prefix
Use `check_prefix` parameter to distinguish check runs from different workflows:

```yaml
# In publish-test.yml
check_name: ${{ inputs.check_prefix != '' && format('{0}-Tests', inputs.check_prefix) || format('Test-Results-{0}', inputs.flav) }}
```

Results:
- CI-Linux with `check_prefix: CI-Lin` → check name `CI-Lin-Tests`
- CI-Windows with `check_prefix: CI-Win` → check name `CI-Win-Tests`
- Fallback (no prefix) → `Test-Results-{flav}`

## CUDA / NvCodec

### Always check ck() return value in constructors
The `ck()` macro logs errors but does NOT throw exceptions - it returns `false`. If you ignore the return value, execution continues with invalid CUDA state.

```cpp
// Bad - continues with invalid cuContext if cuCtxCreate fails
ck(loader.cuCtxCreate(&cuContext, 0, cuDevice));
helper.reset(new NvDecoder(cuContext, ...));  // Crash later with garbage context!

// Good - throw on failure to prevent invalid state
if (!ck(loader.cuCtxCreate(&cuContext, 0, cuDevice))) {
    throw std::runtime_error("cuCtxCreate failed (possibly out of GPU memory)");
}
```

**Symptom:** Memory access violation at address 0x3f8 (offset 1016 bytes from null pointer) when accessing NvDecoder methods.

**Root cause:** `CUDA_ERROR_OUT_OF_MEMORY` at `cuCtxCreate`, but ck() just logs and returns false. Execution continues with uninitialized cuContext, then NvDecoder methods crash.

**Fix:** Check ck() return value and throw exception on failure.

### CUDA contexts must be destroyed to prevent memory leaks
The NvDecoder destructor was missing `cuCtxDestroy(m_cuContext)`. Each H264Decoder created a CUDA context that was never destroyed, leaking GPU memory.

```cpp
// BAD - context leaked (was the original code)
NvDecoder::~NvDecoder() {
    cuvidDestroyVideoParser(m_hParser);
    cuvidDestroyDecoder(m_hDecoder);
    // cuMemFree for device frames...
    // Missing: cuCtxDestroy(m_cuContext)!
}

// GOOD - context properly destroyed
NvDecoder::~NvDecoder() {
    cuvidDestroyVideoParser(m_hParser);
    cuvidDestroyDecoder(m_hDecoder);
    // cuMemFree for device frames...
    if (m_cuContext && loader.cuCtxDestroy) {
        loader.cuCtxDestroy(m_cuContext);
        m_cuContext = nullptr;
    }
}
```

**Symptom:** GPU OOM (`CUDA_ERROR_OUT_OF_MEMORY`) after creating/destroying multiple decoders. Tests fail with OOM on memory-constrained GPUs.

**Root cause:** CUDA contexts consume significant GPU memory. Without destruction, memory accumulates until exhausted.

## CI/Test Workflows

### CRITICAL: Test steps must exit 1 on failure
The test execution step must parse the XML results and exit with code 1 if there are failures or errors. Otherwise workflows show green when tests fail!

```bash
# BAD - swallows the error, workflow shows green
./test_exe --log_format=JUNIT --log_sink=results.xml -p -l all || echo 'error'

# GOOD - parse XML and fail on errors/failures
./test_exe --log_format=JUNIT --log_sink=results.xml -p -l all
TEST_EXIT=$?

if [ -f "results.xml" ]; then
  ERRORS=$(grep -oP 'errors="\K[0-9]+' results.xml | head -1)
  FAILURES=$(grep -oP 'failures="\K[0-9]+' results.xml | head -1)
  if [ "$ERRORS" -gt 0 ] || [ "$FAILURES" -gt 0 ]; then
    echo "::error::Tests failed: $FAILURES failures, $ERRORS errors"
    exit 1
  fi
fi
```

**Symptom:** Workflow shows green (success) but test results artifact shows failures/errors.

**Affected files (fixed):**
- `build-test.yml` - main test step
- `CI-CUDA-Tests.yml` - Linux and Windows CUDA tests
- `build-test-lin-container.yml` - Docker tests
- `build-test-macosx.yml` - macOS tests

**Important:** Ensure `Upload test results` step has `if: always()` and `report` job has `if: always()` so results are published even when tests fail.

### Use primary context API to prevent GPU OOM in tests
When creating CUDA contexts in modules that may be instantiated many times (like decoders), use the primary context API instead of `cuCtxCreate`. The primary context is reference-counted and shared per device, preventing GPU memory exhaustion.

```cpp
// BAD - creates new context each time, consumes GPU memory
CUcontext cuContext;
cuCtxCreate(&cuContext, 0, cuDevice);
// ... use context ...
cuCtxDestroy(cuContext);  // Too late if many instances created

// GOOD - shares primary context, reference counted
CUcontext cuContext;
cuDevicePrimaryCtxRetain(&cuContext, cuDevice);
m_ownedDevice = cuDevice;  // Store device for release
// ... use context ...
cuDevicePrimaryCtxRelease(m_ownedDevice);  // Just decrements refcount
```

**Symptom:** `CUDA_ERROR_OUT_OF_MEMORY` when creating contexts, especially for tests that run late in the test suite (like `h264decoder_tests` which runs last among CUDA tests).

**Root cause:** Each `cuCtxCreate` allocates GPU memory. When running many tests sequentially (e.g., all CUDA tests), memory accumulates even with proper destruction because there are overlapping lifetimes. Primary context avoids this by reusing a single context.

**Fixed file:** `H264DecoderNvCodecHelper.cpp` - Changed from `cuCtxCreate/Destroy` to `cuDevicePrimaryCtxRetain/Release`

**Note:** This matches the pattern used by `ApraCUcontext` in `CudaCommon.h`.

## vcpkg

### Compiler path affects binary cache ABI hash
vcpkg uses the literal compiler PATH in its ABI hash calculation, not just the version. Two builds using the same compiler version but different paths will NOT share cached packages.

```bash
# Cloud build uses explicit path
CC=/usr/bin/gcc-11
CXX=/usr/bin/g++-11

# Docker build uses default symlink
CC=/usr/bin/cc → /usr/bin/gcc-11
CXX=/usr/bin/c++ → /usr/bin/g++-11
```

**Both are GCC 11.4.0** but different paths = different ABI hashes = cache miss.

**Symptom:** GitHub Actions cache is restored (2GB downloaded), but vcpkg logs show `Restored 0 package(s)`. CMake configure takes 2+ hours rebuilding all packages.

**Fix:** Ensure all builds sharing cache use identical compiler paths:
```yaml
# In workflow env:
env:
  CC: /usr/bin/gcc-11
  CXX: /usr/bin/g++-11
```

**Debug tip:** Search cmake configure logs for `Compiler found:` to see the exact path being used:
```
-- The C compiler identification is GNU 11.4.0
...
Compiler found: /usr/bin/g++-11
```
