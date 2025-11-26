# ApraPipes Code Coverage

This document describes how to generate and view code coverage reports for the ApraPipes project.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Manual Coverage Generation](#manual-coverage-generation)
- [CI/CD Integration](#cicd-integration)
- [Viewing Coverage Reports](#viewing-coverage-reports)
- [Understanding Coverage Metrics](#understanding-coverage-metrics)
- [Troubleshooting](#troubleshooting)

## Overview

Code coverage measures how much of the codebase is executed during testing. The ApraPipes project uses:

- **gcov**: GNU coverage tool for instrumenting code
- **lcov**: Test coverage program that uses gcov
- **genhtml**: Generates HTML coverage reports from lcov data

Coverage reports help identify:
- Untested code paths
- Areas needing more tests
- Overall test quality

## Prerequisites

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y lcov gcov gcovr
```

### Required Build Tools

Ensure you have the standard ApraPipes build dependencies installed:

```bash
sudo apt-get install -y \
    build-essential cmake ninja-build \
    git pkg-config python3 python3-pip
```

## Quick Start

### Using the Helper Script

The easiest way to generate coverage is using the provided script:

```bash
# From the project root directory
./generate_coverage.sh
```

For a clean build:

```bash
./generate_coverage.sh clean
```

This script will:
1. Configure CMake with coverage enabled
2. Build the project with instrumentation
3. Run all tests
4. Generate coverage reports (both .info and HTML)
5. Display a summary

### Opening the Report

After generation, open the HTML report:

```bash
# Linux
xdg-open build_coverage/coverage/index.html

# Or with Firefox
firefox build_coverage/coverage/index.html
```

## Manual Coverage Generation

If you prefer manual control:

### Step 1: Configure Build

```bash
mkdir -p build_coverage
cd build_coverage

cmake -B . \
    -DENABLE_LINUX=ON \
    -DENABLE_CUDA=OFF \
    -DCODE_COVERAGE=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
    ../base
```

### Step 2: Build with Coverage

```bash
cmake --build . --config Debug -j $(nproc)
```

### Step 3: Run Tests and Generate Report

```bash
make coverage
```

This target automatically:
- Resets coverage counters
- Runs the test suite (`aprapipesut`)
- Collects coverage data
- Filters out system/third-party libraries
- Generates HTML report in `coverage/` directory

### Step 4: View Coverage Summary

```bash
lcov --summary coverage.info
```

## CI/CD Integration

### GitHub Actions

Coverage is automatically generated on:
- Push to `main`, `master`, `develop` branches
- Pull requests to these branches
- Manual workflow dispatch

### Workflow File

`.github/workflows/code-coverage.yml`

### Artifacts

Coverage reports are uploaded as workflow artifacts:
- HTML coverage report
- `coverage.info` lcov file
- Coverage summary text
- Coverage badge JSON

Download from: GitHub Actions → Workflow Run → Artifacts

### Codecov Integration

Coverage data is automatically uploaded to [Codecov](https://codecov.io) for trend tracking and PR comments.

**Setup Codecov:**

1. Sign up at https://codecov.io with your GitHub account
2. Add your repository
3. Get the upload token
4. Add as GitHub secret: `CODECOV_TOKEN`

## Viewing Coverage Reports

### HTML Report Structure

```
coverage/
├── index.html          # Main page with overall summary
├── base/
│   ├── src/           # Source file coverage
│   │   ├── Module.cpp.gcov.html
│   │   └── ...
│   └── include/       # Header file coverage
└── ...
```

### Report Features

- **Line Coverage**: Which lines were executed
- **Function Coverage**: Which functions were called
- **Branch Coverage**: Which conditional branches were taken
- **Color Coding**:
  - 🟢 Green: Covered lines
  - 🔴 Red: Uncovered lines
  - 🟡 Yellow: Partially covered branches

### Key Files to Review

Focus coverage improvements on:
- Core pipeline modules (`src/Module.cpp`, `src/Pipeline.cpp`)
- Critical components (`src/Mp4ReaderSource.cpp`, `src/Mp4WriterSink.cpp`)
- Cache management (`src/OrderedCacheOfFiles.cpp`)
- Frame processing modules

## Understanding Coverage Metrics

### Line Coverage

```
Lines......: 75.2% (8543 of 11352 lines)
```

Percentage of code lines executed during tests.

### Function Coverage

```
Functions..: 68.4% (1234 of 1804 functions)
```

Percentage of functions called during tests.

### Branch Coverage

```
Branches...: 52.1% (3421 of 6567 branches)
```

Percentage of conditional branches (if/else, switch) taken.

### Coverage Goals

- **Good**: > 70% line coverage
- **Excellent**: > 80% line coverage
- **Outstanding**: > 90% line coverage

**Note**: 100% coverage is often impractical. Focus on critical paths.

## Excluding Code from Coverage

### In CMakeLists.txt

Excluded patterns are defined in `base/CMakeLists.txt`:

```cmake
set(COVERAGE_EXCLUDES
    '*/test/*'           # Test files themselves
    '*/thirdparty/*'     # Third-party libraries
    '*/vcpkg/*'          # Package manager files
    '*/build/*'          # Build artifacts
    '/usr/*'             # System libraries
    '*/boost/*'          # Boost headers
    '*/opencv*'          # OpenCV headers
)
```

### In Code

Use `LCOV_EXCL` markers to exclude specific lines:

```cpp
// LCOV_EXCL_START
void debugOnlyFunction() {
    // This won't be counted in coverage
}
// LCOV_EXCL_STOP

// Or single line:
unreachablecode(); // LCOV_EXCL_LINE
```

## Troubleshooting

### Issue: "gcov not found"

```bash
sudo apt-get install gcc gcov
```

### Issue: "lcov not found"

```bash
sudo apt-get install lcov
```

### Issue: Zero coverage reported

**Causes:**
1. Tests didn't run successfully
2. Wrong build type (must be Debug)
3. CODE_COVERAGE flag not set

**Solution:**

```bash
# Check test execution
./build_coverage/aprapipesut --log_level=all

# Verify build flags
cmake -L build_coverage/ | grep COVERAGE
# Should show: CODE_COVERAGE:BOOL=ON
```

### Issue: Coverage data from old runs

```bash
# Clean coverage data
cd build_coverage
lcov --directory . --zerocounters

# Or rebuild from scratch
cd ..
rm -rf build_coverage
./generate_coverage.sh
```

### Issue: HTML report not generated

```bash
# Check for genhtml
which genhtml

# Manually generate HTML
cd build_coverage
genhtml coverage.info --output-directory coverage
```

### Issue: CUDA code showing in coverage

CUDA code is excluded by building with `ENABLE_CUDA=OFF` for coverage. This is intentional as:
- CUDA code coverage requires different tools
- Most CUDA code is vendor-specific
- Focus on host-side logic first

## Advanced Usage

### Filtering Coverage by Pattern

```bash
# Generate coverage only for specific modules
lcov --capture --directory . \
     --output-file coverage_filtered.info \
     --include "*/src/Module.cpp" \
     --include "*/src/Pipeline.cpp"

genhtml coverage_filtered.info --output-directory coverage_modules
```

### Coverage Diff Between Branches

```bash
# Generate baseline coverage (main branch)
git checkout main
./generate_coverage.sh
cp build_coverage/coverage.info coverage_main.info

# Generate feature coverage
git checkout feature-branch
./generate_coverage.sh clean

# Compare
lcov --diff coverage_main.info build_coverage/coverage.info \
     --output-file coverage_diff.info

genhtml coverage_diff.info --output-directory coverage_diff
```

### Integration with IDEs

#### Visual Studio Code

Install extension: "Coverage Gutters"

1. Generate `coverage.info`
2. Open VS Code in project root
3. Extension automatically shows coverage in editor

#### CLion / IntelliJ

1. Build with coverage: `make coverage`
2. Run → Show Coverage Data
3. Load `build_coverage/coverage.info`

## Best Practices

1. **Run coverage locally before PR**: Catch untested code early
2. **Focus on critical paths**: 100% coverage isn't always necessary
3. **Review uncovered lines**: Understand why they're not tested
4. **Add tests incrementally**: Don't aim for perfection immediately
5. **Check coverage trends**: Monitor coverage over time
6. **Exclude intentionally**: Mark debug/unreachable code explicitly

## Additional Resources

- [gcov Documentation](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html)
- [lcov README](http://ltp.sourceforge.net/coverage/lcov.php)
- [Codecov Documentation](https://docs.codecov.com/)
- [Code Coverage Best Practices](https://testing.googleblog.com/2020/08/code-coverage-best-practices.html)

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing coverage-related issues
- Contact the development team

---

**Last Updated**: 2025-11-26
**Maintainer**: ApraPipes Development Team
