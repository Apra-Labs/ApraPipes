# Code Coverage Setup - Quick Reference

This document provides a quick reference for the code coverage setup that has been added to ApraPipes.

## What Was Added

### 1. CMake Coverage Support
- **File**: `cmake/CodeCoverage.cmake`
  - Complete CMake module for code coverage
  - Supports gcov/lcov toolchain
  - Provides `setup_target_for_coverage_lcov()` function

- **Modified**: `base/CMakeLists.txt`
  - Added `CODE_COVERAGE` option (default: OFF)
  - Integrated CodeCoverage.cmake module
  - Created `coverage` make target
  - Configured exclusion patterns

### 2. CI/CD Integration
- **File**: `.github/workflows/code-coverage.yml`
  - Automated coverage generation on push/PR
  - Runs on: main, master, develop branches
  - Uploads to Codecov
  - Creates artifacts (HTML reports, coverage.info)
  - Posts PR comments with coverage summary

### 3. Local Development Tools
- **Script**: `generate_coverage.sh`
  - One-command coverage generation
  - Automatic dependency checking
  - Clean build support
  - HTML report generation

### 4. Configuration Files
- **File**: `codecov.yml`
  - Codecov service configuration
  - Coverage thresholds and targets
  - Exclusion patterns
  - PR comment formatting

### 5. Documentation
- **File**: `docs/CODE_COVERAGE.md`
  - Complete user guide
  - Prerequisites and setup
  - Troubleshooting section
  - Best practices
  - Advanced usage examples

- **Modified**: `README.md`
  - Added coverage badges
  - Quick links to coverage reports
  - Link to detailed documentation

## Quick Start

### For Developers (Local)

```bash
# Generate coverage report locally
./generate_coverage.sh

# View the report
xdg-open build_coverage/coverage/index.html
```

### For CI/CD

Coverage is automatically generated and published on every push/PR. No manual action needed!

**View Coverage:**
- [Codecov Dashboard](https://codecov.io/gh/Apra-Labs/ApraPipes)
- GitHub Actions → Coverage Workflow → Artifacts

### For Manual CMake Build

```bash
mkdir -p build_coverage && cd build_coverage

cmake -B . \
    -DENABLE_LINUX=ON \
    -DENABLE_CUDA=OFF \
    -DCODE_COVERAGE=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
    ../base

cmake --build . --config Debug -j $(nproc)
make coverage
```

## File Structure

```
ApraPipes/
├── cmake/
│   └── CodeCoverage.cmake          # CMake coverage module
├── .github/workflows/
│   └── code-coverage.yml           # GitHub Actions workflow
├── docs/
│   └── CODE_COVERAGE.md            # Detailed documentation
├── base/
│   └── CMakeLists.txt              # Modified for coverage
├── codecov.yml                     # Codecov configuration
├── generate_coverage.sh            # Local coverage script
├── COVERAGE_SETUP.md               # This file
└── README.md                       # Updated with badges
```

## Prerequisites

### Ubuntu/Debian
```bash
sudo apt-get install lcov gcov gcovr
```

### Build Dependencies
Standard ApraPipes build dependencies (see main README.md)

## Coverage Exclusions

The following are automatically excluded from coverage:
- Test files (`base/test/*`)
- Third-party libraries (`thirdparty/*`)
- VCPKG packages (`vcpkg/*`)
- Build artifacts (`build/*`)
- System headers (`/usr/*`)
- Boost headers
- OpenCV headers

## Integration Points

### Codecov Setup (Optional but Recommended)

1. Visit https://codecov.io
2. Sign in with GitHub
3. Add the ApraPipes repository
4. Get the upload token
5. Add to GitHub Secrets as `CODECOV_TOKEN`

Once configured, every PR will show:
- Coverage percentage
- Coverage diff
- Line-by-line coverage in PR view

### IDE Integration

#### VS Code
1. Install "Coverage Gutters" extension
2. Generate coverage: `./generate_coverage.sh`
3. Extension auto-detects `coverage.info`

#### CLion
1. Build with coverage enabled
2. Run → Show Coverage Data
3. Load `build_coverage/coverage.info`

## Usage Scenarios

### Scenario 1: Pre-Commit Check
```bash
# Before committing, check coverage
./generate_coverage.sh
# Review uncovered lines in HTML report
xdg-open build_coverage/coverage/index.html
```

### Scenario 2: PR Review
- CI automatically generates coverage
- View coverage diff in PR comments
- Check Codecov link for detailed analysis

### Scenario 3: Coverage Investigation
```bash
# Generate coverage
./generate_coverage.sh

# Check summary
lcov --summary build_coverage/coverage.info

# View specific file
xdg-open build_coverage/coverage/base/src/Module.cpp.gcov.html
```

### Scenario 4: Module-Specific Coverage
```bash
# Filter for specific modules
cd build_coverage
lcov --capture --directory . \
     --output-file coverage_mp4.info \
     --include "*/Mp4*.cpp"

genhtml coverage_mp4.info --output-directory coverage_mp4
xdg-open coverage_mp4/index.html
```

## Troubleshooting

### Issue: Zero coverage reported
**Solution**: Ensure you built with `-DCODE_COVERAGE=ON` and `-DCMAKE_BUILD_TYPE=Debug`

### Issue: Tests not running
**Solution**: Check test executable: `./build_coverage/aprapipesut --log_level=all`

### Issue: Old coverage data
**Solution**: Clean and rebuild:
```bash
rm -rf build_coverage
./generate_coverage.sh
```

### Issue: Missing dependencies
**Solution**: Install coverage tools:
```bash
sudo apt-get install lcov gcov
```

## Next Steps

1. **Enable Codecov**
   - Add `CODECOV_TOKEN` to GitHub secrets
   - First run will establish baseline

2. **Review Initial Coverage**
   - Check which modules have low coverage
   - Prioritize critical components

3. **Improve Coverage**
   - Add tests for uncovered code
   - Focus on core functionality first
   - Target 70%+ line coverage

4. **Monitor Trends**
   - Review Codecov graphs weekly
   - Watch for coverage drops in PRs
   - Celebrate coverage improvements!

## Support

- **Full Documentation**: `docs/CODE_COVERAGE.md`
- **Issues**: GitHub Issues
- **Questions**: Contact maintainers

## Coverage Targets

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Line Coverage | > 60% | > 70% | > 80% |
| Function Coverage | > 50% | > 65% | > 75% |
| Branch Coverage | > 40% | > 55% | > 70% |

**Note**: Focus on critical paths. 100% coverage is rarely necessary.

## Maintenance

### Regular Tasks
- Review coverage reports monthly
- Update exclusion patterns as needed
- Keep documentation current
- Monitor CI performance

### When to Update
- Adding new modules: Update exclusions if needed
- New dependencies: May need new exclude patterns
- CMake changes: Verify coverage build still works

## Credits

- Coverage module based on [codecov/example-cpp](https://github.com/codecov/example-cpp)
- CMake integration inspired by [larsbilke/CMake-codecov](https://github.com/bilke/cmake-modules)

---

**Setup Date**: 2025-11-26
**Last Updated**: 2025-11-26
**Maintainer**: ApraPipes Development Team
