# Phase 9: Cloud Build Testing Report

## Overview

**Phase**: 9 - Cloud Build Testing for Component Presets
**Date**: 2025-01-XX
**Status**: Testing in Progress

**Objective**: Validate the component-based build system on GitHub cloud runners (windows-2022 and ubuntu-22.04) using multiple component presets to ensure CI/CD pipeline compatibility.

## Changes Implemented

### 1. Reusable Workflow Updates

#### build-test-win.yml
- **Added**: `preset` parameter (string, optional, default: '')
- **Added**: "Set component preset" step with PowerShell mapping:
  - `minimal` → `CORE`
  - `video` → `CORE;VIDEO;IMAGE_PROCESSING`
  - empty/`full` → `ALL`
- **Modified**: CMake configure command to include `-DENABLE_COMPONENTS=${{env.COMPONENTS}}`

#### build-test-lin.yml
- **Same changes as build-test-win.yml**
- Ensures consistent behavior across Windows and Linux platforms

### 2. CI Workflow Matrix Strategy

#### CI-Win-NoCUDA.yml
- **Added**: Matrix strategy with presets: `['minimal', 'video', 'full']`
- **Applied to**:
  - `win-nocuda-build-prep` job
  - `win-nocuda-build-test` job
  - `win-nocuda-publish` job
- **Updated**: `flav` parameter to `Win-nocuda-${{ matrix.preset }}`
- **Added**: `fail-fast: false` to allow all matrix builds to complete

#### CI-Linux-NoCUDA.yml
- **Added**: Matrix strategy with presets: `['minimal', 'video', 'full']`
- **Applied to**:
  - `linux-nocuda-build-test` job
  - `linux-nocuda-publish` job
- **Updated**: `flav` parameter to `Linux-nocuda-${{ matrix.preset }}`
- **Added**: `fail-fast: false` to allow all matrix builds to complete

## Test Matrix

| Platform | Runner | Preset | Components | Expected Build Time |
|----------|--------|--------|------------|---------------------|
| Windows | windows-2022 | minimal | CORE | ~10-15 min |
| Windows | windows-2022 | video | CORE;VIDEO;IMAGE_PROCESSING | ~25-35 min |
| Windows | windows-2022 | full | ALL | ~45-60 min |
| Linux | ubuntu-22.04 | minimal | CORE | ~8-12 min |
| Linux | ubuntu-22.04 | video | CORE;VIDEO;IMAGE_PROCESSING | ~20-30 min |
| Linux | ubuntu-22.04 | full | ALL | ~35-50 min |

**Total Matrix Combinations**: 6 (2 platforms × 3 presets)

## Expected Outcomes

### Minimal Preset (CORE)
- ✅ Fast build (~10-15 min on Windows, ~8-12 min on Linux)
- ✅ Minimal vcpkg dependencies (Boost, basic libraries)
- ✅ Core pipeline functionality tests pass
- ✅ No video/image processing modules included

### Video Preset (CORE;VIDEO;IMAGE_PROCESSING)
- ✅ Medium build time (~25-35 min on Windows, ~20-30 min on Linux)
- ✅ FFmpeg, OpenH264, OpenCV dependencies installed
- ✅ Video processing tests pass
- ✅ Image processing tests pass
- ✅ No CUDA components included

### Full Preset (ALL)
- ✅ Longest build time (~45-60 min on Windows, ~35-50 min on Linux)
- ✅ All non-CUDA dependencies installed
- ✅ All non-CUDA tests pass
- ✅ Backward compatible with previous CI setup

## How to Trigger Tests

### Automatic Triggers
- **Push to main branch**: All 6 matrix combinations run automatically
- **Pull request to main**: All 6 matrix combinations run automatically

### Manual Trigger (via GitHub UI)
1. Navigate to: https://github.com/Apra-Labs/ApraPipes/actions
2. Select "CI-Win-NoCUDA" or "CI-Linux-NoCUDA" workflow
3. Click "Run workflow" → Select branch → "Run workflow"

### Monitor Test Progress
```bash
# View workflow status
gh run list --workflow=CI-Win-NoCUDA.yml --limit 5
gh run list --workflow=CI-Linux-NoCUDA.yml --limit 5

# View specific run details
gh run view <run-id> --log
```

## Test Results

### Windows-2022 (No CUDA)

#### Minimal Preset
- **Status**: ⏳ Pending / ✅ Passed / ❌ Failed
- **Build Time**: [To be filled]
- **Test Results**: [To be filled]
- **Artifacts**: [Link to be added]
- **Notes**: [Any observations]

#### Video Preset
- **Status**: ⏳ Pending / ✅ Passed / ❌ Failed
- **Build Time**: [To be filled]
- **Test Results**: [To be filled]
- **Artifacts**: [Link to be added]
- **Notes**: [Any observations]

#### Full Preset
- **Status**: ⏳ Pending / ✅ Passed / ❌ Failed
- **Build Time**: [To be filled]
- **Test Results**: [To be filled]
- **Artifacts**: [Link to be added]
- **Notes**: [Any observations]

### Ubuntu-22.04 (No CUDA)

#### Minimal Preset
- **Status**: ⏳ Pending / ✅ Passed / ❌ Failed
- **Build Time**: [To be filled]
- **Test Results**: [To be filled]
- **Artifacts**: [Link to be added]
- **Notes**: [Any observations]

#### Video Preset
- **Status**: ⏳ Pending / ✅ Passed / ❌ Failed
- **Build Time**: [To be filled]
- **Test Results**: [To be filled]
- **Artifacts**: [Link to be added]
- **Notes**: [Any observations]

#### Full Preset
- **Status**: ⏳ Pending / ✅ Passed / ❌ Failed
- **Build Time**: [To be filled]
- **Test Results**: [To be filled]
- **Artifacts**: [Link to be added]
- **Notes**: [Any observations]

## Issues Encountered

### Build Issues
[To be filled with any build failures, dependency issues, or CMake configuration errors]

### Test Failures
[To be filled with any test failures specific to component presets]

### Infrastructure Issues
[To be filled with any GitHub Actions runner issues, timeout problems, or caching issues]

## Validation Criteria

- [ ] All 6 matrix combinations complete successfully
- [ ] Minimal preset builds faster than video preset
- [ ] Video preset builds faster than full preset
- [ ] Component-specific tests run only in relevant presets
- [ ] vcpkg caching works correctly for each preset
- [ ] Artifacts uploaded with correct naming (includes preset in name)
- [ ] Test results published correctly for each matrix combination
- [ ] No unexpected dependency installations in minimal/video presets

## Performance Comparison

### Build Time Reduction
| Platform | Full Build | Minimal Build | Time Saved | % Reduction |
|----------|------------|---------------|------------|-------------|
| Windows | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Linux | [To be filled] | [To be filled] | [To be filled] | [To be filled] |

### vcpkg Cache Performance
| Platform | Preset | Cache Hit Rate | vcpkg Install Time |
|----------|--------|----------------|-------------------|
| Windows | minimal | [To be filled] | [To be filled] |
| Windows | video | [To be filled] | [To be filled] |
| Windows | full | [To be filled] | [To be filled] |
| Linux | minimal | [To be filled] | [To be filled] |
| Linux | video | [To be filled] | [To be filled] |
| Linux | full | [To be filled] | [To be filled] |

## Recommendations

[To be filled after test results are analyzed]

### Short-term
- [ ] [Recommendation 1]
- [ ] [Recommendation 2]

### Long-term
- [ ] [Recommendation 1]
- [ ] [Recommendation 2]

## Conclusion

[To be filled with overall assessment of Phase 9 cloud testing results]

## Next Steps

1. Monitor initial test runs on next push/PR
2. Analyze build times and cache performance
3. Document any issues and create fixes
4. Prepare Phase 10 planning based on results

## Related Documentation

- [COMPONENT_REFACTORING_LOG.md](./COMPONENT_REFACTORING_LOG.md) - Complete refactoring history
- [TESTING_PHASE5_REPORT.md](./TESTING_PHASE5_REPORT.md) - Windows CUDA preset testing
- [.github/workflows/CI-Win-NoCUDA.yml](./.github/workflows/CI-Win-NoCUDA.yml) - Windows cloud workflow
- [.github/workflows/CI-Linux-NoCUDA.yml](./.github/workflows/CI-Linux-NoCUDA.yml) - Linux cloud workflow
- [.github/workflows/build-test-win.yml](./.github/workflows/build-test-win.yml) - Reusable Windows workflow
- [.github/workflows/build-test-lin.yml](./.github/workflows/build-test-lin.yml) - Reusable Linux workflow

## Appendix: GitHub Actions Artifacts

### Expected Artifacts per Matrix Combination
- `TestResults_Win-nocuda-minimal` / `TestResults_Linux-nocuda-minimal`
- `TestResults_Win-nocuda-video` / `TestResults_Linux-nocuda-video`
- `TestResults_Win-nocuda-full` / `TestResults_Linux-nocuda-full`
- `BuildLogs_Win-nocuda-minimal_*` / `BuildLogs_Linux-nocuda-minimal`
- `BuildLogs_Win-nocuda-video_*` / `BuildLogs_Linux-nocuda-video`
- `BuildLogs_Win-nocuda-full_*` / `BuildLogs_Linux-nocuda-full`

### Artifact Retention
- Test results: 30 days
- Build logs: 30 days

---

**Document Version**: 1.0
**Last Updated**: 2025-01-XX
**Author**: Component-based build system refactoring team
