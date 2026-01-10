# Declarative Pipeline Integration Tests

> **Test Results and Issues Documentation**
> Last Updated: 2026-01-10

## Test Environment
- Platforms: Linux x64, macOS, Windows
- Branch: feat-declarative-pipeline-v2
- Registered Modules: 37 cross-platform + 2 Linux + 13 CUDA

---

## Test Script

Run integration tests with:
```bash
# Full test (runs pipelines)
./scripts/test_declarative_pipelines.sh

# Validate only (no runtime execution)
./scripts/test_declarative_pipelines.sh --validate-only

# Verbose output
./scripts/test_declarative_pipelines.sh --verbose

# Test specific pipeline
./scripts/test_declarative_pipelines.sh --pipeline "14_affine"
```

### Runtime Modes
- **Node.js** (preferred): Uses `aprapipes.node` addon via `pipeline_test_runner.js`
- **CLI** (fallback): Uses `aprapipes_cli` executable

The script auto-detects available runtimes and selects the best option.

### Linux Notes
On Linux, the script automatically preloads GTK3 for the Node.js addon:
```bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libgtk-3.so.0
```

---

## Current Test Results

| # | Example | Validate | Run | Status | Notes |
|---|---------|----------|-----|--------|-------|
| 01 | simple_source_sink | ✅ | ✅ | **PASS** | TestSignalGenerator → StatSink |
| 02 | three_module_chain | ✅ | ✅ | **PASS** | Three-module chain |
| 03 | split_pipeline | ✅ | ✅ | **PASS** | Split to multiple sinks |
| 04 | ptz_with_conversion | ✅ | ✅ | **PASS** | PTZ with color conversion |
| 05 | transform_ptz_with_conversion | ✅ | ✅ | **PASS** | Transform + PTZ chain |
| 06 | face_detector_with_conversion | ✅ | ⏭️ | **SKIP** | Requires face detection model |
| 09 | face_detection_demo | ✅ | ⏭️ | **SKIP** | Requires face detection model |
| 10 | bmp_converter_pipeline | ✅ | ✅ | **PASS** | BMP output files verified |
| 14 | affine_transform_chain | ✅ | ✅ | **PASS** | Affine transform chain |
| 14 | affine_transform_demo | ✅ | ⏭️† | **SKIP** | †Node.js only - CLI works |

**Summary:** 7 tests pass, 0 fail, 3 skipped (2 model-dependent + 1 Node.js limitation)

---

## Known Issues

### Issue #7: ImageEncoderCV Node.js Segfault
**Status:** Documented - Known Limitation
**Affected Pipeline:** `14_affine_transform_demo`
**Symptoms:** SIGSEGV crash in `__longjmp_chk` during `cv::imencode`

**Root Cause:**
libjpeg uses setjmp/longjmp for error handling, which conflicts with Node.js threading model. The crash occurs in OpenCV's JPEG encoding when called from a Node.js addon.

**Workaround:**
- The test script skips this pipeline when using Node.js runtime
- CLI runtime handles the pipeline correctly
- Other pipelines using `ImageDecoderCV` (decoding) work fine in Node.js

**Resolution Options:**
1. Use PNG encoding instead of JPEG (no setjmp issue)
2. Run JPEG encoding pipelines with CLI only
3. Build OpenCV with libjpeg-turbo (may help)

---

## Historical Issues (Resolved)

### Issue #1: Connection Syntax
**Status:** Fixed
**Description:** Initial JSON used `source`/`destination` but parser expects `from`/`to`
**Fix:** Use correct syntax: `"from": "module1"`, `"to": "module2"`

### Issue #2: TestSignalGenerator Missing Properties
**Status:** Fixed
**Description:** TestSignalGenerator requires `width` and `height` but registration didn't declare them
**Fix:** Added `.intProp("width", ...)` and `.intProp("height", ...)` to registration

### Issue #3: Module Sets Up Own Pins - Factory Adds Duplicates
**Status:** Fixed
**Description:** Some modules set up their output pins in the constructor. ModuleFactory was adding duplicate pins.
**Fix:** Modified setupOutputPins() to check for existing pins before adding new ones

### Issue #4: Connection::parse Not Handling Module-Only Syntax
**Status:** Fixed
**Description:** JSON connections like `"from": "generator"` resulted in empty `from_module` field
**Fix:** Updated Connection::parse() to set module name even without dot

### Issue #5: Frame Type Mismatch
**Status:** Fixed
**Description:** TestSignalGenerator outputs RawImagePlanar, but transforms expected RawImage
**Fix:** Added ColorConversion module to convert between frame types

### Issue #6: Split/Merge Dynamic Pins
**Status:** Fixed
**Description:** Split module creates pins dynamically, conflicted with factory-created pins
**Fix:** Added DYNAMIC_PINS flag handling in ModuleFactory

---

## Working Pipelines

All pipelines in `docs/declarative-pipeline/examples/working/` have been tested and verified.

### Pipeline Descriptions

| Pipeline | Description |
|----------|-------------|
| 01_simple_source_sink | Minimal: TestSignalGenerator → StatSink |
| 02_three_module_chain | Chain: TestSignalGenerator → ValveModule → StatSink |
| 03_split_pipeline | Split: TestSignalGenerator → Split → 2x StatSink |
| 04_ptz_with_conversion | PTZ: Generator → ColorConversion → VirtualPTZ → StatSink |
| 05_transform_ptz_with_conversion | Transform: Generator → Convert → AffineTransform → PTZ → Sink |
| 10_bmp_converter_pipeline | Output: Generator → Convert → BMPConverter → FileWriter |
| 14_affine_transform_chain | Affine: Generator → Convert → AffineTransform → StatSink |
| 14_affine_transform_demo | JPEG Output: Generator → Convert → Affine → ImageEncoder → FileWriter |

---
