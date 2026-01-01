# Declarative Pipeline Integration Tests

> **Test Results and Issues Documentation**
> Last Updated: 2025-12-31

## Test Environment
- Platform: macOS (Darwin 24.6.0)
- Branch: feat-declarative-pipeline
- Registered Modules: 20

---

## Issues Found During Integration Testing

### Issue #1: TOML Connection Syntax
**Status:** Fixed
**Description:** Initial TOML used `source`/`destination` but parser expects `from`/`to`
**Fix:** Use correct syntax: `from = "module1"`, `to = "module2"`

### Issue #2: TestSignalGenerator Missing Properties in Registration
**Status:** Fixed
**Description:** TestSignalGenerator requires `width` and `height` but registration didn't declare them
**Error:** `Missing required properties: width, height`
**Fix:** Added `.intProp("width", ...)` and `.intProp("height", ...)` to registration

### Issue #3: Module Sets Up Own Pins - Factory Adds Duplicates
**Status:** Fixed
**Description:** Some modules (like TestSignalGenerator) set up their output pins in the constructor with specialized metadata (e.g., RawImagePlanarMetadata with dimensions). ModuleFactory was then adding duplicate pins with generic metadata.
**Error:** `AIP_PINS_VALIDATION_FAILED (7717)` - Module's validateOutputPins() failed because pin count was wrong
**Root Cause:** ModuleFactory.setupOutputPins() unconditionally added output pins without checking if the module already had pins
**Fix:** Modified setupOutputPins() to check for existing pins using getAllOutputPinsByType() before adding new ones

### Issue #4: Connection::parse Not Handling Module-Only Syntax
**Status:** Fixed
**Description:** TOML connections like `from = "generator"` (without pin specification) resulted in empty `from_module` field
**Error:** `Unknown source module: ` (empty string)
**Root Cause:** Connection::parse() only populated from_module when a dot was present; otherwise it left fields empty
**Fix:** Updated Connection::parse() to set module name even without dot, leaving pin empty (uses default)
**Also Fixed:** Updated isValid() to only require module names, not pins

### Issue #5: Frame Type Mismatch - RawImagePlanar vs RawImage
**Status:** Open - Needs Module Registration Update
**Description:** TestSignalGenerator outputs RawImagePlanar, but most transform modules (VirtualPTZ, ColorConversion) expect RawImage. Additionally, analytics modules (FaceDetectorXform, QRReader) are registered as accepting RawImagePlanar but actually validate for RawImage.
**Error:** `input frameType is expected to be RAW_IMAGE. Actual<3>` (3 = RAW_IMAGE_PLANAR)
**Affected Modules:** VirtualPTZ, ColorConversion, FaceDetectorXform (registration says RawImagePlanar but validates for RawImage)
**Proposed Fix:**
1. Fix registrations to match actual module validation requirements
2. Consider adding type conversion modules (RawImagePlanar ↔ RawImage)

### Issue #6: Split/Merge Modules Dynamically Create Pins
**Status:** Open - Needs Factory Enhancement
**Description:** Split module creates output pins dynamically when `addInputPin()` is called (based on input metadata). ModuleFactory pre-creates output pins based on registration, then module adds more, causing validation failure.
**Error:** `Expected <2>. But Actual<3>` in validateOutputPins()
**Root Cause:** Factory creates pins from registration metadata, then module creates pins when connected
**Proposed Fix:** Options:
1. Don't pre-create pins for modules that use DYNAMIC_PINS flag
2. Check if module has dynamic pin creation logic and skip factory pin setup
3. Add a registration flag indicating "module creates its own pins"

---

## Test Results

| # | Example | Validate | Run | Status | Notes |
|---|---------|----------|-----|--------|-------|
| 01 | simple_source_sink | ✅ | ✅ | **PASS** | TestSignalGenerator → StatSink |
| 02 | three_module_chain | ✅ | ✅ | **PASS** | TestSignalGenerator → ValveModule → StatSink |

---

## Working Examples

### 01_simple_source_sink.toml
- **Pipeline:** TestSignalGenerator → StatSink
- **Description:** Minimal pipeline to test basic module connectivity
- **Validates:** ✅
- **Runs:** ✅ (generates test frames, StatSink collects stats)

### 02_three_module_chain.toml
- **Pipeline:** TestSignalGenerator → ValveModule → StatSink
- **Description:** Three-module linear chain
- **Validates:** ✅
- **Runs:** ✅ (valve allows all frames through by default)

---

## Not Working Examples

### 03_transform_ptz.toml
- **Pipeline:** TestSignalGenerator → VirtualPTZ → StatSink
- **Issue:** Frame type mismatch (RawImagePlanar vs RawImage)
- **Error:** `input frameType is expected to be RAW_IMAGE. Actual<3>`
- **Related Issue:** #5

### 03_face_detector.toml
- **Pipeline:** TestSignalGenerator → FaceDetectorXform → StatSink
- **Issue:** Frame type mismatch (registration says RawImagePlanar but module validates for RawImage)
- **Error:** `input frameType is expected to be Raw_Image. Actual<3>`
- **Related Issue:** #5

### 03_split_pipeline.toml
- **Pipeline:** TestSignalGenerator → Split → 2x StatSink
- **Issue:** Split module creates pins dynamically, conflicts with factory-created pins
- **Error:** `Expected <2>. But Actual<3>` in validateOutputPins
- **Related Issue:** #6

---

## Modules Tested

| Module | As Source | As Transform | As Sink | Notes |
|--------|-----------|--------------|---------|-------|
| TestSignalGenerator | ✅ | - | - | Works, requires width/height |
| StatSink | - | - | ✅ | Works |
| ValveModule | - | ✅ | - | Works |
| VirtualPTZ | - | ❌ | - | Expects RawImage, not RawImagePlanar |
| FaceDetectorXform | - | ❌ | - | Registration says RawImagePlanar but validates for RawImage |
| Split | - | ❌ | - | Dynamic pin creation conflicts with factory |

---

## Summary

**Working Pipelines:** 2
**Not Working Pipelines:** 3

**Primary Blockers:**
1. Frame type compatibility between source (RawImagePlanar) and transforms (RawImage)
2. Dynamic pin creation in Split/Merge modules
3. Module registration mismatches with actual validation code

**Recommendation:** Fix Issue #5 (frame type registration) and Issue #6 (dynamic pins) before testing more complex pipelines.

---
