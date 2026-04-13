# H265 Mp4Reader Support + Codec Auto-Detection

**Branch:** `bug_fix/h265-mp4reader-support`
**Base branch:** `NVR_Snapshot_JP6`
**Repo:** `/home/developer/ws_yash/ApraPipes_SNAP`

---

## Phase 0 — Branch Setup (cherry-pick H265Decoder onto NVR_Snapshot_JP6)

### Task 0.1 — Cherry-pick H265Decoder commits from feature/h265-decoder-v4l2
**Type:** task
**Done when:** All commits unique to `feature/h265-decoder-v4l2` (H265Decoder implementation: H265Metadata, H265Utils, H265Decoder, CMakeLists, tests) are cherry-picked onto `bug_fix/h265-mp4reader-support` (based on `NVR_Snapshot_JP6`). Resolve any conflicts. Verify `git log --oneline NVR_Snapshot_JP6..HEAD` shows the H265 decoder commits.

Steps:
```bash
cd /home/developer/ws_yash/ApraPipes_SNAP
# Identify commits unique to feature/h265-decoder-v4l2 not in NVR_Snapshot_JP6
git log --oneline NVR_Snapshot_JP6..feature/h265-decoder-v4l2
# Cherry-pick them oldest-first onto the new branch
git cherry-pick <oldest-sha>^..<newest-sha>
```

### VERIFY 0 — Cherry-pick checkpoint
**Type:** verify
**Done when:** Cherry-pick complete, `git log --oneline NVR_Snapshot_JP6..HEAD` confirms all H265Decoder commits present, no conflicts. PM reviews before proceeding.

---

## Phase 1 — Code Implementation

### Task 1.1 — Confirm libmp4 codec detection call site
**Type:** task
**Files:** `base/src/Mp4ReaderSource.cpp` (~line 1281), `_build/vcpkg_installed/arm64-linux/include/libmp4.h`
**Done when:** Confirm that at line ~1281 in Mp4ReaderSource.cpp, after `mp4_demux_get_track_video_decoder_config()` is called, `vdc->codec` can be checked against `MP4_VIDEO_CODEC_HEVC` (defined in libmp4.h line 91) to detect H265 tracks. Add a brief code comment at the detection point explaining the field. No code changes needed for this task — verification and documentation only.

### Task 1.2 — Add headers and pin member
**Type:** task
**Files:** `base/src/Mp4ReaderSource.cpp`, `base/include/Mp4ReaderSource.h`
**Done when:**
- `#include "H265Metadata.h"` and `#include "H265Utils.h"` added to Mp4ReaderSource.cpp
- `std::string h265ImagePinId;` added to `Mp4ReaderDetailAbs` base class
- `std::string h265ImagePinId;` added to `Mp4ReaderSource.h` class declaration alongside `h264ImagePinId`

### Task 1.3 — Implement Mp4ReaderDetailH265 class
**Type:** task
**Files:** `base/src/Mp4ReaderSource.cpp`
**Done when:** New class `Mp4ReaderDetailH265` added after `Mp4ReaderDetailH264`, implementing:
- Constructor mirrors H264 constructor
- `readVpsSpsPps()` — calls `mp4_demux_get_track_video_decoder_config()`, reads `vdc->hevc.vps`, `vdc->hevc.sps`, `vdc->hevc.pps` and their sizes
- `prependVpsSpsPps()` — prepends 3 NAL units (VPS first, then SPS, then PPS) before IDR frames
- `setMetadata()` — creates `H265Metadata(mWidth, mHeight)` output
- `produceFrames()` — uses `H265Utils::getNALUType()` and `H265Utils::isIDR()`; calls `prependVpsSpsPps()` on IDR frames
- `mp4Seek()` — same as H264: calls `mp4_demux_seek()`
- `getGop()` — same sync-sample calculation as H264
- `sendEndOfStream()` — same EoS logic as H264
- Uses `h265ImagePinId` for output pin references

### Task 1.4 — Auto-detect codec in init() and update addOutPutPin() / validateOutputPins()
**Type:** task
**Files:** `base/src/Mp4ReaderSource.cpp`
**Done when:**
- `init()` auto-detects track codec using libmp4 API (from Task 1.1 research) and instantiates `Mp4ReaderDetailH265` for HEVC tracks, `Mp4ReaderDetailH264` for H264 tracks (existing path unchanged). Explicit `outputFormat` override still respected.
- `addOutPutPin()` has `HEVC_DATA` branch that assigns `h265ImagePinId`
- `validateOutputPins()` whitelist includes `HEVC_DATA`
- Declarative auto-pin block adds `"h265"` / `"hevc"` explicit branch AND `"auto"` / empty-string branch that queries codec type from the container

### VERIFY 1 — Code review checkpoint
**Type:** verify
**Done when:** All Phase 1 tasks committed and pushed. PM dispatches 🟩 cicd-reviewer for code review. APPROVED required to proceed to Phase 2.

---

## Phase 2 — Build

### Task 2.1 — Fix build environment and compile
**Type:** task
**Done when:**
```bash
sudo ln -sf /usr/lib/aarch64-linux-gnu/libgtk-3.so.0 /usr/lib/aarch64-linux-gnu/libgtk-3.so 2>/dev/null || true
cd /home/developer/ws_yash/ApraPipes_SNAP/_build
cmake --build . --target aprapipesut -j$(nproc)
```
Exits with zero errors. Fix any compilation errors before marking done.

### VERIFY 2 — Build review checkpoint
**Type:** verify
**Done when:** Binary built successfully with zero errors. PM dispatches 🟩 cicd-reviewer. APPROVED required to proceed to Phase 3.

---

## Phase 3 — Tests

### Task 3.1 — Run H265 tests and fix failures
**Type:** task
**Done when:**
```bash
cd /home/developer/ws_yash/ApraPipes_SNAP/_build
./aprapipesut --gtest_filter=*h265*
```
All 3 H265 test cases pass: `mp4reader_h265decoder_eglrenderer`, `mp4reader_h265decoder_extsink`, `mp4reader_h265decoder_statsink`.

If test video files are git-LFS pointer stubs (< 100 KB):
```bash
ffmpeg -f lavfi -i testsrc=size=640x480:rate=30 -t 2 -c:v libx265 -preset ultrafast /path/to/test_h265.mp4
```
Update test data file path references accordingly.

### Task 3.2 — H264 regression check
**Type:** task
**Done when:**
```bash
./aprapipesut --gtest_filter=*h264*
```
All existing H264 tests pass unchanged.

### VERIFY 3 — Final review checkpoint
**Type:** verify
**Done when:** All H265 tests pass, all H264 regression tests pass. PM dispatches 🟩 cicd-reviewer for final sign-off. APPROVED = sprint complete.
