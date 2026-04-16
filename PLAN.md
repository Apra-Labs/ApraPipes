# PLAN.md — VideoDecoder Unified Module

## Branch
`feat/video-decoder-unified` (from `NVR_Snapshot_JP6`)

## Overview
Build a unified `VideoDecoder` module that auto-detects H264 or H265 codec from incoming frame types and delegates to the existing `h264DecoderV4L2Helper` backend with the appropriate `V4L2_PIX_FMT_*`. A new `SOS_FRAME` frame type enables mid-stream codec switching: sources emit `EOS_FRAME` then `SOS_FRAME` to signal a codec change, and `VideoDecoder` flushes the old backend and re-initializes a new one without rebuilding the pipeline.

## Investigation Findings

> **CRITICAL: Base branch `NVR_Snapshot_JP6` is missing H265 infrastructure.**

The following files exist ONLY on `bug_fix/h265-mp4reader-support` and are NOT merged into the base branch:

| Missing File | Purpose |
|---|---|
| `base/include/H265Decoder.h` | H265 decoder module header |
| `base/src/H265Decoder.cpp` | H265 decoder module impl |
| `base/include/H265Metadata.h` | H265 frame metadata class |
| `base/include/H265Utils.h` | H265 NAL parsing utilities |
| `base/src/H265Utils.cpp` | H265 NAL parsing impl |
| `data/h265_bunny_30frames.mp4` | H265 test data file |

Additionally, the V4L2 helper on the base branch:
- `h264DecoderV4L2Helper::init()` signature is `bool init(std::function<void(frame_sp&)> send, std::function<frame_sp()> makeFrame)` — **no `decode_pixfmt` parameter**.
- `initializeDecoder()` hardcodes `ctx.decode_pixfmt = V4L2_PIX_FMT_H264` at line 1326.
- The `bug_fix/h265-mp4reader-support` branch added the `uint32_t decode_pixfmt = V4L2_PIX_FMT_H264` parameter to `init()`.

### Other findings (stable on base branch)
1. **FrameType enum** — `base/include/FrameMetadata.h:29`. Values end at `TEXT`. `HEVC_DATA` already exists (value 20). No `SOS_FRAME`.
2. **H264Decoder** — `base/include/H264Decoder.h`, `base/src/H264Decoder.cpp`. Uses `h264DecoderV4L2Helper` on ARM64. Props: `lowerWaterMark`, `upperWaterMark`.
3. **EoSFrame** — `base/include/Frame.h:53`. Types: `GENERAL`, `MP4_PLYB_EOS`, `MP4_SEEK_EOS`.
4. **Module base** — `processSOS(frame_sp&)`, `shouldTriggerSOS()`, `processEOS(string&)`, `sendEOS()` all exist as virtual methods.
5. **Test framework** — Boost.Test. H264 test data at `data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4`.
6. **CMakeLists.txt** — `H264Decoder.cpp` at line 407 in `CUDA_IP_FILES`. `H264Decoder.h` at line 456 in `CUDA_IP_FILES_H`. ARM64 tests at line 548. Non-ARM64 CUDA tests at line 572.

---

## Phase 0 — H265 Prerequisites (Cherry-pick / Merge)

### Task 0.1 — Merge H265 infrastructure from `bug_fix/h265-mp4reader-support`
- **Action:** Merge or cherry-pick the H265 support commits from `bug_fix/h265-mp4reader-support` into this branch.
- **Required commits (from `git log`):**
  - H265Metadata, H265Utils headers and source
  - H265Decoder header and source
  - V4L2 helper `decode_pixfmt` parameter addition
  - Mp4ReaderSource H265 codec detection
  - CMakeLists.txt H265 entries
  - H265 test data file
- **Alternative:** Rebase this branch onto `bug_fix/h265-mp4reader-support` instead of `NVR_Snapshot_JP6`.
- **Risk:** Medium. Merge conflicts possible in CMakeLists.txt and V4L2 helper.
- **Decision needed:** PM should confirm whether to:
  1. Merge `bug_fix/h265-mp4reader-support` into this branch (preferred — gets all H265 infra), OR
  2. Cherry-pick only the minimal files needed (H265Metadata, H265Utils, V4L2 helper pixfmt param), OR
  3. Rebase this branch onto `bug_fix/h265-mp4reader-support`.

### VERIFY Checkpoint — Phase 0
- `cmake --build . --target aprapipesut -j$(nproc)` — zero errors
- `./aprapipesut --run_test=h264decoder_tests` — passes
- `./aprapipesut --run_test=h265decoder_tests` — passes (confirms H265 infra is working)

**STOP after Phase 0 checkpoint. Do not proceed to Phase 1 until H265 infrastructure is confirmed working.**

---

## Phase 1 — SOS_FRAME Infrastructure

### Task 1.1 — Add SOS_FRAME to FrameType enum
- **File:** `base/include/FrameMetadata.h`
- **Change:** Add `SOS_FRAME` after `TEXT` in the `FrameType` enum (value auto-assigned, will be 25).
  ```cpp
  TEXT,
  SOS_FRAME
  ```
- **Risk:** Low. Additive enum change. Existing switch-case defaults will ignore it.

### Task 1.2 — Add SOS frame payload class
- **File:** `base/include/SosFrameMetadata.h` (new)
- **Change:** Create a lightweight metadata class carrying codec type:
  ```cpp
  #pragma once
  #include "FrameMetadata.h"

  class SosFrameMetadata : public FrameMetadata {
  public:
      enum CodecType { H264 = 0, H265 = 1 };

      SosFrameMetadata(CodecType _codec)
          : FrameMetadata(FrameType::SOS_FRAME), codec(_codec) {}

      CodecType getCodec() const { return codec; }

  private:
      CodecType codec;
  };
  ```
- **Risk:** Low. New file, no existing code impact.

### Task 1.3 — Add CODEC_SWITCH_EOS to EoSFrame
- **File:** `base/include/Frame.h`
- **Change:** Add `CODEC_SWITCH_EOS` to `EoSFrame::EoSFrameType` enum after `MP4_SEEK_EOS`:
  ```cpp
  enum EoSFrameType
  {
      GENERAL = 0,
      MP4_PLYB_EOS,
      MP4_SEEK_EOS,
      CODEC_SWITCH_EOS,
  };
  ```
- **Risk:** Low. Additive enum change.

---

## Phase 2 — VideoDecoder Module

### Task 2.1 — VideoDecoder.h
- **File:** `base/include/VideoDecoder.h` (new)
- **Design:**
  ```cpp
  #pragma once
  #include "Module.h"
  #include <vector>

  class VideoDecoderProps : public ModuleProps {
  public:
      VideoDecoderProps(uint _lowerWaterMark = 300, uint _upperWaterMark = 350);
      uint lowerWaterMark;
      uint upperWaterMark;
  };

  class VideoDecoder : public Module {
  public:
      enum State { UNINIT, DECODING, FLUSHING, REINIT };

      VideoDecoder(VideoDecoderProps _props);
      virtual ~VideoDecoder();
      bool init();
      bool term();
      bool processEOS(string& pinId);

  protected:
      bool process(frame_container& frames);
      bool processSOS(frame_sp& frame);
      void addInputPin(framemetadata_sp& metadata, string& pinId);
      bool validateInputPins();
      bool validateOutputPins();
      bool shouldTriggerSOS();
      void flushQue();
      bool handleCommand(Command::CommandType type, frame_sp& frame);

  private:
      class Detail;
      boost::shared_ptr<Detail> mDetail;
      bool mShouldTriggerSOS;
      framemetadata_sp mOutputMetadata;
      std::string mOutputPinId;
      VideoDecoderProps mProps;
      State mState;
      int mCurrentCodec;  // FrameMetadata::H264_DATA or FrameMetadata::HEVC_DATA
  };
  ```
- **Key decisions:**
  - `validateInputPins()` accepts `H264_DATA` or `HEVC_DATA` (either works).
  - State machine tracks UNINIT → DECODING → FLUSHING → REINIT → DECODING.
  - Single `Detail` class wraps `h264DecoderV4L2Helper`, selects pixfmt based on codec.
- **Risk:** Medium. Core new module — needs thorough testing.

### Task 2.2 — VideoDecoder.cpp
- **File:** `base/src/VideoDecoder.cpp` (new)
- **Implementation details:**
  - `Detail` class parametrizes codec:
    - `setMetadata()` checks frame type. Uses `V4L2_PIX_FMT_H264` for H264_DATA, `V4L2_PIX_FMT_H265` for HEVC_DATA.
    - For H264: uses `H264Utils::getNALUType()`, `H264ParserUtils::parse_sps()` for resolution.
    - For H265: uses `H265Utils::getNALUType()`, `H265Utils::isIDR()`. Default resolution 1920x1080.
    - Calls `helper->init(send, makeFrame, decode_pixfmt)`.
  - `process()` state machine:
    - **UNINIT:** On first H264_DATA/HEVC_DATA frame, detect codec from `metadata->getFrameType()`, trigger SOS to init backend, set state DECODING.
    - **DECODING:** Forward frames to `mDetail->compute()`. Handle SPS/PPS/VPS saving and IDR prepend per codec type.
    - On `CODEC_SWITCH_EOS`: call `mDetail->closeAllThreads()`, set state FLUSHING.
    - **FLUSHING:** On next SOS_FRAME, read `SosFrameMetadata::getCodec()`, destroy old detail, init new backend, set state DECODING.
  - `processEOS()` — flushes backend via `closeAllThreads()`.
  - `processSOS()` — initializes or re-initializes the V4L2 helper with correct pixfmt.
  - Header prepend logic: H264 prepends SPS+PPS. H265 prepends VPS+SPS+PPS.
- **Risk:** Medium-high. The codec-switch path (FLUSHING→REINIT) needs careful V4L2 device teardown/reinit.

### Task 2.3 — CMakeLists.txt update
- **File:** `base/CMakeLists.txt`
- **Changes (line numbers refer to base branch):**
  - Add `src/VideoDecoder.cpp` after line 407 (`src/H264Decoder.cpp`) in `CUDA_IP_FILES`.
  - Add `include/VideoDecoder.h` and `include/SosFrameMetadata.h` after line 456 (`include/H264Decoder.h`) in `CUDA_IP_FILES_H`.
  - Add `test/videodecoder_tests.cpp` after line 548 (`test/h264decoder_tests.cpp`) in ARM64 test block.
  - Add `test/videodecoder_tests.cpp` after line 572 (`test/h264decoder_tests.cpp`) in non-ARM64 CUDA test block.
- **Risk:** Low. Build config only. Line numbers may shift after Phase 0 merge.

---

## Phase 3 — Tests

### Task 3.1 — videodecoder_tests.cpp
- **File:** `base/test/videodecoder_tests.cpp` (new)
- **Test suite:** `BOOST_AUTO_TEST_SUITE(videodecoder_tests)`
- **Test cases:**

  #### 3.1a — `video_decoder_h264_basic`
  Pipeline: `Mp4ReaderSource(h264 mp4) → VideoDecoder → StatSink`
  - Uses `/home/developer/ws_yash/ApraPipes_SNAP/data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4`
  - Metadata pin: `H264Metadata(0,0)` + `Mp4VideoMetadata("v_1")`
  - Connect via `getAllOutputPinsByType(FrameMetadata::H264_DATA)`
  - Run threaded 10s, stop, verify no crash.

  #### 3.1b — `video_decoder_h265_basic`
  Pipeline: `Mp4ReaderSource(h265 mp4) → VideoDecoder → StatSink`
  - Uses `/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4` (must exist after Phase 0)
  - Metadata pin: `H265Metadata(0,0)` + `Mp4VideoMetadata("v_1")`
  - Connect via `getAllOutputPinsByType(FrameMetadata::HEVC_DATA)`
  - Run threaded 10s, stop, verify no crash.

  #### 3.1c — `video_decoder_codec_switch`
  Two-phase sequential pipeline test:
  - Phase 1: Create pipeline with H264 source → VideoDecoder → StatSink. Run 5s. Stop/term.
  - Phase 2: Create new pipeline with H265 source → VideoDecoder → StatSink. Run 5s. Stop/term.
  - Verify both phases decoded frames without crash.
  - (Full mid-stream EOS/SOS injection test deferred — requires Mp4ReaderSource modifications.)

### Task 3.2 — Build and verify
- **Build:**
  ```bash
  cd /home/developer/ws_yash/ApraPipes_SNAP/_build
  cmake --build . --target aprapipesut -j$(nproc) 2>&1 | tail -50
  ```
- **Expected:** Clean compilation, no errors.
- **Test run:**
  ```bash
  ./aprapipesut --run_test=videodecoder_tests 2>&1 | tail -30
  ```
- **Expected:** `*** No errors detected`

---

## VERIFY Checkpoint — Phase 3

Before marking Phase 3 complete, ALL of the following must pass:

1. `cmake --build . --target aprapipesut -j$(nproc)` — zero errors
2. `./aprapipesut --run_test=videodecoder_tests` — `*** No errors detected`
3. `./aprapipesut --run_test=h264decoder_tests` — `*** No errors detected` (regression)
4. `./aprapipesut --run_test=h265decoder_tests` — `*** No errors detected` (regression)
5. NVDEC hardware path confirmed (no software fallback logs)

**STOP after this checkpoint and report status.**

---

## Risks

### 1. H265 infrastructure dependency (BLOCKING)
- **Impact:** Critical — VideoDecoder cannot decode H265 without H265Metadata, H265Utils, and the V4L2 helper `decode_pixfmt` parameter. None of these exist on the base branch.
- **Mitigation:** Phase 0 merges `bug_fix/h265-mp4reader-support` first. This is a hard prerequisite.

### 2. V4L2 device teardown/reinit race condition
- **Impact:** High — NVDEC device `/dev/nvhost-nvdec` may not release cleanly if `closeAllThreads()` doesn't fully drain capture thread.
- **Mitigation:** Add explicit thread join and device close verification. Test with repeated init/term cycles.

### 3. SOS_FRAME not routed through existing Module pipeline
- **Impact:** Medium — `stepNonSource()` processes EoSFrame and calls `processEOS()`, but has no special handling for SOS_FRAME. SOS_FRAME will arrive as a regular data frame in `process()`.
- **Mitigation:** Handle SOS_FRAME detection inside `VideoDecoder::process()` by checking `frame->getMetadata()->getFrameType() == FrameMetadata::SOS_FRAME`. No Module base changes needed.

---

## Files Changed

| Action | File |
|--------|------|
| **Phase 0** | |
| Merge | All H265 infrastructure from `bug_fix/h265-mp4reader-support` |
| **Phase 1** | |
| Modify | `base/include/FrameMetadata.h` — add `SOS_FRAME` to enum |
| Create | `base/include/SosFrameMetadata.h` — SOS payload metadata class |
| Modify | `base/include/Frame.h` — add `CODEC_SWITCH_EOS` to `EoSFrame::EoSFrameType` |
| **Phase 2** | |
| Create | `base/include/VideoDecoder.h` — unified decoder header |
| Create | `base/src/VideoDecoder.cpp` — unified decoder implementation |
| Modify | `base/CMakeLists.txt` — add VideoDecoder source/header/test |
| **Phase 3** | |
| Create | `base/test/videodecoder_tests.cpp` — test suite |
