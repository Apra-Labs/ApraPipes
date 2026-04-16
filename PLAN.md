# PLAN.md — VideoDecoder Unified Module

## Branch
`feat/video-decoder-unified` (from `NVR_Snapshot_JP6`)
Commit: 896f44e3fa84282e5ae384bc4ddbac4a45549293

## Overview
Build a unified `VideoDecoder` module that auto-detects H264 or H265 codec from incoming frame types and delegates to the existing `H264DecoderV4L2Helper` backend with the appropriate `V4L2_PIX_FMT_*`. A new `SOS_FRAME` frame type enables mid-stream codec switching: sources emit `EOS_FRAME` (CODEC_SWITCH_EOS) then `SOS_FRAME` to signal a codec change, and `VideoDecoder` flushes the old backend and re-initializes a new one without rebuilding the pipeline.

---

## Phase 0 — H265 Prerequisites (Merge)

**Task 0.1 — Merge H265 infrastructure from `bug_fix/h265-mp4reader-support`**
- Merge full branch into `feat/video-decoder-unified`
- Brings: H265Metadata.h, H265Utils.h/.cpp, H265Decoder.h/.cpp, V4L2 helper decode_pixfmt param, Mp4ReaderSource HEVC detection, CMakeLists H265 entries, test data, test fixes
- Command: `git merge origin/bug_fix/h265-mp4reader-support --no-ff`
- Risk: Medium. Merge conflicts possible in CMakeLists.txt and V4L2 helper.

**VERIFY Checkpoint — Phase 0**
- `cmake --build . --target aprapipesut -j$(nproc)` — zero errors
- `./aprapipesut --run_test=h264decoder_tests` — passes
- `./aprapipesut --run_test=h265decoder_tests` — passes

---

## Phase 1 — SOS_FRAME Infrastructure

**Task 1.1 — Add `SOS_FRAME` to FrameType enum**
- File: `base/include/FrameMetadata.h`
- Add `SOS_FRAME` after `TEXT` (value 25)
- Risk: Low. Additive. Existing switch-case defaults will ignore it.

**Task 1.2 — Add SOS frame payload class**
- File: `base/include/SosFrameMetadata.h` (new)
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

**Task 1.3 — Add `CODEC_SWITCH_EOS` to `EoSFrame::EoSFrameType`**
- File: `base/include/Frame.h`
- Add `CODEC_SWITCH_EOS` after `MP4_SEEK_EOS`
- Risk: Low. Additive enum change.

---

## Phase 2 — VideoDecoder Module

**Task 2.1 — VideoDecoder.h** (`base/include/VideoDecoder.h`, new)
```cpp
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
    int mCurrentCodec;
};
```

**Task 2.2 — VideoDecoder.cpp** (`base/src/VideoDecoder.cpp`, new)
- `Detail` class selects `V4L2_PIX_FMT_H264` or `V4L2_PIX_FMT_H265` based on codec
- `process()` state machine:
  - First data frame → detect codec, trigger SOS → DECODING
  - `CODEC_SWITCH_EOS` → closeAllThreads() → FLUSHING
  - `SOS_FRAME` → destroy old backend, reinit with new codec → DECODING
- H264 prepends SPS+PPS; H265 prepends VPS+SPS+PPS
- Risk: Medium-high. Codec-switch teardown/reinit path needs careful V4L2 drain.

**Task 2.3 — CMakeLists.txt update**
- Add `src/VideoDecoder.cpp` to `CUDA_IP_FILES`
- Add `include/VideoDecoder.h`, `include/SosFrameMetadata.h` to `CUDA_IP_FILES_H`
- Add `test/videodecoder_tests.cpp` to ARM64 test block

---

## Phase 3 — Tests

**Task 3.1 — `base/test/videodecoder_tests.cpp`** (new)
- `video_decoder_h264_basic`: `Mp4ReaderSource(h264)` → `VideoDecoder` → `StatSink`
- `video_decoder_h265_basic`: `Mp4ReaderSource(h265)` → `VideoDecoder` → `StatSink`
- `video_decoder_codec_switch`: two sequential pipelines (H264 then H265), 5s each — verifies both codecs work through VideoDecoder

**Task 3.2 — Build and verify**
```bash
cd /home/developer/ws_yash/ApraPipes_SNAP/_build
cmake --build . --target aprapipesut -j$(nproc) 2>&1 | tail -50
cd /home/developer/ws_yash/ApraPipes_SNAP
_build/aprapipesut --run_test=videodecoder_tests 2>&1 | tail -30
```

### VERIFY Checkpoint — Phase 3
1. Build: zero errors
2. `_build/aprapipesut --run_test=videodecoder_tests` — `*** No errors detected`
3. `_build/aprapipesut --run_test=h264decoder_tests` — no regression
4. `_build/aprapipesut --run_test=h265decoder_tests` — no regression
5. NVDEC hardware path confirmed in logs

---

## Risks
1. **H265 infrastructure dependency (BLOCKING):** Phase 0 merge is a hard prerequisite.
2. **V4L2 device teardown/reinit race:** NVDEC may not release cleanly. Mitigation: explicit thread join + device close verification.
3. **SOS_FRAME routing:** detect inside `process()` via `frame->getMetadata()->getFrameType() == SOS_FRAME`. No Module base changes needed.

---

## Files Changed
| Action | File |
|--------|------|
| Merge | All H265 infrastructure from `bug_fix/h265-mp4reader-support` |
| Modify | `base/include/FrameMetadata.h` — add `SOS_FRAME` |
| Create | `base/include/SosFrameMetadata.h` |
| Modify | `base/include/Frame.h` — add `CODEC_SWITCH_EOS` |
| Create | `base/include/VideoDecoder.h` |
| Create | `base/src/VideoDecoder.cpp` |
| Modify | `base/CMakeLists.txt` |
| Create | `base/test/videodecoder_tests.cpp` |
