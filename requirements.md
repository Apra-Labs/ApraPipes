# Requirements — VideoDecoder Unified Module

## Base Branch
`NVR_Snapshot_JP6` — branch to fork from and merge back to

## Goal
Build a single `VideoDecoder` module that auto-detects and hardware-decodes both H264 and H265 streams, and handles a live mid-stream codec switch (e.g., camera reconfigured from H264 → H265) without restarting the pipeline.

## Background
**Repo:** `/home/developer/ws_yash/ApraPipes_SNAP`  
**Existing modules:**
- `H264Decoder` — decodes H264_DATA frames via H264DecoderV4L2Helper → /dev/nvhost-nvdec (NvMMLite BlockType=261)
- `H265Decoder` — decodes HEVC_DATA frames via same helper with V4L2_PIX_FMT_H265 FourCC
- `Mp4ReaderSource` — detects HEVC vs H264 via vdc->codec at init time, emits H264_DATA or HEVC_DATA frames
- `H264DecoderV4L2Helper` — V4L2 kernel interface, manages JP5/JP6 USERPTR/MMAP split, NvMMLite

**The gap:** The user must know in advance which codec a source uses and choose H264Decoder or H265Decoder at pipeline construction time. When a live camera changes codec (e.g., H264 → H265), the pipeline must be rebuilt. This is disruptive and impractical.

## Signal Protocol (EOS/SOS)

**Proposed design:**  
Sources signal codec changes via a two-frame protocol:
1. **EOS_FRAME** — "end of current codec stream, decoder must flush/drain"
2. **SOS_FRAME** — "new codec stream starting" — carries codec type + header NALUs (SPS/PPS for H264; VPS+SPS+PPS for H265)

This decouples the source codec from the decoder selection. `VideoDecoder` becomes a state machine:

```
UNINIT
  │ first H264_DATA or HEVC_DATA frame
  ▼
DECODING (H264 or H265 backend active)
  │ EOS_FRAME received
  ▼
FLUSHING (drain current backend, propagate EOS downstream)
  │ SOS_FRAME received  
  ▼
REINIT (init new backend from SOS headers)
  │ complete
  ▼
DECODING (new codec active)
```

## Scope

1. **`SOS_FRAME` frame type** — add to `FrameType` enum. Payload carries: codec enum (H264/H265) + header NALUs blob (SPS/PPS or VPS+SPS+PPS).
2. **`Mp4ReaderSource` SOS/EOS emission** — when iterating multi-codec MP4 tracks or re-opened with a different codec file, emit `EOS_FRAME` then `SOS_FRAME` before the first frame of the new codec.
3. **`VideoDecoder` module (new)** — `base/include/VideoDecoder.h` + `base/src/VideoDecoder.cpp`:
   - Input: accepts `H264_DATA`, `HEVC_DATA`, `EOS_FRAME`, `SOS_FRAME`
   - On `H264_DATA`/`HEVC_DATA` (first frame): init appropriate backend (auto-detect from frame type)
   - On `EOS_FRAME`: flush backend, propagate EOS, set state FLUSHING
   - On `SOS_FRAME`: init new backend from header NALUs in payload, set state DECODING
   - Output: always `NV12`/raw YUV frames — downstream never changes type
4. **CMakeLists.txt** — add VideoDecoder to build
5. **Tests** — `base/test/videodecoder_tests.cpp`:
   - `video_decoder_h264_basic` — H264 MP4 → VideoDecoder → StatSink, confirm frames decoded
   - `video_decoder_h265_basic` — H265 MP4 → VideoDecoder → StatSink, confirm frames decoded  
   - `video_decoder_codec_switch` — programmatic EOS/SOS injection mid-stream: pipeline runs H264, receives EOS+SOS(H265), continues decoding H265 frames, confirm no frame loss at switch boundary

## Out of Scope
- RTSPSource codec-switch signaling — future work (RTSPSource would emit same EOS/SOS when SPS/PPS changes; identical VideoDecoder handles it)
- Simultaneous multi-codec decode (not needed — one active backend at a time)
- Software fallback decode — always use NVDEC hardware path

## Constraints
- Must not regress existing H264Decoder or H265Decoder tests
- Hardware decode only — NVDEC via H264DecoderV4L2Helper
- JP5 and JP6 compatible (USERPTR vs MMAP split already handled in helper)
- Boost.Test (not Google Test) — run with `--run_test=`, not `--gtest_filter=`
- Tests must use absolute paths (not CWD-relative) for data files

## Acceptance Criteria
- [ ] `VideoDecoder` module compiles in ARM64 build without errors
- [ ] `video_decoder_h264_basic` passes — `*** No errors detected`
- [ ] `video_decoder_h265_basic` passes — `*** No errors detected`
- [ ] `video_decoder_codec_switch` passes — frames decoded before and after switch, no crash
- [ ] Existing `h264decoder_tests` still pass (no regression)
- [ ] Existing `h265decoder_tests` still pass (no regression)
- [ ] NVDEC hardware path confirmed active (NvMMLite block opens, no software fallback)
