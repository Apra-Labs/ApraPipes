# H265 Mp4Reader Support — Requirements

## Problem

`Mp4ReaderSource` in `ApraPipes_SNAP` only handles H264 and JPEG streams. All three H265 test cases fail immediately at pipeline `init()` with:

```
Mp4ReaderSource::validateOutputPins — input frameType is expected to be MP4_VIDEO_METADATA or ENCODED_IMAGE. Actual<20>
```

`FrameType 20 = HEVC_DATA`. The pipeline dies before any decoding happens.

### Root causes (both must be fixed)

1. **`validateOutputPins()` whitelist** (`Mp4ReaderSource.cpp` ~line 1652) — `HEVC_DATA` not listed as an accepted output frame type.
2. **No `Mp4ReaderDetailH265` class** — only `Mp4ReaderDetailH264` and `Mp4ReaderDetailJpeg` exist. There is no implementation path for HEVC tracks.

## Requirements

### R1 — H265 decode support
Implement `Mp4ReaderDetailH265` class in `Mp4ReaderSource.cpp`, mirroring `Mp4ReaderDetailH264` with:
- VPS+SPS+PPS extraction from libmp4's `mp4_video_decoder_config.hevc.{vps,sps,pps}` fields
- `prependVpsSpsPps()` prepending 3 NAL units (not 2 as in H264)
- `H265Utils::isIDR()` / `H265Utils::getNALUType()` for frame classification
- `H265Metadata(width, height)` output metadata
- Uses `h265ImagePinId` pin

### R2 — Codec auto-detection
`Mp4ReaderSource` must automatically detect whether a track is H264 or H265 by querying the libmp4 API at `init()` time. The caller should NOT need to manually specify codec type — the correct detail class is selected automatically based on what the MP4 container reports.

- Add an `"auto"` output format option (and make it the default) in the declarative auto-pin block
- Existing explicit `"h264"` and `"jpeg"` paths must still work

### R3 — No H264 regressions
All existing H264 tests must continue to pass. H264 code paths are unchanged.

### R4 — Tests pass
All 3 H265 test cases in `h265decoder_tests` must pass:
- `mp4reader_h265decoder_eglrenderer`
- `mp4reader_h265decoder_extsink`
- `mp4reader_h265decoder_statsink`

If test video files are git-LFS pointer stubs, generate real H265 clips with ffmpeg and update test data paths.

### R5 — validateOutputPins and addOutPutPin updated
- `validateOutputPins()` must accept `HEVC_DATA`
- `addOutPutPin()` must handle `HEVC_DATA` and assign `h265ImagePinId`

## Files to modify

| File | Changes |
|------|---------|
| `base/src/Mp4ReaderSource.cpp` | Add includes, `Mp4ReaderDetailH265` class, `HEVC_DATA` branches in `init()`/`addOutPutPin()`/`validateOutputPins()`, auto-detect logic, declarative `"h265"`/`"auto"` branch |
| `base/include/Mp4ReaderSource.h` | Add `h265ImagePinId` member |

## Reference

- H265Metadata: `/home/developer/ws_yash/ApraPipes_SNAP/base/include/H265Metadata.h`
- H265Utils: `/home/developer/ws_yash/ApraPipes_SNAP/base/include/H265Utils.h`
- libmp4 HEVC struct: `mp4_video_decoder_config.hevc.{vps,vps_size,sps,sps_size,pps,pps_size}`
- Same libmp4 seek API works for HEVC — no new libmp4 calls needed for seek/GoP
- apranvr_clean at `/home/developer/ws_yash/apranvr_clean` — does NOT have H265 support; use as structural reference only

## Constraints

- Do not break existing H264 or JPEG paths
- Do not modify CMakeLists unless required for new headers (H265Metadata.h and H265Utils.h are already built in the prior sprint)
- GTK3 dev symlinks are missing on the build machine — create them before building (known workaround from prior session)
