# H265 Mp4Reader Support — Design

## Architecture

### Class hierarchy (after this sprint)

```
Mp4ReaderDetailAbs  (abstract base)
├── Mp4ReaderDetailJpeg   — ENCODED_IMAGE tracks
├── Mp4ReaderDetailH264   — H264_DATA tracks (unchanged)
└── Mp4ReaderDetailH265   — HEVC_DATA tracks (NEW)
```

### Auto-detection flow

```
Mp4ReaderSource::init()
  └── query libmp4: mp4_demux_get_track_video_decoder_config()
        ├── if vdc->hevc fields populated → HEVC_DATA → new Mp4ReaderDetailH265(...)
        ├── if vdc->avc fields populated  → H264_DATA → new Mp4ReaderDetailH264(...)   [unchanged]
        └── explicit outputFormat override still respected if provided
```

### Mp4ReaderDetailH265 — key differences from H264

| Aspect | H264 | H265 |
|--------|------|------|
| Parameter sets | SPS + PPS (2 NALs) | VPS + SPS + PPS (3 NALs) |
| libmp4 fields | `vdc->avc.sps/pps` | `vdc->hevc.vps/sps/pps` |
| IDR detection | `H264Utils::isIDR()` | `H265Utils::isIDR()` |
| NAL type | `H264Utils::getNALUType()` | `H265Utils::getNALUType()` |
| Metadata | `H264Metadata(w,h)` | `H265Metadata(w,h)` |
| Output pin | `h264ImagePinId` | `h265ImagePinId` |
| Seek API | `mp4_demux_seek()` | same — `mp4_demux_seek()` |

### Declarative auto-pin block

Current:
```cpp
if (_props.outputFormat == "h264") { ... }
else if (_props.outputFormat == "jpeg") { ... }
```

After:
```cpp
if (_props.outputFormat == "h264") { ... }
else if (_props.outputFormat == "jpeg") { ... }
else if (_props.outputFormat == "h265" || _props.outputFormat == "hevc") { ... }
else if (_props.outputFormat == "auto" || _props.outputFormat.empty()) {
    // query libmp4 for track codec, auto-select
}
```

## Invariants

- H264 tests must pass before and after this change (no regression)
- The HEVC path uses the same seek, GoP, and EoS machinery as H264 — only parameter set injection and NAL parsing differ
