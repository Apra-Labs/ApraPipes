// ============================================================
// File: declarative/FrameTypeRegistrations.cpp
// Task F1-F4: Frame Type Metadata
//
// Central registration file for all frame types.
// Defines the frame type hierarchy for connection validation.
// ============================================================

#include "declarative/FrameTypeRegistry.h"

namespace apra {

namespace {

// Helper to register a frame type
void registerType(const std::string& name, const std::string& parent,
                  const std::string& description,
                  const std::vector<std::string>& tags = {}) {
    FrameTypeInfo info;
    info.name = name;
    info.parent = parent;
    info.description = description;
    info.tags = tags;
    FrameTypeRegistry::instance().registerFrameType(std::move(info));
}

// Register all built-in frame types
bool registerBuiltinFrameTypes() {
    // ================================================================
    // Root types
    // ================================================================
    registerType("Frame", "", "Base frame type - all frames derive from this",
                 {"base"});

    // ================================================================
    // F1: Image frame types
    // ================================================================

    // Raw images (uncompressed pixel data)
    registerType("RawImage", "Frame",
                 "Uncompressed image with pixel data (RGB, BGR, YUV, etc.)",
                 {"image", "raw", "uncompressed"});

    registerType("RawImagePlanar", "RawImage",
                 "Planar format raw image (separate Y, U, V planes)",
                 {"image", "raw", "planar"});

    // Encoded images (compressed)
    registerType("EncodedImage", "Frame",
                 "Compressed/encoded image data",
                 {"image", "encoded", "compressed"});

    registerType("BMPImage", "EncodedImage",
                 "BMP format image",
                 {"image", "encoded", "bmp"});

    // ================================================================
    // F2: Video codec frame types
    // ================================================================

    registerType("H264Data", "EncodedImage",
                 "H.264/AVC encoded video frame",
                 {"video", "encoded", "h264", "avc"});

    registerType("HEVCData", "EncodedImage",
                 "H.265/HEVC encoded video frame",
                 {"video", "encoded", "h265", "hevc"});

    registerType("MotionVectorData", "Frame",
                 "Motion vector data extracted from video codec",
                 {"video", "motion", "analysis"});

    // ================================================================
    // F3: Audio and metadata frame types
    // ================================================================

    registerType("Audio", "Frame",
                 "Audio sample data",
                 {"audio"});

    registerType("Array", "Frame",
                 "Generic array data",
                 {"data", "array"});

    registerType("MP4VideoMetadata", "Frame",
                 "MP4 container video metadata",
                 {"metadata", "mp4", "video"});

    registerType("Text", "Frame",
                 "Text data (e.g., transcriptions, captions)",
                 {"text", "string"});

    // ================================================================
    // F4: Control and analytics frame types
    // ================================================================

    // Control frames (internal signaling)
    registerType("ControlFrame", "Frame",
                 "Internal control/signaling frame",
                 {"control", "internal"});

    registerType("ChangeDetection", "ControlFrame",
                 "Change detection signal",
                 {"control", "detection"});

    registerType("PropsChange", "ControlFrame",
                 "Property change notification",
                 {"control", "props"});

    registerType("PausePlay", "ControlFrame",
                 "Pause/play control signal",
                 {"control", "playback"});

    registerType("Command", "ControlFrame",
                 "Generic command frame",
                 {"control", "command"});

    registerType("GPIO", "ControlFrame",
                 "GPIO signal frame",
                 {"control", "gpio", "hardware"});

    // Analytics/detection results
    registerType("AnalyticsFrame", "Frame",
                 "Analytics/detection result data",
                 {"analytics", "detection"});

    registerType("FaceDetectsInfo", "AnalyticsFrame",
                 "Face detection results (bounding boxes)",
                 {"analytics", "face", "detection"});

    registerType("FaceLandmarksInfo", "AnalyticsFrame",
                 "Facial landmark points",
                 {"analytics", "face", "landmarks"});

    registerType("DefectsInfo", "AnalyticsFrame",
                 "Defect detection results",
                 {"analytics", "defects", "inspection"});

    registerType("EdgeDefectAnalysisInfo", "AnalyticsFrame",
                 "Edge defect analysis results",
                 {"analytics", "defects", "edge"});

    registerType("ROI", "AnalyticsFrame",
                 "Region of interest data",
                 {"analytics", "roi", "region"});

    registerType("Line", "AnalyticsFrame",
                 "Line detection result",
                 {"analytics", "line", "geometry"});

    registerType("ApraLines", "AnalyticsFrame",
                 "Collection of detected lines",
                 {"analytics", "lines", "geometry"});

    // Overlay data
    registerType("OverlayInfoImage", "Frame",
                 "Overlay information for rendering",
                 {"overlay", "render"});

    return true;
}

// Static initialization - registers frame types at startup
static bool _registered = registerBuiltinFrameTypes();

} // anonymous namespace

// ============================================================
// Public API to ensure frame types are registered
// ============================================================

void ensureBuiltinFrameTypesRegistered() {
    // The static bool _registered ensures types are registered at startup.
    // This function is a no-op but can be called to force initialization
    // in dynamic loading scenarios.
    (void)_registered;

    // Double-check by checking if Frame type exists
    if (!FrameTypeRegistry::instance().hasFrameType("Frame")) {
        registerBuiltinFrameTypes();
    }
}

} // namespace apra
