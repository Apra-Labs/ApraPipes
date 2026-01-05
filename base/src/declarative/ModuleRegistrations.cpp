// ============================================================
// File: declarative/ModuleRegistrations.cpp
// Task D2: Property Binding System
//
// Central registration file for all built-in modules.
// Uses fluent builder pattern for readable registrations.
// ============================================================

#include "declarative/ModuleRegistrations.h"
#include "declarative/ModuleRegistrationBuilder.h"
#include "declarative/ModuleRegistry.h"
#include <iostream>  // Debug

// Include module headers (always available)
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "FaceDetectorXform.h"
#include "QRReader.h"
#include "ValveModule.h"
#include "StatSink.h"
#include "TestSignalGeneratorSrc.h"

// D2 Phase 5: Additional priority modules
#include "Split.h"
#include "Merge.h"
#include "ImageDecoderCV.h"
#include "ImageEncoderCV.h"
#include "Mp4ReaderSource.h"
#include "Mp4WriterSink.h"
#include "ImageResizeCV.h"
#include "RotateCV.h"
#include "ColorConversionXForm.h"
#include "VirtualPTZ.h"
#include "TextOverlayXForm.h"
#include "BrightnessContrastControlXform.h"
#include "CalcHistogramCV.h"

// Batch 1: Additional Source modules
#include "WebCamSource.h"
#include "RTSPClientSrc.h"
#include "ExternalSourceModule.h"

// Batch 2: Additional Transform modules
#include "OverlayModule.h"
#include "HistogramOverlay.h"
#include "BMPConverter.h"
#include "MotionVectorExtractor.h"
#include "AffineTransform.h"
#include "MultimediaQueueXform.h"

// Batch 3: Additional Sink modules
#include "ExternalSinkModule.h"
// Note: RTSPPusher and VirtualCameraSink require mandatory constructor args
// and need applyProperties support - skipped for now

// Batch 5: Utility modules
#include "FramesMuxer.h"
// Note: H264FrameDemuxer is not derived from Module - cannot be registered

// Conditionally include CUDA modules
#ifdef ENABLE_CUDA
#include "H264Decoder.h"
#endif

#include <mutex>

namespace apra {

void ensureBuiltinModulesRegistered() {
    auto& registry = ModuleRegistry::instance();

    // If registry is empty (e.g., after clear() in tests), re-run all
    // registered callbacks. This allows modules registered via REGISTER_MODULE
    // to be re-registered with their full property metadata.
    if (!registry.hasModule("FileReaderModule")) {
        registry.rerunRegistrations();
    }

    // Register all built-in modules using central registration pattern.
    // Each module registration checks if it's already registered to avoid duplicates.
    // This allows modules to be registered here even if some modules were already
    // registered via REGISTER_MODULE macro.
    {
        // FileReaderModule - reads frames from files
        // NOTE: Output type depends on what user specifies - this is a generic file reader
        // The outputFrameType property allows users to specify the frame type in TOML
        if (!registry.hasModule("FileReaderModule")) {
            registerModule<FileReaderModule, FileReaderModuleProps>()
                .category(ModuleCategory::Source)
                .description("Reads frames from files matching a pattern. Supports image sequences and raw frame files.")
                .tags("source", "file", "reader")
                .output("output", "Frame")  // Generic - actual type set via outputFrameType prop
                .stringProp("strFullFileNameWithPattern", "File path pattern (e.g., /path/frame_????.raw)", true)
                .intProp("startIndex", "Starting file index", false, 0, 0)
                .intProp("maxIndex", "Maximum file index (-1 for unlimited)", false, -1, -1)
                .boolProp("readLoop", "Loop back to start when reaching end", false, true)
                .enumProp("outputFrameType", "Output frame type", false, "Frame",
                    "Frame", "EncodedImage", "RawImage", "RawImagePlanar");
        }

        // FileWriterModule - writes frames to files
        if (!registry.hasModule("FileWriterModule")) {
            registerModule<FileWriterModule, FileWriterModuleProps>()
                .category(ModuleCategory::Sink)
                .description("Writes frames to files. Supports file sequences with pattern-based naming.")
                .tags("sink", "file", "writer")
                .input("input", "Frame")
                .stringProp("strFullFileNameWithPattern", "Output file path pattern (e.g., /path/frame_????.raw)", true)
                .boolProp("append", "Append to existing files instead of overwriting", false, false);
        }

        // FaceDetectorXform - face detection
        // NOTE: Validates for RAW_IMAGE only (not RawImagePlanar despite what name suggests)
        if (!registry.hasModule("FaceDetectorXform")) {
            registerModule<FaceDetectorXform, FaceDetectorXformProps>()
                .category(ModuleCategory::Analytics)
                .description("Detects faces in image frames using deep learning models.")
                .tags("analytics", "face", "detection", "transform")
                .input("input", "RawImage")
                .output("output", "Frame")
                .selfManagedOutputPins();  // Creates output pin in addInputPin()
        }

        // QRReader - QR code and barcode detection
        // NOTE: Accepts both RAW_IMAGE and RAW_IMAGE_PLANAR
        if (!registry.hasModule("QRReader")) {
            registerModule<QRReader, QRReaderProps>()
                .category(ModuleCategory::Analytics)
                .description("Reads and decodes QR codes and barcodes from image frames.")
                .tags("analytics", "qr", "barcode", "reader")
                .input("input", "RawImage", "RawImagePlanar")
                .output("output", "Frame")
                .selfManagedOutputPins();  // Creates output pin in addInputPin()
        }

        // TestSignalGenerator - generates test frames
        if (!registry.hasModule("TestSignalGenerator")) {
            registerModule<TestSignalGenerator, TestSignalGeneratorProps>()
                .category(ModuleCategory::Source)
                .description("Generates test signal frames for testing pipelines")
                .tags("source", "test", "generator", "signal")
                .output("output", "RawImagePlanar")
                .intProp("width", "Frame width in pixels", true, 0, 1, 4096)
                .intProp("height", "Frame height in pixels", true, 0, 1, 4096);
        }

        // StatSink - might not have REGISTER_MODULE yet
        if (!registry.hasModule("StatSink")) {
            registerModule<StatSink, StatSinkProps>()
                .category(ModuleCategory::Sink)
                .description("Statistics sink for measuring pipeline performance")
                .tags("sink", "stats", "performance", "debug")
                .input("input", "Frame");
        }

        // ValveModule - might not have REGISTER_MODULE yet
        if (!registry.hasModule("ValveModule")) {
            registerModule<ValveModule, ValveModuleProps>()
                .category(ModuleCategory::Utility)
                .description("Controls frame flow by allowing a specified number of frames to pass")
                .tags("utility", "valve", "control", "flow")
                .input("input", "Frame")
                .output("output", "Frame");
        }

        // ============================================================
        // D2 Phase 5: Additional priority modules
        // ============================================================

        // Split - splits frames across multiple outputs
        if (!registry.hasModule("Split")) {
            registerModule<Split, SplitProps>()
                .category(ModuleCategory::Utility)
                .description("Splits input frames across multiple output pins")
                .tags("utility", "split", "routing")
                .input("input", "Frame")
                .output("output_1", "Frame")
                .output("output_2", "Frame")
                .selfManagedOutputPins();  // Creates output pins in addInputPin()
        }

        // Merge - merges frames from multiple inputs
        if (!registry.hasModule("Merge")) {
            registerModule<Merge, MergeProps>()
                .category(ModuleCategory::Utility)
                .description("Merges frames from multiple input pins")
                .tags("utility", "merge", "sync")
                .input("input_1", "Frame")
                .input("input_2", "Frame")
                .output("output", "Frame")
                .selfManagedOutputPins();  // Creates output pin dynamically
        }

        // ImageDecoderCV - decodes encoded images to raw
        if (!registry.hasModule("ImageDecoderCV")) {
            registerModule<ImageDecoderCV, ImageDecoderCVProps>()
                .category(ModuleCategory::Transform)
                .description("Decodes encoded images (JPEG, PNG, BMP) to raw image format using OpenCV")
                .tags("decoder", "image", "opencv")
                .input("input", "EncodedImage")
                .output("output", "RawImage");
        }

        // ImageEncoderCV - encodes raw images
        if (!registry.hasModule("ImageEncoderCV")) {
            registerModule<ImageEncoderCV, ImageEncoderCVProps>()
                .category(ModuleCategory::Transform)
                .description("Encodes raw images to JPEG/PNG format using OpenCV")
                .tags("encoder", "image", "opencv")
                .input("input", "RawImage")
                .output("output", "EncodedImage")
                .selfManagedOutputPins();  // Creates output pin in addInputPin()
        }

        // Mp4ReaderSource - reads MP4 video files
        // Note: Output pin type is determined at runtime based on actual MP4 content
        // Requires valid video file path to run - validation will fail without real file
        if (!registry.hasModule("Mp4ReaderSource")) {
            registerModule<Mp4ReaderSource, Mp4ReaderSourceProps>()
                .category(ModuleCategory::Source)
                .description("Reads video frames from MP4 files")
                .tags("source", "mp4", "video", "file")
                .output("output", "H264Data", "EncodedImage")
                .stringProp("videoPath", "Path to MP4 video file", true)
                .boolProp("parseFS", "Parse filesystem for metadata", false, true)
                .boolProp("direction", "Playback direction (true=forward)", false, true)
                .boolProp("bFramesEnabled", "Enable B-frame decoding", false, false)
                .intProp("reInitInterval", "Re-initialization interval in frames", false, 0, 0)
                .intProp("parseFSTimeoutDuration", "Filesystem parse timeout (ms)", false, 0, 0)
                .boolProp("readLoop", "Loop playback when reaching end", false, false)
                .boolProp("giveLiveTS", "Use live timestamps", false, false);
        }

        // Mp4WriterSink - writes MP4 video files
        if (!registry.hasModule("Mp4WriterSink")) {
            registerModule<Mp4WriterSink, Mp4WriterSinkProps>()
                .category(ModuleCategory::Sink)
                .description("Writes video frames to MP4 files")
                .tags("sink", "mp4", "video", "file")
                .input("input", "H264Data", "EncodedImage")
                .stringProp("baseFolder", "Output folder for MP4 files", false, "")
                .intProp("chunkTime", "Chunk duration in seconds", false, 60, 1)
                .intProp("syncTimeInSecs", "Sync interval in seconds", false, 1, 1)
                .intProp("fps", "Output frame rate", false, 30, 1, 120)
                .boolProp("recordedTSBasedDTS", "Use recorded timestamps for DTS", false, false)
                .boolProp("enableMetadata", "Enable metadata track", false, false);
        }

        // ImageResizeCV - resizes images using OpenCV
        if (!registry.hasModule("ImageResizeCV")) {
            registerModule<ImageResizeCV, ImageResizeCVProps>()
                .category(ModuleCategory::Transform)
                .description("Resizes images to specified dimensions using OpenCV")
                .tags("transform", "resize", "image", "opencv")
                .input("input", "RawImage")
                .output("output", "RawImage")
                .intProp("width", "Target width in pixels", true, 0, 1, 8192)
                .intProp("height", "Target height in pixels", true, 0, 1, 8192)
                .selfManagedOutputPins();  // Creates output pin in addInputPin()
        }

        // RotateCV - rotates images using OpenCV
        // NOTE: Accepts both RAW_IMAGE and RAW_IMAGE_PLANAR
        if (!registry.hasModule("RotateCV")) {
            registerModule<RotateCV, RotateCVProps>()
                .category(ModuleCategory::Transform)
                .description("Rotates images by a specified angle using OpenCV")
                .tags("transform", "rotate", "image", "opencv")
                .input("input", "RawImage", "RawImagePlanar")
                .output("output", "RawImage")
                .floatProp("angle", "Rotation angle in degrees", true, 0.0, -360.0, 360.0)
                .selfManagedOutputPins();  // Creates output pin in addInputPin()
        }

        // ColorConversion - converts between color spaces
        // NOTE: Accepts both RAW_IMAGE and RAW_IMAGE_PLANAR, can convert between them
        // Key conversions: YUV420PLANAR_TO_RGB converts RawImagePlanar to RawImage
        if (!registry.hasModule("ColorConversion")) {
            registerModule<ColorConversion, ColorConversionProps>()
                .category(ModuleCategory::Transform)
                .description("Converts images between different color spaces (RGB, BGR, YUV, Mono, Bayer). Use YUV420PLANAR_TO_RGB to convert planar to packed format.")
                .tags("transform", "color", "conversion", "opencv", "planar")
                .input("input", "RawImage", "RawImagePlanar")
                .output("output", "RawImage", "RawImagePlanar")
                .enumProp("conversionType", "Color space conversion to perform", true, "RGB_TO_MONO",
                    "RGB_TO_MONO", "BGR_TO_MONO", "BGR_TO_RGB", "RGB_TO_BGR",
                    "RGB_TO_YUV420PLANAR", "YUV420PLANAR_TO_RGB",
                    "BAYERBG8_TO_MONO", "BAYERBG8_TO_RGB", "BAYERGB8_TO_RGB",
                    "BAYERRG8_TO_RGB", "BAYERGR8_TO_RGB")
                .selfManagedOutputPins();  // Creates output pin in addInputPin()
        }

        // VirtualPTZ - virtual pan/tilt/zoom
        if (!registry.hasModule("VirtualPTZ")) {
            registerModule<VirtualPTZ, VirtualPTZProps>()
                .category(ModuleCategory::Transform)
                .description("Virtual pan/tilt/zoom by extracting a region of interest from the input image")
                .tags("transform", "ptz", "crop", "roi")
                .input("input", "RawImage")
                .output("output", "RawImage")
                .dynamicProp("roiX", "float", "X coordinate of ROI (0-1 normalized)", false, "0")
                .dynamicProp("roiY", "float", "Y coordinate of ROI (0-1 normalized)", false, "0")
                .dynamicProp("roiWidth", "float", "Width of ROI (0-1 normalized)", false, "1")
                .dynamicProp("roiHeight", "float", "Height of ROI (0-1 normalized)", false, "1")
                .selfManagedOutputPins();  // Creates output pin in addInputPin()
        }

        // TextOverlayXForm - text overlay on images
        if (!registry.hasModule("TextOverlayXForm")) {
            registerModule<TextOverlayXForm, TextOverlayXFormProps>()
                .category(ModuleCategory::Transform)
                .description("Overlays text on images with customizable font, color, and position")
                .tags("transform", "overlay", "text", "annotation")
                .input("input", "RawImage")
                .output("output", "RawImage")
                .dynamicProp("text", "string", "Text to overlay on the image", false, "")
                .dynamicProp("position", "string", "Position of text (top-left, top-right, bottom-left, bottom-right)", false, "top-left")
                .dynamicProp("fontSize", "int", "Font size in pixels", false, "24")
                .dynamicProp("fontColor", "string", "Font color (e.g., white, black, red)", false, "white")
                .dynamicProp("backgroundColor", "string", "Background color (transparent for none)", false, "transparent")
                .dynamicProp("alpha", "float", "Opacity (0-1)", false, "1.0")
                .boolProp("isDateTime", "Display current date/time instead of text", false, false)
                .selfManagedOutputPins();  // Creates output pin in addInputPin()
        }

        // BrightnessContrastControl - adjusts image brightness and contrast
        if (!registry.hasModule("BrightnessContrastControl")) {
            registerModule<BrightnessContrastControl, BrightnessContrastControlProps>()
                .category(ModuleCategory::Transform)
                .description("Adjusts image brightness and contrast in real-time")
                .tags("transform", "brightness", "contrast", "adjustment")
                .input("input", "RawImage")
                .output("output", "RawImage")
                .dynamicProp("contrast", "float", "Contrast multiplier (1.0 = no change)", false, "1.0")
                .dynamicProp("brightness", "float", "Brightness offset (0.0 = no change)", false, "0.0")
                .selfManagedOutputPins();  // Creates output pin in addInputPin()
        }

        // CalcHistogramCV - calculates image histogram
        if (!registry.hasModule("CalcHistogramCV")) {
            registerModule<CalcHistogramCV, CalcHistogramCVProps>()
                .category(ModuleCategory::Analytics)
                .description("Calculates histogram of image intensity values using OpenCV")
                .tags("analytics", "histogram", "opencv", "statistics")
                .input("input", "RawImage")
                .output("output", "Array")
                .dynamicProp("bins", "int", "Number of histogram bins", false, "8")
                .dynamicProp("maskImgPath", "string", "Path to mask image (optional)", false, "")
                .selfManagedOutputPins();  // Creates output pin dynamically
        }

        // ============================================================
        // Batch 1: Additional Source Modules
        // ============================================================

        // WebCamSource - captures from webcam/camera
        if (!registry.hasModule("WebCamSource")) {
            registerModule<WebCamSource, WebCamSourceProps>()
                .category(ModuleCategory::Source)
                .description("Captures video frames from webcam or USB camera using OpenCV")
                .tags("source", "camera", "webcam", "capture", "opencv")
                .output("output", "RawImage")
                .intProp("cameraId", "Camera device ID (-1 for default)", false, -1, -1, 10)
                .intProp("width", "Capture width in pixels", false, 640, 1, 4096)
                .intProp("height", "Capture height in pixels", false, 480, 1, 4096)
                .intProp("fps", "Target frames per second", false, 30, 1, 120);
        }

        // RTSPClientSrc - receives video from RTSP stream
        if (!registry.hasModule("RTSPClientSrc")) {
            registerModule<RTSPClientSrc, RTSPClientSrcProps>()
                .category(ModuleCategory::Source)
                .description("Receives video from RTSP stream (IP cameras, media servers)")
                .tags("source", "rtsp", "network", "stream", "camera")
                .output("output", "H264Data", "EncodedImage")
                .stringProp("rtspURL", "RTSP stream URL (e.g., rtsp://host:port/path)", true)
                .stringProp("userName", "Authentication username", false, "")
                .stringProp("password", "Authentication password", false, "")
                .boolProp("useTCP", "Use TCP transport instead of UDP", false, true);
        }

        // ExternalSourceModule - allows external frame injection
        if (!registry.hasModule("ExternalSourceModule")) {
            registerModule<ExternalSourceModule, ExternalSourceModuleProps>()
                .category(ModuleCategory::Source)
                .description("Source module for external frame injection. Used for programmatic frame feeding.")
                .tags("source", "external", "api", "programmatic")
                .output("output", "Frame");
        }

        // ============================================================
        // Batch 2: Additional Transform Modules
        // ============================================================

        // Note: AffineTransform requires TransformType + angle in constructor - needs applyProperties
        // Note: MultimediaQueueXform has ambiguous constructors - needs applyProperties

        // OverlayModule - overlays graphics on frames
        if (!registry.hasModule("OverlayModule")) {
            registerModule<OverlayModule, OverlayModuleProps>()
                .category(ModuleCategory::Transform)
                .description("Overlays graphics, shapes, and annotations on image frames")
                .tags("transform", "overlay", "graphics", "annotation")
                .input("input", "RawImage")
                .output("output", "RawImage")
                .selfManagedOutputPins();
        }

        // HistogramOverlay - overlays histogram visualization
        if (!registry.hasModule("HistogramOverlay")) {
            registerModule<HistogramOverlay, HistogramOverlayProps>()
                .category(ModuleCategory::Transform)
                .description("Overlays histogram visualization on image frames")
                .tags("transform", "histogram", "overlay", "visualization")
                .input("input", "RawImage")
                .output("output", "RawImage")
                .selfManagedOutputPins();
        }

        // BMPConverter - converts raw images to BMP format
        if (!registry.hasModule("BMPConverter")) {
            registerModule<BMPConverter, BMPConverterProps>()
                .category(ModuleCategory::Transform)
                .description("Converts raw images to BMP (bitmap) format. Useful for saving images or viewing in standard image viewers.")
                .tags("transform", "image", "bmp", "converter", "encoder")
                .input("input", "RawImage")
                .output("output", "EncodedImage")
                .selfManagedOutputPins();
        }

        // MotionVectorExtractor - extracts motion vectors from H264 video
        if (!registry.hasModule("MotionVectorExtractor")) {
            registerModule<MotionVectorExtractor, MotionVectorExtractorProps>()
                .category(ModuleCategory::Analytics)
                .description("Extracts motion vectors from H.264 encoded video frames. Useful for motion analysis and activity detection.")
                .tags("analytics", "motion", "vector", "h264", "video")
                .input("input", "H264Data")
                .output("output", "MotionVectorData")
                .enumProp("MVExtractMethod", "Extraction method to use", false, "FFMPEG", "FFMPEG", "OPENH264")
                .boolProp("sendDecodedFrame", "Also output decoded raw frames", false, false)
                .intProp("motionVectorThreshold", "Minimum motion vector magnitude to report", false, 2, 0, 100)
                .selfManagedOutputPins();
        }

        // AffineTransform - applies affine transformations (rotation, scale, translate)
        // Note: Creates output pin in addInputPin(), so selfManagedOutputPins is required
        if (!registry.hasModule("AffineTransform")) {
            registerModule<AffineTransform, AffineTransformProps>()
                .category(ModuleCategory::Transform)
                .description("Applies affine transformations to images including rotation, scaling, and translation using OpenCV.")
                .tags("transform", "affine", "rotate", "scale", "translate", "opencv")
                .input("input", "RawImage")  // Only accepts RawImage (not planar)
                .output("output", "RawImage")
                .floatProp("angle", "Rotation angle in degrees", false, 0.0, -360.0, 360.0)
                .intProp("x", "Horizontal translation in pixels", false, 0)
                .intProp("y", "Vertical translation in pixels", false, 0)
                .floatProp("scale", "Scale factor (1.0 = no scaling)", false, 1.0, 0.01, 100.0)
                .selfManagedOutputPins();
        }

        // MultimediaQueueXform - multimedia frame queue for buffering and playback
        if (!registry.hasModule("MultimediaQueueXform")) {
            registerModule<MultimediaQueueXform, MultimediaQueueXformProps>()
                .category(ModuleCategory::Utility)
                .description("Multimedia frame queue for buffering, playback control, and temporal frame access. Useful for video recording and playback systems.")
                .tags("utility", "queue", "buffer", "playback", "multimedia")
                .input("input", "Frame")
                .output("output", "Frame")
                .intProp("queueLength", "Queue capacity (frames or milliseconds)", false, 10000, 1, 1000000)
                .intProp("tolerance", "Additional buffer when downstream is full", false, 5000, 0, 100000)
                .intProp("mmqFps", "Target FPS for queue timing calculations", false, 24, 1, 120)
                .boolProp("isMapDelayInTime", "Interpret queue length as time (true) or frames (false)", false, true)
                .selfManagedOutputPins();
        }

        // ============================================================
        // Batch 3: Additional Sink Modules
        // ============================================================

        // ExternalSinkModule - allows external frame consumption
        if (!registry.hasModule("ExternalSinkModule")) {
            registerModule<ExternalSinkModule, ExternalSinkModuleProps>()
                .category(ModuleCategory::Sink)
                .description("Sink module for external frame consumption. Used for programmatic frame retrieval.")
                .tags("sink", "external", "api", "programmatic")
                .input("input", "Frame");
        }

        // Note: RTSPPusher and VirtualCameraSink require mandatory constructor args
        // and need custom applyProperties support - will be added after builder enhancement

        // ============================================================
        // Batch 5: Additional Utility Modules
        // ============================================================

        // FramesMuxer - synchronizes multiple frame streams
        if (!registry.hasModule("FramesMuxer")) {
            registerModule<FramesMuxer, FramesMuxerProps>()
                .category(ModuleCategory::Utility)
                .description("Synchronizes and multiplexes multiple input frame streams based on timestamp or frame index")
                .tags("utility", "mux", "sync", "multiplex")
                .input("input_1", "Frame")
                .input("input_2", "Frame")
                .output("output", "Frame")
                .intProp("maxDelay", "Maximum frame delay before dropping", false, 30, 1, 1000)
                .floatProp("maxTsDelayInMS", "Maximum timestamp delay in milliseconds", false, 16.67, 0.0, 1000.0)
                .enumProp("strategy", "Muxing strategy", false, "ALL_OR_NONE",
                    "ALL_OR_NONE", "MAX_DELAY_ANY", "MAX_TIMESTAMP_DELAY")
                .selfManagedOutputPins();
        }

        // Note: H264FrameDemuxer is not derived from Module, cannot be registered

        // ============================================================
        // CUDA-only modules
        // ============================================================
#ifdef ENABLE_CUDA
        // H264Decoder - decodes H.264 video
        if (!registry.hasModule("H264Decoder")) {
            registerModule<H264Decoder, H264DecoderProps>()
                .category(ModuleCategory::Transform)
                .description("Decodes H.264/AVC encoded video frames to raw image frames.")
                .tags("decoder", "h264", "video", "transform")
                .input("input", "H264Frame")
                .output("output", "RawImagePlanar");
        }
#endif
    }
}

} // namespace apra
