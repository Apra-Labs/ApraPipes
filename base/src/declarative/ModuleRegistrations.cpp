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
#ifndef APRAPIPES_NO_FFMPEG_MODULES
#include "RTSPClientSrc.h"
#endif
#include "ExternalSourceModule.h"

// Batch 2: Additional Transform modules
#include "OverlayModule.h"
#include "HistogramOverlay.h"
#include "BMPConverter.h"
#ifndef APRAPIPES_NO_FFMPEG_MODULES
#include "MotionVectorExtractor.h"
#endif
#include "AffineTransform.h"
#include "MultimediaQueueXform.h"

// Batch 3: Additional Sink modules
#include "ExternalSinkModule.h"
#ifndef APRAPIPES_NO_FFMPEG_MODULES
#include "RTSPPusher.h"
#endif
#include "ThumbnailListGenerator.h"

// Batch 4: Analytics modules with face detection
#include "FacialLandmarksCV.h"

// Batch 6: Additional source and utility modules
#include "AudioCaptureSrc.h"
#include "ArchiveSpaceManager.h"
#include "AudioToTextXForm.h"

// Platform-specific modules (conditionally compiled)
#ifdef ENABLE_LINUX
#include "VirtualCameraSink.h"
#include "H264EncoderV4L2.h"
#endif

// Jetson-specific modules (ARM64 + CUDA)
#ifdef ARM64
#include "NvArgusCamera.h"
#include "NvV4L2Camera.h"
#include "NvTransform.h"
#include "JPEGDecoderL4TM.h"
#include "JPEGEncoderL4TM.h"
#include "EglRenderer.h"
#include "DMAFDToHostCopy.h"
#endif

// Batch 5: Utility modules
#include "FramesMuxer.h"
// Note: H264FrameDemuxer is not derived from Module - cannot be registered

// Conditionally include CUDA modules
#ifdef ENABLE_CUDA
#include "H264Decoder.h"
#include "CudaCommon.h"
#include "GaussianBlur.h"
#include "ResizeNPPI.h"
#include "RotateNPPI.h"
#include "CCNPPI.h"
#include "EffectsNPPI.h"
#include "OverlayNPPI.h"
#include "CudaMemCopy.h"
#include "CudaStreamSynchronize.h"
#ifndef ARM64
// nvJPEG and NVCodec modules not available on ARM64/Jetson
#include "JPEGDecoderNVJPEG.h"
#include "JPEGEncoderNVJPEG.h"
#include "H264EncoderNVCodec.h"
#endif
#include "MemTypeConversion.h"
#include "CuCtxSynchronize.h"
#endif

#include <mutex>

namespace apra {

// ============================================================
// CUDA Module Registration Helper
// For modules that require cudastream_sp in their Props constructor
// ============================================================

#ifdef ENABLE_CUDA

// Helper to register a CUDA module with proper cudaFactory
// Usage:
//   registerCudaModule<GaussianBlur, GaussianBlurProps>("GaussianBlur")
//       .category(ModuleCategory::Transform)
//       .input("input", "RawImage")
//       .output("output", "RawImage")
//       .intProp("kernelSize", "Gaussian kernel size", false, 11, 3, 31)
//       .finalizeCuda([](const auto& props, cudastream_sp stream) {
//           GaussianBlurProps moduleProps(stream);
//           // Apply properties from props map...
//           if (auto it = props.find("kernelSize"); it != props.end()) {
//               if (auto* val = std::get_if<int64_t>(&it->second)) {
//                   moduleProps.kernelSize = static_cast<int>(*val);
//               }
//           }
//           return std::make_unique<GaussianBlur>(moduleProps);
//       });

template<typename ModuleClass, typename PropsClass>
class CudaModuleRegistrationBuilder {
    ModuleInfo info_;
    bool registered_ = false;

public:
    CudaModuleRegistrationBuilder(const std::string& name) {
        info_.name = name;
        info_.requiresCudaStream = true;
    }

    CudaModuleRegistrationBuilder& category(ModuleCategory cat) {
        info_.category = cat;
        return *this;
    }

    CudaModuleRegistrationBuilder& description(const std::string& desc) {
        info_.description = desc;
        return *this;
    }

    CudaModuleRegistrationBuilder& version(const std::string& ver) {
        info_.version = ver;
        return *this;
    }

    template<typename... Tags>
    CudaModuleRegistrationBuilder& tags(Tags... t) {
        (info_.tags.push_back(std::string(t)), ...);
        return *this;
    }

    CudaModuleRegistrationBuilder& input(const std::string& pinName, const std::string& frameType,
                                         MemType memType = FrameMetadata::HOST) {
        ModuleInfo::PinInfo pin;
        pin.name = pinName;
        pin.frame_types.push_back(frameType);
        pin.required = true;
        pin.memType = memType;
        info_.inputs.push_back(std::move(pin));
        return *this;
    }

    // Convenience method for CUDA input pins
    CudaModuleRegistrationBuilder& cudaInput(const std::string& pinName, const std::string& frameType) {
        return input(pinName, frameType, FrameMetadata::CUDA_DEVICE);
    }

    CudaModuleRegistrationBuilder& output(const std::string& pinName, const std::string& frameType,
                                          MemType memType = FrameMetadata::HOST) {
        ModuleInfo::PinInfo pin;
        pin.name = pinName;
        pin.frame_types.push_back(frameType);
        pin.required = true;
        pin.memType = memType;
        info_.outputs.push_back(std::move(pin));
        return *this;
    }

    // Convenience method for CUDA output pins
    CudaModuleRegistrationBuilder& cudaOutput(const std::string& pinName, const std::string& frameType) {
        return output(pinName, frameType, FrameMetadata::CUDA_DEVICE);
    }

    // Set image types on the last input pin
    template<typename... ImageTypes>
    CudaModuleRegistrationBuilder& inputImageTypes(ImageTypes... types) {
        if (!info_.inputs.empty()) {
            (info_.inputs.back().image_types.push_back(types), ...);
        }
        return *this;
    }

    // Set image types on the last output pin
    template<typename... ImageTypes>
    CudaModuleRegistrationBuilder& outputImageTypes(ImageTypes... types) {
        if (!info_.outputs.empty()) {
            (info_.outputs.back().image_types.push_back(types), ...);
        }
        return *this;
    }

    CudaModuleRegistrationBuilder& intProp(const std::string& name, const std::string& desc,
                                           bool required = false, int64_t defaultVal = 0,
                                           int64_t minVal = INT64_MIN, int64_t maxVal = INT64_MAX) {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = "int";
        prop.mutability = "static";
        prop.required = required;
        prop.default_value = std::to_string(defaultVal);
        if (minVal != INT64_MIN) prop.min_value = std::to_string(minVal);
        if (maxVal != INT64_MAX) prop.max_value = std::to_string(maxVal);
        prop.description = desc;
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    CudaModuleRegistrationBuilder& floatProp(const std::string& name, const std::string& desc,
                                             bool required = false, double defaultVal = 0.0) {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = "float";
        prop.mutability = "static";
        prop.required = required;
        prop.default_value = std::to_string(defaultVal);
        prop.description = desc;
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    CudaModuleRegistrationBuilder& boolProp(const std::string& name, const std::string& desc,
                                            bool required = false, bool defaultVal = false) {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = "bool";
        prop.mutability = "static";
        prop.required = required;
        prop.default_value = defaultVal ? "true" : "false";
        prop.description = desc;
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    template<typename... EnumValues>
    CudaModuleRegistrationBuilder& enumProp(const std::string& name, const std::string& desc,
                                            bool required, const std::string& defaultVal,
                                            EnumValues... values) {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = "enum";
        prop.mutability = "static";
        prop.required = required;
        prop.default_value = defaultVal;
        prop.description = desc;
        (prop.enum_values.push_back(std::string(values)), ...);
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    // Mark module as managing its own output pins (creates them in addInputPin)
    CudaModuleRegistrationBuilder& selfManagedOutputPins() {
        info_.selfManagedOutputPins = true;
        return *this;
    }

    // Finalize with CUDA factory - takes a lambda that creates the module
    template<typename CudaFactoryLambda>
    void finalizeCuda(CudaFactoryLambda&& cudaFactoryFn) {
        if (registered_) return;
        registered_ = true;

        if (info_.version.empty()) {
            info_.version = "1.0";
        }

        // Create type-erased CUDA factory
        info_.cudaFactory = [fn = std::forward<CudaFactoryLambda>(cudaFactoryFn)](
            const std::map<std::string, ScalarPropertyValue>& props,
            void* cudaStreamPtr
        ) -> std::unique_ptr<Module> {
            // Cast void* back to cudastream_sp*
            cudastream_sp* streamPtr = static_cast<cudastream_sp*>(cudaStreamPtr);
            if (!streamPtr || !*streamPtr) {
                throw std::runtime_error("CUDA stream not available");
            }
            return fn(props, *streamPtr);
        };

        // No regular factory - CUDA modules require CUDA stream
        info_.factory = nullptr;
        info_.requiresCudaStream = true;

        ModuleRegistry::instance().registerModule(std::move(info_));
    }

    // Prevent copying
    CudaModuleRegistrationBuilder(const CudaModuleRegistrationBuilder&) = delete;
    CudaModuleRegistrationBuilder& operator=(const CudaModuleRegistrationBuilder&) = delete;

    // Allow moving
    CudaModuleRegistrationBuilder(CudaModuleRegistrationBuilder&& other) noexcept
        : info_(std::move(other.info_)), registered_(other.registered_) {
        other.registered_ = true;
    }
};

template<typename ModuleClass, typename PropsClass>
CudaModuleRegistrationBuilder<ModuleClass, PropsClass> registerCudaModule(const std::string& name) {
    return CudaModuleRegistrationBuilder<ModuleClass, PropsClass>(name);
}

// ============================================================
// CuContext Module Registration Helper
// For modules that require apracucontext_sp in their Props constructor (NVCodec)
// ============================================================

template<typename ModuleClass, typename PropsClass>
class CuContextModuleRegistrationBuilder {
    ModuleInfo info_;
    bool registered_ = false;

public:
    CuContextModuleRegistrationBuilder(const std::string& name) {
        info_.name = name;
        info_.requiresCuContext = true;
    }

    CuContextModuleRegistrationBuilder& category(ModuleCategory cat) {
        info_.category = cat;
        return *this;
    }

    CuContextModuleRegistrationBuilder& description(const std::string& desc) {
        info_.description = desc;
        return *this;
    }

    CuContextModuleRegistrationBuilder& version(const std::string& ver) {
        info_.version = ver;
        return *this;
    }

    template<typename... Tags>
    CuContextModuleRegistrationBuilder& tags(Tags... t) {
        (info_.tags.push_back(std::string(t)), ...);
        return *this;
    }

    CuContextModuleRegistrationBuilder& input(const std::string& pinName, const std::string& frameType,
                                               MemType memType = FrameMetadata::HOST) {
        ModuleInfo::PinInfo pin;
        pin.name = pinName;
        pin.frame_types.push_back(frameType);
        pin.required = true;
        pin.memType = memType;
        info_.inputs.push_back(std::move(pin));
        return *this;
    }

    // Convenience method for CUDA input pins
    CuContextModuleRegistrationBuilder& cudaInput(const std::string& pinName, const std::string& frameType) {
        return input(pinName, frameType, FrameMetadata::CUDA_DEVICE);
    }

    CuContextModuleRegistrationBuilder& output(const std::string& pinName, const std::string& frameType,
                                                MemType memType = FrameMetadata::HOST) {
        ModuleInfo::PinInfo pin;
        pin.name = pinName;
        pin.frame_types.push_back(frameType);
        pin.required = true;
        pin.memType = memType;
        info_.outputs.push_back(std::move(pin));
        return *this;
    }

    CuContextModuleRegistrationBuilder& intProp(const std::string& name, const std::string& desc,
                                                 bool required = false, int64_t defaultVal = 0,
                                                 int64_t minVal = INT64_MIN, int64_t maxVal = INT64_MAX) {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = "int";
        prop.mutability = "static";
        prop.required = required;
        prop.default_value = std::to_string(defaultVal);
        if (minVal != INT64_MIN) prop.min_value = std::to_string(minVal);
        if (maxVal != INT64_MAX) prop.max_value = std::to_string(maxVal);
        prop.description = desc;
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    CuContextModuleRegistrationBuilder& boolProp(const std::string& name, const std::string& desc,
                                                  bool required = false, bool defaultVal = false) {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = "bool";
        prop.mutability = "static";
        prop.required = required;
        prop.default_value = defaultVal ? "true" : "false";
        prop.description = desc;
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    template<typename... EnumValues>
    CuContextModuleRegistrationBuilder& enumProp(const std::string& name, const std::string& desc,
                                                  bool required, const std::string& defaultVal,
                                                  EnumValues... values) {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = "enum";
        prop.mutability = "static";
        prop.required = required;
        prop.default_value = defaultVal;
        prop.description = desc;
        (prop.enum_values.push_back(std::string(values)), ...);
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    // Mark module as managing its own output pins
    CuContextModuleRegistrationBuilder& selfManagedOutputPins() {
        info_.selfManagedOutputPins = true;
        return *this;
    }

    // Finalize with CuContext factory - takes a lambda that creates the module
    template<typename CuContextFactoryLambda>
    void finalizeCuContext(CuContextFactoryLambda&& cuContextFactoryFn) {
        if (registered_) return;
        registered_ = true;

        if (info_.version.empty()) {
            info_.version = "1.0";
        }

        // Create type-erased CuContext factory
        info_.cuContextFactory = [fn = std::forward<CuContextFactoryLambda>(cuContextFactoryFn)](
            const std::map<std::string, ScalarPropertyValue>& props,
            void* cuContextPtr
        ) -> std::unique_ptr<Module> {
            // Cast void* back to apracucontext_sp*
            apracucontext_sp* contextPtr = static_cast<apracucontext_sp*>(cuContextPtr);
            if (!contextPtr || !*contextPtr) {
                throw std::runtime_error("CUDA context not available");
            }
            return fn(props, *contextPtr);
        };

        // No regular factory - CuContext modules require CUDA context
        info_.factory = nullptr;
        info_.requiresCuContext = true;

        ModuleRegistry::instance().registerModule(std::move(info_));
    }

    // Prevent copying
    CuContextModuleRegistrationBuilder(const CuContextModuleRegistrationBuilder&) = delete;
    CuContextModuleRegistrationBuilder& operator=(const CuContextModuleRegistrationBuilder&) = delete;

    // Allow moving
    CuContextModuleRegistrationBuilder(CuContextModuleRegistrationBuilder&& other) noexcept
        : info_(std::move(other.info_)), registered_(other.registered_) {
        other.registered_ = true;
    }
};

template<typename ModuleClass, typename PropsClass>
CuContextModuleRegistrationBuilder<ModuleClass, PropsClass> registerCuContextModule(const std::string& name) {
    return CuContextModuleRegistrationBuilder<ModuleClass, PropsClass>(name);
}

#endif // ENABLE_CUDA

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
            FileReaderModuleProps fileReaderDefaults;  // Query defaults from Props class
            registerModule<FileReaderModule, FileReaderModuleProps>()
                .category(ModuleCategory::Source)
                .description("Reads frames from files matching a pattern. Supports image sequences and raw frame files.")
                .tags("source", "file", "reader")
                .output("output", "Frame")  // Generic - actual type set via outputFrameType prop
                .stringProp("strFullFileNameWithPattern", "File path pattern (e.g., /path/frame_????.raw)", true)
                .intProp("startIndex", "Starting file index", false, fileReaderDefaults.startIndex, 0)
                .intProp("maxIndex", "Maximum file index (-1 for unlimited)", false, fileReaderDefaults.maxIndex, -1)
                .boolProp("readLoop", "Loop back to start when reaching end", false, fileReaderDefaults.readLoop)
                .enumProp("outputFrameType", "Output frame type", false, "Frame",
                    "Frame", "EncodedImage", "RawImage", "RawImagePlanar");
        }

        // FileWriterModule - writes frames to files
        if (!registry.hasModule("FileWriterModule")) {
            FileWriterModuleProps fileWriterDefaults;  // Query defaults from Props class
            registerModule<FileWriterModule, FileWriterModuleProps>()
                .category(ModuleCategory::Sink)
                .description("Writes frames to files. Supports file sequences with pattern-based naming.")
                .tags("sink", "file", "writer")
                .input("input", "Frame")
                .stringProp("strFullFileNameWithPattern", "Output file path pattern (e.g., /path/frame_????.raw)", true)
                .boolProp("append", "Append to existing files instead of overwriting", false, fileWriterDefaults.append);
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
                .intProp("height", "Frame height in pixels", true, 0, 1, 4096)
                .stringProp("pattern", "Test pattern type: GRADIENT, CHECKERBOARD, COLOR_BARS, GRID", false, "GRADIENT")
                .intProp("maxFrames", "Maximum frames to generate (0 = unlimited)", false, 0, 0, INT_MAX);
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
        // Set outputFormat to "h264" or "jpeg" for declarative pipelines
        if (!registry.hasModule("Mp4ReaderSource")) {
            registerModule<Mp4ReaderSource, Mp4ReaderSourceProps>()
                .category(ModuleCategory::Source)
                .description("Reads video frames from MP4 files. Set outputFormat='h264' or 'jpeg' for declarative use.")
                .tags("source", "mp4", "video", "file")
                .output("output", "H264Data", "EncodedImage")
                .stringProp("videoPath", "Path to MP4 video file", true)
                .boolProp("parseFS", "Parse filesystem for metadata", false, true)
                .boolProp("direction", "Playback direction (true=forward)", false, true)
                .boolProp("bFramesEnabled", "Enable B-frame decoding", false, false)
                .intProp("reInitInterval", "Re-initialization interval in seconds (0=disabled)", false, 0, 0)
                .intProp("parseFSTimeoutDuration", "Filesystem parse timeout in seconds", false, 15, 0)
                .boolProp("readLoop", "Loop playback when reaching end", false, false)
                .boolProp("giveLiveTS", "Use live timestamps", false, false)
                .stringProp("outputFormat", "Output format: 'h264' or 'jpeg' (required for declarative)", false, "")
                .selfManagedOutputPins();  // Output pins created in constructor when outputFormat is set
        }

        // Mp4WriterSink - writes MP4 video files
        if (!registry.hasModule("Mp4WriterSink")) {
            registerModule<Mp4WriterSink, Mp4WriterSinkProps>()
                .category(ModuleCategory::Sink)
                .description("Writes video frames to MP4 files")
                .tags("sink", "mp4", "video", "file")
                .input("input", "H264Data", "EncodedImage")
                .stringProp("baseFolder", "Output folder for MP4 files", false, "./data/Mp4_videos/")
                .intProp("chunkTime", "Chunk duration in minutes (1-60)", false, 1, 1, 60)
                .intProp("syncTimeInSecs", "Sync interval in seconds (1-60)", false, 1, 1, 60)
                .intProp("fps", "Output frame rate", false, 30, 1, 120)
                .boolProp("recordedTSBasedDTS", "Use recorded timestamps for DTS", false, true)
                .boolProp("enableMetadata", "Enable metadata track", false, true);
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
                .dynamicProp("isDateTime", "bool", "Display current date/time instead of text", false, "false")
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

#ifndef APRAPIPES_NO_FFMPEG_MODULES
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
#endif

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

#ifndef APRAPIPES_NO_FFMPEG_MODULES
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
#endif

        // AffineTransform - applies affine transformations (rotation, scale, translate)
        // Note: Creates output pin in addInputPin(), so selfManagedOutputPins is required
        if (!registry.hasModule("AffineTransform")) {
            AffineTransformProps affineDefaults;  // Query defaults from Props class
            registerModule<AffineTransform, AffineTransformProps>()
                .category(ModuleCategory::Transform)
                .description("Applies affine transformations to images including rotation, scaling, and translation using OpenCV.")
                .tags("transform", "affine", "rotate", "scale", "translate", "opencv")
                .input("input", "RawImage")  // Only accepts RawImage (not planar)
                .output("output", "RawImage")
                .floatProp("angle", "Rotation angle in degrees", false, affineDefaults.angle, -360.0, 360.0)
                .intProp("x", "Horizontal translation in pixels", false, affineDefaults.x)
                .intProp("y", "Vertical translation in pixels", false, affineDefaults.y)
                .floatProp("scale", "Scale factor (1.0 = no scaling)", false, affineDefaults.scale, 0.01, 100.0)
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

#ifndef APRAPIPES_NO_FFMPEG_MODULES
        // RTSPPusher - pushes video to RTSP server
        if (!registry.hasModule("RTSPPusher")) {
            registerModule<RTSPPusher, RTSPPusherProps>()
                .category(ModuleCategory::Sink)
                .description("Pushes video stream to an RTSP server for remote viewing")
                .tags("sink", "rtsp", "network", "stream", "push")
                .input("input", "H264Data")
                .stringProp("url", "RTSP server URL to push to", true)
                .stringProp("title", "Stream title", false, "stream")
                .boolProp("isTCP", "Use TCP transport", false, true)
                .intProp("encoderTargetKbps", "Target bitrate in Kbps", false, 2048, 100, 50000);
        }
#endif

        // ThumbnailListGenerator - generates thumbnail strip (ARM/Jetson only)
        if (!registry.hasModule("ThumbnailListGenerator")) {
            registerModule<ThumbnailListGenerator, ThumbnailListGeneratorProps>()
                .category(ModuleCategory::Sink)
                .description("Generates thumbnail strip image from video frames (ARM/Jetson platforms only)")
                .tags("sink", "thumbnail", "image", "jetson")
                .input("input", "RawImagePlanar")
                .intProp("thumbnailWidth", "Thumbnail width in pixels", false, 128, 16, 1024)
                .intProp("thumbnailHeight", "Thumbnail height in pixels", false, 128, 16, 1024)
                .stringProp("fileToStore", "Output file path for thumbnail strip", true);
        }

        // ============================================================
        // Batch 4: Analytics Modules
        // ============================================================

        // FacialLandmarkCV - facial landmark detection
        // Note: This is a transform module that:
        // - Modifies the input image (draws green rectangles on detected faces)
        // - Outputs landmarks data (FaceLandmarksInfo)
        // With sieve=false (default), the modified input image passes through automatically
        if (!registry.hasModule("FacialLandmarkCV")) {
            registerModule<FacialLandmarkCV, FacialLandmarkCVProps>()
                .category(ModuleCategory::Analytics)
                .description("Detects facial landmarks (eyes, nose, mouth) using OpenCV face module. Draws green rectangles on faces and passes through the modified image. Supports SSD and Haar cascade detection methods.")
                .tags("analytics", "face", "landmarks", "opencv", "detection")
                .input("input", "RawImage")
                .output("landmarks", "FaceLandmarksInfo")
                .enumProp("modelType", "Face detection model type", false, "SSD", "SSD", "HAAR_CASCADE")
                .stringProp("faceDetectionConfig", "Path to SSD config file", false, "./data/assets/deploy.prototxt")
                .stringProp("faceDetectionWeights", "Path to SSD weights file", false, "./data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel")
                .stringProp("landmarksModel", "Path to facial landmarks model", false, "./data/assets/face_landmark_model.dat")
                .stringProp("haarCascadeModel", "Path to Haar cascade model", false, "./data/assets/haarcascade.xml")
                .selfManagedOutputPins();
        }

        // ============================================================
        // Batch 6: Additional Source and Utility Modules
        // ============================================================

        // AudioCaptureSrc - captures audio from microphone
        if (!registry.hasModule("AudioCaptureSrc")) {
            registerModule<AudioCaptureSrc, AudioCaptureSrcProps>()
                .category(ModuleCategory::Source)
                .description("Captures audio from microphone or audio input device using PortAudio")
                .tags("source", "audio", "capture", "microphone")
                .output("output", "AudioFrame")
                .intProp("sampleRate", "Audio sample rate in Hz", false, 44100, 8000, 192000)
                .intProp("channels", "Number of audio channels", false, 2, 1, 8)
                .intProp("audioInputDeviceIndex", "Audio device index (0 = default)", false, 0, 0, 100)
                .intProp("processingIntervalMS", "Processing interval in milliseconds", false, 100, 10, 1000);
        }

        // ArchiveSpaceManager - manages disk space for video archives
        if (!registry.hasModule("ArchiveSpaceManager")) {
            registerModule<ArchiveSpaceManager, ArchiveSpaceManagerProps>()
                .category(ModuleCategory::Utility)
                .description("Monitors and manages disk space by deleting oldest files when storage exceeds threshold")
                .tags("utility", "archive", "storage", "disk", "management")
                .stringProp("pathToWatch", "Directory path to monitor for space management", true)
                .intProp("lowerWaterMark", "Lower threshold in bytes - stop deleting when reached", true, 0)
                .intProp("upperWaterMark", "Upper threshold in bytes - start deleting when exceeded", true, 0)
                .intProp("samplingFreq", "Sampling frequency for size estimation", false, 60, 1, 1000);
        }

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
        // Batch 7: Additional Transform/Analytics Modules
        // ============================================================

        // AudioToTextXForm - speech-to-text using Whisper
        if (!registry.hasModule("AudioToTextXForm")) {
            registerModule<AudioToTextXForm, AudioToTextXFormProps>()
                .category(ModuleCategory::Transform)
                .description("Transforms audio input to text using Whisper speech recognition model")
                .tags("transform", "audio", "speech", "text", "whisper", "ml")
                .input("input", "AudioFrame")
                .output("output", "TextFrame")
                .stringProp("modelPath", "Path to Whisper model file", true)
                .intProp("bufferSize", "Audio buffer size in samples", false, 16000, 1000, 100000)
                .enumProp("samplingStrategy", "Decoder sampling strategy", false, "GREEDY", "GREEDY", "BEAM_SEARCH")
                .selfManagedOutputPins();
        }

        // ============================================================
        // Linux-only modules
        // ============================================================
#ifdef ENABLE_LINUX
        // VirtualCameraSink - outputs to virtual camera device
        if (!registry.hasModule("VirtualCameraSink")) {
            registerModule<VirtualCameraSink, VirtualCameraSinkProps>()
                .category(ModuleCategory::Sink)
                .description("Outputs video frames to a virtual camera device (e.g., /dev/video0 on Linux)")
                .tags("sink", "camera", "virtual", "v4l2")
                .input("input", "RawImage")
                .stringProp("device", "Virtual camera device path (e.g., /dev/video0)", true);
        }
        // H264EncoderV4L2 - H.264 encoding via V4L2
        if (!registry.hasModule("H264EncoderV4L2")) {
            registerModule<H264EncoderV4L2, H264EncoderV4L2Props>()
                .category(ModuleCategory::Transform)
                .description("Encodes raw video frames to H.264 using V4L2 hardware encoder (Linux only)")
                .tags("encoder", "h264", "video", "v4l2", "linux")
                .input("input", "RawImagePlanar")
                .output("output", "H264Data")
                .intProp("targetKbps", "Target bitrate in Kbps", false, 1024, 100, 50000)
                .boolProp("enableMotionVectors", "Enable motion vector extraction", false, false)
                .intProp("motionVectorThreshold", "Motion vector threshold", false, 5, 0, 100)
                .selfManagedOutputPins();
        }
#endif

        // ============================================================
        // Jetson/ARM64-only modules
        // ============================================================
        // Note: Jetson modules use direct ModuleInfo construction to avoid
        // GCC 9 template lambda issues with registerModule<>() template.
        // Also use explicit PropInfo/PinInfo construction for GCC 9 compatibility.
#ifdef ARM64
        // Helper lambdas for GCC 9 compatible PropInfo construction
        auto makeIntProp = [](const std::string& name, const std::string& desc,
                              bool required, const std::string& defVal,
                              const std::string& minVal, const std::string& maxVal) {
            ModuleInfo::PropInfo p;
            p.name = name; p.type = "int"; p.mutability = "static";
            p.required = required; p.default_value = defVal;
            p.min_value = minVal; p.max_value = maxVal; p.description = desc;
            return p;
        };
        auto makeFloatProp = [](const std::string& name, const std::string& desc,
                                bool required, const std::string& defVal,
                                const std::string& minVal, const std::string& maxVal) {
            ModuleInfo::PropInfo p;
            p.name = name; p.type = "float"; p.mutability = "static";
            p.required = required; p.default_value = defVal;
            p.min_value = minVal; p.max_value = maxVal; p.description = desc;
            return p;
        };
        auto makeBoolProp = [](const std::string& name, const std::string& desc,
                               bool required, const std::string& defVal) {
            ModuleInfo::PropInfo p;
            p.name = name; p.type = "bool"; p.mutability = "static";
            p.required = required; p.default_value = defVal; p.description = desc;
            return p;
        };
        auto makeEnumProp = [](const std::string& name, const std::string& desc,
                               bool required, const std::string& defVal,
                               const std::vector<std::string>& values) {
            ModuleInfo::PropInfo p;
            p.name = name; p.type = "enum"; p.mutability = "static";
            p.required = required; p.default_value = defVal;
            p.enum_values = values; p.description = desc;
            return p;
        };
        // NvArgusCamera - Jetson CSI camera via Argus API
        if (!registry.hasModule("NvArgusCamera")) {
            ModuleInfo info;
            info.name = "NvArgusCamera";
            info.category = ModuleCategory::Source;
            info.description = "Captures video from Jetson CSI camera using NVIDIA Argus API";
            info.version = "1.0";
            info.tags = {"source", "camera", "jetson", "argus", "csi", "arm64"};
            { ModuleInfo::PinInfo pin; pin.name = "output"; pin.frame_types = {"RawImagePlanar"}; pin.memType = FrameMetadata::DMABUF; info.outputs.push_back(std::move(pin)); }
            info.properties.push_back(makeIntProp("width", "Capture width in pixels", true, "1920", "320", "4096"));
            info.properties.push_back(makeIntProp("height", "Capture height in pixels", true, "1080", "240", "2160"));
            info.properties.push_back(makeIntProp("cameraId", "CSI camera sensor ID", false, "0", "0", "7"));
            info.selfManagedOutputPins = true;
            info.factory = [](const std::map<std::string, ScalarPropertyValue>& props) -> std::unique_ptr<Module> {
                auto width = static_cast<uint32_t>(std::get<int64_t>(props.at("width")));
                auto height = static_cast<uint32_t>(std::get<int64_t>(props.at("height")));
                NvArgusCameraProps moduleProps(width, height);
                if (props.count("cameraId")) {
                    moduleProps.cameraId = static_cast<int>(std::get<int64_t>(props.at("cameraId")));
                }
                return std::make_unique<NvArgusCamera>(moduleProps);
            };
            registry.registerModule(std::move(info));
        }

        // NvV4L2Camera - Jetson USB camera via V4L2
        if (!registry.hasModule("NvV4L2Camera")) {
            ModuleInfo info;
            info.name = "NvV4L2Camera";
            info.category = ModuleCategory::Source;
            info.description = "Captures video from USB camera on Jetson using V4L2";
            info.version = "1.0";
            info.tags = {"source", "camera", "jetson", "v4l2", "usb", "arm64"};
            { ModuleInfo::PinInfo pin; pin.name = "output"; pin.frame_types = {"RawImagePlanar"}; pin.memType = FrameMetadata::DMABUF; info.outputs.push_back(std::move(pin)); }
            info.properties.push_back(makeIntProp("width", "Capture width in pixels", true, "640", "1", "4096"));
            info.properties.push_back(makeIntProp("height", "Capture height in pixels", true, "480", "1", "4096"));
            info.properties.push_back(makeIntProp("maxConcurrentFrames", "Maximum concurrent frames buffer", false, "10", "1", "100"));
            info.properties.push_back(makeBoolProp("isMirror", "Mirror image horizontally", false, "false"));
            info.selfManagedOutputPins = true;
            info.factory = [](const std::map<std::string, ScalarPropertyValue>& props) -> std::unique_ptr<Module> {
                auto width = static_cast<uint32_t>(std::get<int64_t>(props.at("width")));
                auto height = static_cast<uint32_t>(std::get<int64_t>(props.at("height")));
                uint32_t maxConcurrentFrames = 10;
                if (props.count("maxConcurrentFrames")) {
                    maxConcurrentFrames = static_cast<uint32_t>(std::get<int64_t>(props.at("maxConcurrentFrames")));
                }
                NvV4L2CameraProps moduleProps(width, height, maxConcurrentFrames);
                if (props.count("isMirror")) {
                    moduleProps.isMirror = std::get<bool>(props.at("isMirror"));
                }
                return std::make_unique<NvV4L2Camera>(moduleProps);
            };
            registry.registerModule(std::move(info));
        }

        // NvTransform - GPU-accelerated resize/crop/transform on Jetson
        if (!registry.hasModule("NvTransform")) {
            ModuleInfo info;
            info.name = "NvTransform";
            info.category = ModuleCategory::Transform;
            info.description = "GPU-accelerated image resize, crop, and transform using Jetson hardware";
            info.version = "1.0";
            info.tags = {"transform", "resize", "crop", "jetson", "gpu", "arm64"};
            { ModuleInfo::PinInfo pin; pin.name = "input"; pin.frame_types = {"RawImagePlanar"}; pin.memType = FrameMetadata::DMABUF; info.inputs.push_back(std::move(pin)); }
            { ModuleInfo::PinInfo pin; pin.name = "output"; pin.frame_types = {"RawImagePlanar"}; pin.memType = FrameMetadata::DMABUF; info.outputs.push_back(std::move(pin)); }
            info.properties.push_back(makeEnumProp("imageType", "Output image format", true, "YUV420", {"RGB", "BGR", "RGBA", "BGRA", "MONO", "YUV420", "YUV444", "NV12", "UYVY", "YUYV"}));
            info.properties.push_back(makeIntProp("width", "Output width in pixels (0 = input width)", false, "0", "0", "8192"));
            info.properties.push_back(makeIntProp("height", "Output height in pixels (0 = input height)", false, "0", "0", "8192"));
            info.properties.push_back(makeIntProp("top", "Crop top offset", false, "0", "0", "8192"));
            info.properties.push_back(makeIntProp("left", "Crop left offset", false, "0", "0", "8192"));
            info.properties.push_back(makeFloatProp("scaleWidth", "Scale factor for width", false, "1.0", "0.01", "10.0"));
            info.properties.push_back(makeFloatProp("scaleHeight", "Scale factor for height", false, "1.0", "0.01", "10.0"));
            info.properties.push_back(makeEnumProp("filterType", "Interpolation filter type", false, "SMART", {"NEAREST", "BILINEAR", "TAP_5", "TAP_10", "SMART", "NICEST"}));
            info.selfManagedOutputPins = true;
            info.factory = [](const std::map<std::string, ScalarPropertyValue>& props) -> std::unique_ptr<Module> {
                auto imageTypeStr = std::get<std::string>(props.at("imageType"));
                ImageMetadata::ImageType imageType = ImageMetadata::YUV420;
                if (imageTypeStr == "RGB") imageType = ImageMetadata::RGB;
                else if (imageTypeStr == "BGR") imageType = ImageMetadata::BGR;
                else if (imageTypeStr == "RGBA") imageType = ImageMetadata::RGBA;
                else if (imageTypeStr == "BGRA") imageType = ImageMetadata::BGRA;
                else if (imageTypeStr == "MONO") imageType = ImageMetadata::MONO;
                else if (imageTypeStr == "YUV444") imageType = ImageMetadata::YUV444;
                else if (imageTypeStr == "NV12") imageType = ImageMetadata::NV12;
                else if (imageTypeStr == "UYVY") imageType = ImageMetadata::UYVY;
                else if (imageTypeStr == "YUYV") imageType = ImageMetadata::YUYV;

                NvTransformProps moduleProps(imageType);
                if (props.count("width")) moduleProps.width = static_cast<int>(std::get<int64_t>(props.at("width")));
                if (props.count("height")) moduleProps.height = static_cast<int>(std::get<int64_t>(props.at("height")));
                if (props.count("top")) moduleProps.top = static_cast<int>(std::get<int64_t>(props.at("top")));
                if (props.count("left")) moduleProps.left = static_cast<int>(std::get<int64_t>(props.at("left")));
                if (props.count("scaleWidth")) moduleProps.scaleWidth = static_cast<float>(std::get<double>(props.at("scaleWidth")));
                if (props.count("scaleHeight")) moduleProps.scaleHeight = static_cast<float>(std::get<double>(props.at("scaleHeight")));
                return std::make_unique<NvTransform>(moduleProps);
            };
            registry.registerModule(std::move(info));
        }

        // JPEGDecoderL4TM - Hardware JPEG decoder on Jetson
        if (!registry.hasModule("JPEGDecoderL4TM")) {
            ModuleInfo info;
            info.name = "JPEGDecoderL4TM";
            info.category = ModuleCategory::Transform;
            info.description = "Hardware-accelerated JPEG decoder using Jetson L4T Multimedia API";
            info.version = "1.0";
            info.tags = {"decoder", "jpeg", "image", "jetson", "l4tm", "arm64"};
            { ModuleInfo::PinInfo pin; pin.name = "input"; pin.frame_types = {"EncodedImage"}; pin.memType = FrameMetadata::HOST; info.inputs.push_back(std::move(pin)); }
            { ModuleInfo::PinInfo pin; pin.name = "output"; pin.frame_types = {"RawImagePlanar"}; pin.memType = FrameMetadata::DMABUF; info.outputs.push_back(std::move(pin)); }
            info.selfManagedOutputPins = true;
            info.factory = [](const std::map<std::string, ScalarPropertyValue>&) -> std::unique_ptr<Module> {
                JPEGDecoderL4TMProps moduleProps;
                return std::make_unique<JPEGDecoderL4TM>(moduleProps);
            };
            registry.registerModule(std::move(info));
        }

        // JPEGEncoderL4TM - Hardware JPEG encoder on Jetson
        if (!registry.hasModule("JPEGEncoderL4TM")) {
            ModuleInfo info;
            info.name = "JPEGEncoderL4TM";
            info.category = ModuleCategory::Transform;
            info.description = "Hardware-accelerated JPEG encoder using Jetson L4T Multimedia API";
            info.version = "1.0";
            info.tags = {"encoder", "jpeg", "image", "jetson", "l4tm", "arm64"};
            { ModuleInfo::PinInfo pin; pin.name = "input"; pin.frame_types = {"RawImagePlanar"}; pin.memType = FrameMetadata::DMABUF; info.inputs.push_back(std::move(pin)); }
            { ModuleInfo::PinInfo pin; pin.name = "output"; pin.frame_types = {"EncodedImage"}; pin.memType = FrameMetadata::HOST; info.outputs.push_back(std::move(pin)); }
            info.properties.push_back(makeIntProp("quality", "JPEG quality (1-100)", false, "90", "1", "100"));
            info.properties.push_back(makeFloatProp("scale", "Output scale factor", false, "1.0", "0.1", "4.0"));
            info.selfManagedOutputPins = true;
            info.factory = [](const std::map<std::string, ScalarPropertyValue>& props) -> std::unique_ptr<Module> {
                JPEGEncoderL4TMProps moduleProps;
                if (props.count("quality")) {
                    moduleProps.quality = static_cast<unsigned short>(std::get<int64_t>(props.at("quality")));
                }
                if (props.count("scale")) {
                    moduleProps.scale = static_cast<float>(std::get<double>(props.at("scale")));
                }
                return std::make_unique<JPEGEncoderL4TM>(moduleProps);
            };
            registry.registerModule(std::move(info));
        }

        // EglRenderer - Display renderer on Jetson
        if (!registry.hasModule("EglRenderer")) {
            ModuleInfo info;
            info.name = "EglRenderer";
            info.category = ModuleCategory::Sink;
            info.description = "Renders frames to display using EGL on Jetson (requires display or Xvfb)";
            info.version = "1.0";
            info.tags = {"sink", "display", "render", "egl", "jetson", "arm64"};
            { ModuleInfo::PinInfo pin; pin.name = "input"; pin.frame_types = {"RawImagePlanar"}; pin.memType = FrameMetadata::DMABUF; info.inputs.push_back(std::move(pin)); }
            info.properties.push_back(makeIntProp("x_offset", "Window X position", false, "0", "0", "8192"));
            info.properties.push_back(makeIntProp("y_offset", "Window Y position", false, "0", "0", "8192"));
            info.properties.push_back(makeIntProp("width", "Window width (0 = auto from input)", false, "0", "0", "8192"));
            info.properties.push_back(makeIntProp("height", "Window height (0 = auto from input)", false, "0", "0", "8192"));
            info.properties.push_back(makeBoolProp("displayOnTop", "Keep window always on top", false, "true"));
            info.factory = [](const std::map<std::string, ScalarPropertyValue>& props) -> std::unique_ptr<Module> {
                uint32_t x_offset = 0, y_offset = 0, width = 0, height = 0;
                bool displayOnTop = true;
                if (props.count("x_offset")) x_offset = static_cast<uint32_t>(std::get<int64_t>(props.at("x_offset")));
                if (props.count("y_offset")) y_offset = static_cast<uint32_t>(std::get<int64_t>(props.at("y_offset")));
                if (props.count("width")) width = static_cast<uint32_t>(std::get<int64_t>(props.at("width")));
                if (props.count("height")) height = static_cast<uint32_t>(std::get<int64_t>(props.at("height")));
                if (props.count("displayOnTop")) displayOnTop = std::get<bool>(props.at("displayOnTop"));

                if (width > 0 && height > 0) {
                    EglRendererProps moduleProps(x_offset, y_offset, width, height);
                    return std::make_unique<EglRenderer>(moduleProps);
                } else {
                    EglRendererProps moduleProps(x_offset, y_offset, displayOnTop);
                    return std::make_unique<EglRenderer>(moduleProps);
                }
            };
            registry.registerModule(std::move(info));
        }

        // DMAFDToHostCopy - Bridge module for DMABUF to HOST memory
        if (!registry.hasModule("DMAFDToHostCopy")) {
            ModuleInfo info;
            info.name = "DMAFDToHostCopy";
            info.category = ModuleCategory::Utility;
            info.description = "Copies frame data from DMA buffer to host (CPU) memory";
            info.version = "1.0";
            info.tags = {"utility", "memory", "copy", "dma", "bridge", "jetson", "arm64"};
            { ModuleInfo::PinInfo pin; pin.name = "input"; pin.frame_types = {"RawImagePlanar"}; pin.memType = FrameMetadata::DMABUF; info.inputs.push_back(std::move(pin)); }
            { ModuleInfo::PinInfo pin; pin.name = "output"; pin.frame_types = {"RawImagePlanar"}; pin.memType = FrameMetadata::HOST; info.outputs.push_back(std::move(pin)); }
            info.selfManagedOutputPins = true;
            info.factory = [](const std::map<std::string, ScalarPropertyValue>&) -> std::unique_ptr<Module> {
                DMAFDToHostCopyProps moduleProps;
                return std::make_unique<DMAFDToHostCopy>(moduleProps);
            };
            registry.registerModule(std::move(info));
        }
#endif // ARM64

        // ============================================================
        // CUDA-only modules
        // ============================================================
#ifdef ENABLE_CUDA
        // H264Decoder - decodes H.264 video
        // Input: HOST memory (encoded H264), Output: CUDA_DEVICE memory (decoded RawImagePlanar)
        if (!registry.hasModule("H264Decoder")) {
            registerModule<H264Decoder, H264DecoderProps>()
                .category(ModuleCategory::Transform)
                .description("Decodes H.264/AVC encoded video frames to raw image frames.")
                .tags("decoder", "h264", "video", "transform", "cuda")
                .input("input", "H264Frame", FrameMetadata::HOST)
                .cudaOutput("output", "RawImagePlanar");
        }

        // ============================================================
        // CUDA Stream-dependent modules (require cudastream_sp)
        // ============================================================

        // GaussianBlur - GPU-accelerated Gaussian blur
        if (!registry.hasModule("GaussianBlur")) {
            registerCudaModule<GaussianBlur, GaussianBlurProps>("GaussianBlur")
                .category(ModuleCategory::Transform)
                .description("Applies Gaussian blur filter using CUDA/NPP.")
                .tags("transform", "blur", "cuda", "npp")
                .cudaInput("input", "RawImage")
                .cudaOutput("output", "RawImage")
                .intProp("kernelSize", "Gaussian kernel size (must be odd)", false, 11, 3, 31)
                .selfManagedOutputPins()  // Creates output pin in addInputPin()
                .finalizeCuda([](const auto& props, cudastream_sp stream) {
                    int kernelSize = 11;
                    if (auto it = props.find("kernelSize"); it != props.end()) {
                        if (auto* val = std::get_if<int64_t>(&it->second)) {
                            kernelSize = static_cast<int>(*val);
                        }
                    }
                    GaussianBlurProps moduleProps(stream, kernelSize);
                    return std::make_unique<GaussianBlur>(moduleProps);
                });
        }

        // ResizeNPPI - GPU-accelerated image resizing
        if (!registry.hasModule("ResizeNPPI")) {
            registerCudaModule<ResizeNPPI, ResizeNPPIProps>("ResizeNPPI")
                .category(ModuleCategory::Transform)
                .description("Resizes images using NVIDIA Performance Primitives (NPP).")
                .tags("transform", "resize", "cuda", "npp")
                .cudaInput("input", "RawImage")
                .cudaOutput("output", "RawImage")
                .intProp("width", "Output width in pixels", true, 0, 1, 8192)
                .intProp("height", "Output height in pixels", true, 0, 1, 8192)
                .selfManagedOutputPins()  // Creates output pin in addInputPin()
                .finalizeCuda([](const auto& props, cudastream_sp stream) {
                    int width = 640, height = 480;
                    if (auto it = props.find("width"); it != props.end()) {
                        if (auto* val = std::get_if<int64_t>(&it->second)) {
                            width = static_cast<int>(*val);
                        }
                    }
                    if (auto it = props.find("height"); it != props.end()) {
                        if (auto* val = std::get_if<int64_t>(&it->second)) {
                            height = static_cast<int>(*val);
                        }
                    }
                    ResizeNPPIProps moduleProps(width, height, stream);
                    return std::make_unique<ResizeNPPI>(moduleProps);
                });
        }

        // RotateNPPI - GPU-accelerated image rotation
        if (!registry.hasModule("RotateNPPI")) {
            registerCudaModule<RotateNPPI, RotateNPPIProps>("RotateNPPI")
                .category(ModuleCategory::Transform)
                .description("Rotates images using NVIDIA Performance Primitives (NPP). Only supports 90-degree increments.")
                .tags("transform", "rotate", "cuda", "npp")
                .cudaInput("input", "RawImage")
                .cudaOutput("output", "RawImage")
                .floatProp("angle", "Rotation angle in degrees (must be 90, 180, or 270)", true, 0.0)
                .selfManagedOutputPins()  // Creates output pin in addInputPin()
                .finalizeCuda([](const auto& props, cudastream_sp stream) {
                    double angle = 0.0;
                    if (auto it = props.find("angle"); it != props.end()) {
                        if (auto* val = std::get_if<double>(&it->second)) {
                            angle = *val;
                        } else if (auto* val = std::get_if<int64_t>(&it->second)) {
                            angle = static_cast<double>(*val);
                        }
                    }
                    RotateNPPIProps moduleProps(stream, angle);
                    return std::make_unique<RotateNPPI>(moduleProps);
                });
        }

        // CCNPPI - GPU-accelerated color conversion
        if (!registry.hasModule("CCNPPI")) {
            registerCudaModule<CCNPPI, CCNPPIProps>("CCNPPI")
                .category(ModuleCategory::Transform)
                .description("Converts image color spaces using NVIDIA Performance Primitives (NPP).")
                .tags("transform", "color", "conversion", "cuda", "npp")
                .cudaInput("input", "RawImage")
                .cudaOutput("output", "RawImage")
                .enumProp("imageType", "Target image type", true, "RGB",
                    "RGB", "BGR", "RGBA", "BGRA", "MONO", "YUV420", "YUV444", "NV12")
                .selfManagedOutputPins()  // Creates output pin in addInputPin()
                .finalizeCuda([](const auto& props, cudastream_sp stream) {
                    ImageMetadata::ImageType imageType = ImageMetadata::RGB;
                    if (auto it = props.find("imageType"); it != props.end()) {
                        if (auto* val = std::get_if<std::string>(&it->second)) {
                            if (*val == "RGB") imageType = ImageMetadata::RGB;
                            else if (*val == "BGR") imageType = ImageMetadata::BGR;
                            else if (*val == "RGBA") imageType = ImageMetadata::RGBA;
                            else if (*val == "BGRA") imageType = ImageMetadata::BGRA;
                            else if (*val == "MONO") imageType = ImageMetadata::MONO;
                            else if (*val == "YUV420") imageType = ImageMetadata::YUV420;
                            else if (*val == "YUV444") imageType = ImageMetadata::YUV444;
                            else if (*val == "NV12") imageType = ImageMetadata::NV12;
                        }
                    }
                    CCNPPIProps moduleProps(imageType, stream);
                    return std::make_unique<CCNPPI>(moduleProps);
                });
        }

        // EffectsNPPI - GPU-accelerated image effects
        if (!registry.hasModule("EffectsNPPI")) {
            registerCudaModule<EffectsNPPI, EffectsNPPIProps>("EffectsNPPI")
                .category(ModuleCategory::Transform)
                .description("Applies image effects (brightness, contrast, saturation, hue) using NPP.")
                .tags("transform", "effects", "brightness", "contrast", "cuda", "npp")
                .cudaInput("input", "RawImage")
                .cudaOutput("output", "RawImage")
                .floatProp("hue", "Hue adjustment (0-255, 0=no change)", false, 0.0)
                .floatProp("saturation", "Saturation multiplier (1=no change)", false, 1.0)
                .floatProp("contrast", "Contrast multiplier (1=no change)", false, 1.0)
                .intProp("brightness", "Brightness adjustment (-100 to 100)", false, 0, -100, 100)
                .selfManagedOutputPins()  // Creates output pin in addInputPin()
                .finalizeCuda([](const auto& props, cudastream_sp stream) {
                    EffectsNPPIProps moduleProps(stream);
                    if (auto it = props.find("hue"); it != props.end()) {
                        if (auto* val = std::get_if<double>(&it->second)) moduleProps.hue = *val;
                    }
                    if (auto it = props.find("saturation"); it != props.end()) {
                        if (auto* val = std::get_if<double>(&it->second)) moduleProps.saturation = *val;
                    }
                    if (auto it = props.find("contrast"); it != props.end()) {
                        if (auto* val = std::get_if<double>(&it->second)) moduleProps.contrast = *val;
                    }
                    if (auto it = props.find("brightness"); it != props.end()) {
                        if (auto* val = std::get_if<int64_t>(&it->second)) moduleProps.brightness = static_cast<int>(*val);
                    }
                    return std::make_unique<EffectsNPPI>(moduleProps);
                });
        }

        // OverlayNPPI - GPU-accelerated image overlay
        if (!registry.hasModule("OverlayNPPI")) {
            registerCudaModule<OverlayNPPI, OverlayNPPIProps>("OverlayNPPI")
                .category(ModuleCategory::Transform)
                .description("Overlays one image on another using NPP with alpha blending.")
                .tags("transform", "overlay", "blend", "cuda", "npp")
                .cudaInput("input", "RawImage")
                .cudaOutput("output", "RawImage")
                .intProp("offsetX", "X offset for overlay", false, 0)
                .intProp("offsetY", "Y offset for overlay", false, 0)
                .intProp("globalAlpha", "Global alpha value (-1 for source alpha)", false, -1, -1, 255)
                .finalizeCuda([](const auto& props, cudastream_sp stream) {
                    OverlayNPPIProps moduleProps(stream);
                    if (auto it = props.find("offsetX"); it != props.end()) {
                        if (auto* val = std::get_if<int64_t>(&it->second)) moduleProps.offsetX = static_cast<int>(*val);
                    }
                    if (auto it = props.find("offsetY"); it != props.end()) {
                        if (auto* val = std::get_if<int64_t>(&it->second)) moduleProps.offsetY = static_cast<int>(*val);
                    }
                    if (auto it = props.find("globalAlpha"); it != props.end()) {
                        if (auto* val = std::get_if<int64_t>(&it->second)) moduleProps.globalAlpha = static_cast<int>(*val);
                    }
                    return std::make_unique<OverlayNPPI>(moduleProps);
                });
        }

        // CudaMemCopyH2D - Host to Device memory transfer (BRIDGE MODULE)
        // Used by auto-bridging when HOST → CUDA_DEVICE transfer is needed
        if (!registry.hasModule("CudaMemCopyH2D")) {
            registerCudaModule<CudaMemCopy, CudaMemCopyProps>("CudaMemCopyH2D")
                .category(ModuleCategory::Utility)
                .description("Copies data from host (CPU) to device (GPU) memory.")
                .tags("utility", "memory", "cuda", "transfer", "bridge")
                .input("input", "Frame")              // HOST (default)
                .cudaOutput("output", "Frame")        // CUDA_DEVICE
                .boolProp("sync", "Synchronize after copy", false, false)
                .selfManagedOutputPins()  // Creates output pin in addInputPin()
                .finalizeCuda([](const auto& props, cudastream_sp stream) {
                    CudaMemCopyProps moduleProps(cudaMemcpyHostToDevice, stream);
                    if (auto it = props.find("sync"); it != props.end()) {
                        if (auto* val = std::get_if<bool>(&it->second)) moduleProps.sync = *val;
                    }
                    return std::make_unique<CudaMemCopy>(moduleProps);
                });
        }

        // CudaMemCopyD2H - Device to Host memory transfer (BRIDGE MODULE)
        // Used by auto-bridging when CUDA_DEVICE → HOST transfer is needed
        if (!registry.hasModule("CudaMemCopyD2H")) {
            registerCudaModule<CudaMemCopy, CudaMemCopyProps>("CudaMemCopyD2H")
                .category(ModuleCategory::Utility)
                .description("Copies data from device (GPU) to host (CPU) memory.")
                .tags("utility", "memory", "cuda", "transfer", "bridge")
                .cudaInput("input", "Frame")          // CUDA_DEVICE
                .output("output", "Frame")            // HOST (default)
                .boolProp("sync", "Synchronize after copy", false, false)
                .selfManagedOutputPins()  // Creates output pin in addInputPin()
                .finalizeCuda([](const auto& props, cudastream_sp stream) {
                    CudaMemCopyProps moduleProps(cudaMemcpyDeviceToHost, stream);
                    if (auto it = props.find("sync"); it != props.end()) {
                        if (auto* val = std::get_if<bool>(&it->second)) moduleProps.sync = *val;
                    }
                    return std::make_unique<CudaMemCopy>(moduleProps);
                });
        }

        // CudaStreamSynchronize - CUDA stream synchronization
        if (!registry.hasModule("CudaStreamSynchronize")) {
            registerCudaModule<CudaStreamSynchronize, CudaStreamSynchronizeProps>("CudaStreamSynchronize")
                .category(ModuleCategory::Utility)
                .description("Synchronizes CUDA stream to ensure all operations complete.")
                .tags("utility", "sync", "cuda")
                .cudaInput("input", "Frame")
                .cudaOutput("output", "Frame")
                .selfManagedOutputPins()
                .finalizeCuda([](const auto& props, cudastream_sp stream) {
                    CudaStreamSynchronizeProps moduleProps(stream);
                    return std::make_unique<CudaStreamSynchronize>(moduleProps);
                });
        }

#ifndef ARM64
        // nvJPEG modules not available on ARM64/Jetson (nvJPEG library not supported)

        // JPEGDecoderNVJPEG - GPU-accelerated JPEG decoder
        // Input: HOST memory (encoded JPEG), Output: CUDA_DEVICE memory (decoded RawImage)
        if (!registry.hasModule("JPEGDecoderNVJPEG")) {
            registerCudaModule<JPEGDecoderNVJPEG, JPEGDecoderNVJPEGProps>("JPEGDecoderNVJPEG")
                .category(ModuleCategory::Transform)
                .description("Decodes JPEG images using NVIDIA nvJPEG library.")
                .tags("decoder", "jpeg", "image", "cuda", "nvjpeg")
                .input("input", "EncodedImage", FrameMetadata::HOST)
                .cudaOutput("output", "RawImage")
                .finalizeCuda([](const auto& props, cudastream_sp stream) {
                    JPEGDecoderNVJPEGProps moduleProps(stream);
                    return std::make_unique<JPEGDecoderNVJPEG>(moduleProps);
                });
        }

        // JPEGEncoderNVJPEG - GPU-accelerated JPEG encoder
        // Input: CUDA_DEVICE memory (RawImage), Output: HOST memory (encoded JPEG)
        if (!registry.hasModule("JPEGEncoderNVJPEG")) {
            registerCudaModule<JPEGEncoderNVJPEG, JPEGEncoderNVJPEGProps>("JPEGEncoderNVJPEG")
                .category(ModuleCategory::Transform)
                .description("Encodes images to JPEG using NVIDIA nvJPEG library.")
                .tags("encoder", "jpeg", "image", "cuda", "nvjpeg")
                .cudaInput("input", "RawImage")
                .output("output", "EncodedImage", FrameMetadata::HOST)
                .intProp("quality", "JPEG quality (1-100)", false, 90, 1, 100)
                .finalizeCuda([](const auto& props, cudastream_sp stream) {
                    JPEGEncoderNVJPEGProps moduleProps(stream);
                    if (auto it = props.find("quality"); it != props.end()) {
                        if (auto* val = std::get_if<int64_t>(&it->second)) {
                            moduleProps.quality = static_cast<unsigned short>(*val);
                        }
                    }
                    return std::make_unique<JPEGEncoderNVJPEG>(moduleProps);
                });
        }
#endif // !ARM64

        // MemTypeConversion - GPU memory type conversion (BRIDGE MODULE)
        // Note: memType is kept as HOST because actual I/O memType depends on "outputMemType" property.
        // This is a bridge module used by auto-bridging to convert between memory types.
        if (!registry.hasModule("MemTypeConversion")) {
            registerCudaModule<MemTypeConversion, MemTypeConversionProps>("MemTypeConversion")
                .category(ModuleCategory::Utility)
                .description("Converts frame data between HOST, DEVICE, and DMA memory types.")
                .tags("utility", "memory", "cuda", "conversion", "bridge")
                .input("input", "Frame")
                .output("output", "Frame")
                .enumProp("outputMemType", "Target memory type", true, "DEVICE",
                    "HOST", "DEVICE", "DMA")
                .finalizeCuda([](const auto& props, cudastream_sp stream) {
                    FrameMetadata::MemType memType = FrameMetadata::CUDA_DEVICE;
                    if (auto it = props.find("outputMemType"); it != props.end()) {
                        if (auto* val = std::get_if<std::string>(&it->second)) {
                            if (*val == "HOST") memType = FrameMetadata::HOST;
                            else if (*val == "DEVICE") memType = FrameMetadata::CUDA_DEVICE;
                            else if (*val == "DMA") memType = FrameMetadata::DMABUF;
                        }
                    }
                    MemTypeConversionProps moduleProps(memType, stream);
                    return std::make_unique<MemTypeConversion>(moduleProps);
                });
        }

        // CuCtxSynchronize - CUDA context synchronization
        // Note: This does not need a CUDA stream - it synchronizes the entire context
        // Typically used in CUDA pipelines so marked with CUDA_DEVICE memType
        if (!registry.hasModule("CuCtxSynchronize")) {
            registerModule<CuCtxSynchronize, CuCtxSynchronizeProps>()
                .category(ModuleCategory::Utility)
                .description("Synchronizes the CUDA context to ensure all GPU operations are complete.")
                .tags("utility", "sync", "cuda", "context")
                .cudaInput("input", "Frame")
                .cudaOutput("output", "Frame")
                .selfManagedOutputPins();
        }

#ifndef ARM64
        // H264EncoderNVCodec - GPU-accelerated H.264 encoder using NVCodec
        // Requires apracucontext_sp (CUDA Driver API context)
        // Not available on ARM64/Jetson (uses different encoder)
        if (!registry.hasModule("H264EncoderNVCodec")) {
            registerCuContextModule<H264EncoderNVCodec, H264EncoderNVCodecProps>("H264EncoderNVCodec")
                .category(ModuleCategory::Transform)
                .description("GPU-accelerated H.264 video encoder using NVIDIA NVCodec.")
                .tags("transform", "encode", "h264", "cuda", "nvcodec", "video")
                .cudaInput("input", "RawImagePlanar")
                .output("output", "H264Frame", FrameMetadata::HOST)
                .intProp("bitRateKbps", "Target bit rate in kilobits per second", false, 1000, 100, 50000)
                .intProp("gopLength", "Group of Pictures (GOP) length - frames between keyframes", false, 30, 1, 300)
                .intProp("frameRate", "Target frame rate", false, 30, 1, 120)
                .enumProp("profile", "H.264 codec profile", false, "BASELINE", "BASELINE", "MAIN", "HIGH")
                .boolProp("enableBFrames", "Enable B-frames for better compression (increases latency)", false, false)
                .intProp("bufferThres", "Buffer threshold for encoder", false, 30, 1, 100)
                .finalizeCuContext([](const auto& props, apracucontext_sp cuContext) {
                    // Extract properties from props map
                    uint32_t bitRateKbps = 1000;
                    uint32_t gopLength = 30;
                    uint32_t frameRate = 30;
                    H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::BASELINE;
                    bool enableBFrames = false;

                    if (auto it = props.find("bitRateKbps"); it != props.end()) {
                        if (auto* val = std::get_if<int64_t>(&it->second)) {
                            bitRateKbps = static_cast<uint32_t>(*val);
                        }
                    }
                    if (auto it = props.find("gopLength"); it != props.end()) {
                        if (auto* val = std::get_if<int64_t>(&it->second)) {
                            gopLength = static_cast<uint32_t>(*val);
                        }
                    }
                    if (auto it = props.find("frameRate"); it != props.end()) {
                        if (auto* val = std::get_if<int64_t>(&it->second)) {
                            frameRate = static_cast<uint32_t>(*val);
                        }
                    }
                    if (auto it = props.find("profile"); it != props.end()) {
                        if (auto* val = std::get_if<std::string>(&it->second)) {
                            if (*val == "MAIN") profile = H264EncoderNVCodecProps::MAIN;
                            else if (*val == "HIGH") profile = H264EncoderNVCodecProps::HIGH;
                            // Default BASELINE
                        }
                    }
                    if (auto it = props.find("enableBFrames"); it != props.end()) {
                        if (auto* val = std::get_if<bool>(&it->second)) {
                            enableBFrames = *val;
                        }
                    }

                    H264EncoderNVCodecProps moduleProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames);
                    return std::make_unique<H264EncoderNVCodec>(moduleProps);
                });
        }
#endif // !ARM64
#endif // ENABLE_CUDA
    }
}

} // namespace apra
