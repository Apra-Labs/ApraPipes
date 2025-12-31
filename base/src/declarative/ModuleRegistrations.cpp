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

// Conditionally include CUDA modules
#ifdef ENABLE_CUDA
#include "H264Decoder.h"
#endif

#include <mutex>

namespace apra {

void ensureBuiltinModulesRegistered() {
    // Check if modules are already registered.
    auto& registry = ModuleRegistry::instance();

    if (registry.hasModule("FileReaderModule")) {
        // Already registered (via REGISTER_MODULE macro in .cpp files)
        return;
    }

    // If registry is empty (e.g., after clear() in tests), re-run all
    // registered callbacks. This allows modules registered via REGISTER_MODULE
    // to be re-registered with their full property metadata.
    registry.rerunRegistrations();

    // If modules are still not registered (e.g., static init hasn't run yet),
    // register fallback modules without full property metadata.
    // This is a safety net for edge cases like dynamic loading.

    // Register all built-in modules using central registration pattern
    // (Wave 2 approach: metadata here, only applyProperties in module headers)
    {
        // FileReaderModule - reads frames from files
        if (!registry.hasModule("FileReaderModule")) {
            registerModule<FileReaderModule, FileReaderModuleProps>()
                .category(ModuleCategory::Source)
                .description("Reads frames from files matching a pattern. Supports image sequences and raw frame files.")
                .tags("source", "file", "reader")
                .output("output", "Frame");
        }

        // FileWriterModule - writes frames to files
        if (!registry.hasModule("FileWriterModule")) {
            registerModule<FileWriterModule, FileWriterModuleProps>()
                .category(ModuleCategory::Sink)
                .description("Writes frames to files. Supports file sequences with pattern-based naming.")
                .tags("sink", "file", "writer")
                .input("input", "Frame");
        }

        // FaceDetectorXform - face detection
        if (!registry.hasModule("FaceDetectorXform")) {
            registerModule<FaceDetectorXform, FaceDetectorXformProps>()
                .category(ModuleCategory::Analytics)
                .description("Detects faces in image frames using deep learning models.")
                .tags("analytics", "face", "detection", "transform")
                .input("input", "RawImagePlanar")
                .output("output", "Frame");
        }

        // QRReader - QR code and barcode detection
        if (!registry.hasModule("QRReader")) {
            registerModule<QRReader, QRReaderProps>()
                .category(ModuleCategory::Analytics)
                .description("Reads and decodes QR codes and barcodes from image frames.")
                .tags("analytics", "qr", "barcode", "reader")
                .input("input", "RawImagePlanar")
                .output("output", "Frame");
        }

        // TestSignalGenerator - generates test frames
        if (!registry.hasModule("TestSignalGenerator")) {
            registerModule<TestSignalGenerator, TestSignalGeneratorProps>()
                .category(ModuleCategory::Source)
                .description("Generates test signal frames for testing pipelines")
                .tags("source", "test", "generator", "signal")
                .output("output", "RawImage");
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
                .output("output_2", "Frame");
        }

        // Merge - merges frames from multiple inputs
        if (!registry.hasModule("Merge")) {
            registerModule<Merge, MergeProps>()
                .category(ModuleCategory::Utility)
                .description("Merges frames from multiple input pins")
                .tags("utility", "merge", "sync")
                .input("input_1", "Frame")
                .input("input_2", "Frame")
                .output("output", "Frame");
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
                .output("output", "EncodedImage");
        }

        // Mp4ReaderSource - reads MP4 video files
        if (!registry.hasModule("Mp4ReaderSource")) {
            registerModule<Mp4ReaderSource, Mp4ReaderSourceProps>()
                .category(ModuleCategory::Source)
                .description("Reads video frames from MP4 files")
                .tags("source", "mp4", "video", "file")
                .output("output", "H264Data", "EncodedImage");
        }

        // Mp4WriterSink - writes MP4 video files
        if (!registry.hasModule("Mp4WriterSink")) {
            registerModule<Mp4WriterSink, Mp4WriterSinkProps>()
                .category(ModuleCategory::Sink)
                .description("Writes video frames to MP4 files")
                .tags("sink", "mp4", "video", "file")
                .input("input", "H264Data", "EncodedImage");
        }

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
