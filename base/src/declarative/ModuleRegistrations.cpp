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

// Conditionally include CUDA modules
#ifdef ENABLE_CUDA
#include "H264Decoder.h"
#endif

#include <mutex>

namespace apra {

void ensureBuiltinModulesRegistered() {
    // Check if modules are already registered.
    // Note: We can't use std::once_flag because tests may clear the registry,
    // requiring re-registration. Instead, check if registry has modules.
    auto& registry = ModuleRegistry::instance();

    if (registry.hasModule("FileReaderModule")) {
        // Already registered
        return;
    }

    // Register built-in modules
    {

        // ============================================================
        // Source Modules
        // ============================================================

        registerModule<FileReaderModule, FileReaderModuleProps>()
            .category(ModuleCategory::Source)
            .description("Reads frames from image/video files using file patterns")
            .tags("reader", "file", "source", "video", "image")
            .output("output", "EncodedImage", "RawImage");

        registerModule<TestSignalGenerator, TestSignalGeneratorProps>()
            .category(ModuleCategory::Source)
            .description("Generates test signal frames for testing pipelines")
            .tags("source", "test", "generator", "signal")
            .output("output", "RawImage");

        // ============================================================
        // Sink Modules
        // ============================================================

        registerModule<FileWriterModule, FileWriterModuleProps>()
            .category(ModuleCategory::Sink)
            .description("Writes frames to files with configurable patterns")
            .tags("writer", "file", "sink")
            .input("input", "RawImage", "EncodedImage", "H264Frame");

        registerModule<StatSink, StatSinkProps>()
            .category(ModuleCategory::Sink)
            .description("Statistics sink for measuring pipeline performance")
            .tags("sink", "stats", "performance", "debug")
            .input("input", "Frame");

        // ============================================================
        // Transform Modules
        // ============================================================

#ifdef ENABLE_CUDA
        registerModule<H264Decoder, H264DecoderProps>()
            .category(ModuleCategory::Transform)
            .description("Decodes H264/H265 video frames")
            .tags("decoder", "h264", "h265", "video", "transform")
            .input("encoded", "H264Frame")
            .output("decoded", "RawImagePlanar");
#endif

        // ============================================================
        // Analytics Modules
        // ============================================================

        registerModule<FaceDetectorXform, FaceDetectorXformProps>()
            .category(ModuleCategory::Analytics)
            .description("Detects faces in video frames using OpenCV cascade classifier")
            .tags("face", "detection", "analytics", "opencv")
            .input("input", "RawImage", "RawImagePlanar")
            .output("output", "RawImage", "RawImagePlanar");

        registerModule<QRReader, QRReaderProps>()
            .category(ModuleCategory::Analytics)
            .description("Reads QR codes and barcodes from frames")
            .tags("qr", "barcode", "reader", "analytics")
            .input("input", "RawImage")
            .output("output", "RawImage");

        // ============================================================
        // Utility Modules
        // ============================================================

        registerModule<ValveModule, ValveModuleProps>()
            .category(ModuleCategory::Utility)
            .description("Controls frame flow by allowing a specified number of frames to pass")
            .tags("utility", "valve", "control", "flow")
            .input("input", "Frame")
            .output("output", "Frame");

        // ============================================================
        // TODO: Add remaining modules here
        //
        // When adding a new module:
        // 1. #include the module header at the top
        // 2. Add registerModule<...>() call in appropriate category
        // 3. Run tests - AllConcreteModulesRegistered will verify
        // ============================================================

    }
}

} // namespace apra
