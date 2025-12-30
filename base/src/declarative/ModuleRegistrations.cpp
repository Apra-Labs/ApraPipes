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

// Include module headers (always available)
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "FaceDetectorXform.h"
#include "QRReader.h"

// Conditionally include CUDA modules
#ifdef ENABLE_CUDA
#include "H264Decoder.h"
#endif

#include <mutex>

namespace apra {

void ensureBuiltinModulesRegistered() {
    static std::once_flag flag;
    std::call_once(flag, []() {

        // ============================================================
        // Source Modules
        // ============================================================

        registerModule<FileReaderModule, FileReaderModuleProps>()
            .category(ModuleCategory::Source)
            .description("Reads frames from image/video files using file patterns")
            .tags("reader", "file", "source", "video", "image")
            .output("output", "EncodedImage", "RawImage");

        // ============================================================
        // Sink Modules
        // ============================================================

        registerModule<FileWriterModule, FileWriterModuleProps>()
            .category(ModuleCategory::Sink)
            .description("Writes frames to files with configurable patterns")
            .tags("writer", "file", "sink")
            .input("input", "RawImage", "EncodedImage", "H264Frame");

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
        // TODO: Add remaining modules here
        //
        // When adding a new module:
        // 1. #include the module header at the top
        // 2. Add registerModule<...>() call in appropriate category
        // 3. Run tests - AllConcreteModulesRegistered will verify
        // ============================================================

    });
}

} // namespace apra
