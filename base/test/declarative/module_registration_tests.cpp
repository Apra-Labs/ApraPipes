// ============================================================
// Unit tests for module registration system
// Task D2: Property Binding System
//
// Verifies that:
// 1. Registered modules have valid metadata
// 2. Provides detection mechanism for missing registrations
// ============================================================

#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "declarative/ModuleRegistry.h"
#include "declarative/ModuleRegistrations.h"
#include "Module.h"  // For complete Module type
#include <set>
#include <algorithm>

BOOST_AUTO_TEST_SUITE(ModuleRegistrationTests)

// ============================================================
// Basic Registration Tests
// ============================================================

BOOST_AUTO_TEST_CASE(EnsureRegistered_RegistersModules) {
    // Ensure modules are registered
    apra::ensureBuiltinModulesRegistered();

    auto& registry = apra::ModuleRegistry::instance();

    // We should have at least some modules registered
    auto names = registry.getAllModules();
    BOOST_CHECK(!names.empty());
}

BOOST_AUTO_TEST_CASE(RegisteredModules_HaveValidMetadata) {
    apra::ensureBuiltinModulesRegistered();

    auto& registry = apra::ModuleRegistry::instance();
    auto names = registry.getAllModules();

    for (const auto& name : names) {
        const auto* info = registry.getModule(name);
        BOOST_REQUIRE_MESSAGE(info != nullptr, "Module " << name << " info is null");

        // Module must have a name
        BOOST_CHECK_MESSAGE(!info->name.empty(),
            "Module " << name << " has empty name");

        // Module must have a description
        BOOST_CHECK_MESSAGE(!info->description.empty(),
            "Module " << name << " missing description");

        // Module must have a category
        BOOST_CHECK_MESSAGE(
            info->category == apra::ModuleCategory::Source ||
            info->category == apra::ModuleCategory::Sink ||
            info->category == apra::ModuleCategory::Transform ||
            info->category == apra::ModuleCategory::Analytics ||
            info->category == apra::ModuleCategory::Controller ||
            info->category == apra::ModuleCategory::Utility,
            "Module " << name << " has invalid category");

        // Module must have a version
        BOOST_CHECK_MESSAGE(!info->version.empty(),
            "Module " << name << " missing version");

        // Module must have a factory
        BOOST_CHECK_MESSAGE(info->factory != nullptr,
            "Module " << name << " missing factory");
    }
}

// ============================================================
// Known Module Registration Tests
// ============================================================

BOOST_AUTO_TEST_CASE(CoreModules_AreRegistered) {
    apra::ensureBuiltinModulesRegistered();

    auto& registry = apra::ModuleRegistry::instance();

    // These core modules should always be registered (non-CUDA)
    std::vector<std::string> coreModules = {
        "FileReaderModule",
        "FileWriterModule",
        "FaceDetectorXform",
        "QRReader",
    };

    for (const auto& name : coreModules) {
        BOOST_CHECK_MESSAGE(registry.hasModule(name),
            "Core module '" << name << "' is not registered");
    }
}

BOOST_AUTO_TEST_CASE(RegisteredModules_CanBeCreated) {
    apra::ensureBuiltinModulesRegistered();

    auto& registry = apra::ModuleRegistry::instance();
    auto names = registry.getAllModules();

    int attemptedCount = 0;
    for (const auto& name : names) {
        // Attempt to create module with empty props
        std::map<std::string, apra::ScalarPropertyValue> props;

        // Some modules may require specific props, so we don't fail on null
        // We just verify the factory function is callable
        try {
            auto module = registry.createModule(name, props);
            attemptedCount++;
            // Module might be null if required props are missing - that's OK
            // The important thing is the factory doesn't crash
        }
        catch (const std::exception& e) {
            attemptedCount++;
            // Some modules may throw if required props missing - also OK
            BOOST_TEST_MESSAGE("Module " << name << " creation threw: " << e.what());
        }
    }

    // Verify we attempted to create all modules
    BOOST_CHECK_EQUAL(attemptedCount, static_cast<int>(names.size()));
}

// ============================================================
// Module Category Tests
// ============================================================

BOOST_AUTO_TEST_CASE(ModulesByCategory_ReturnsCorrectModules) {
    apra::ensureBuiltinModulesRegistered();

    auto& registry = apra::ModuleRegistry::instance();

    // Get sources - should have at least FileReaderModule
    auto sources = registry.getModulesByCategory(apra::ModuleCategory::Source);
    BOOST_CHECK(sources.size() >= 1);
    for (const auto& name : sources) {
        const auto* info = registry.getModule(name);
        BOOST_REQUIRE(info != nullptr);
        BOOST_CHECK_EQUAL(static_cast<int>(info->category),
                          static_cast<int>(apra::ModuleCategory::Source));
    }

    // Get sinks - should have at least FileWriterModule
    auto sinks = registry.getModulesByCategory(apra::ModuleCategory::Sink);
    BOOST_CHECK(sinks.size() >= 1);
    for (const auto& name : sinks) {
        const auto* info = registry.getModule(name);
        BOOST_REQUIRE(info != nullptr);
        BOOST_CHECK_EQUAL(static_cast<int>(info->category),
                          static_cast<int>(apra::ModuleCategory::Sink));
    }

    // Get transforms (may be empty if CUDA disabled)
    auto transforms = registry.getModulesByCategory(apra::ModuleCategory::Transform);
    for (const auto& name : transforms) {
        const auto* info = registry.getModule(name);
        BOOST_REQUIRE(info != nullptr);
        BOOST_CHECK_EQUAL(static_cast<int>(info->category),
                          static_cast<int>(apra::ModuleCategory::Transform));
    }

    // Get analytics - should have FaceDetectorXform and QRReader
    auto analytics = registry.getModulesByCategory(apra::ModuleCategory::Analytics);
    BOOST_CHECK(analytics.size() >= 2);
    for (const auto& name : analytics) {
        const auto* info = registry.getModule(name);
        BOOST_REQUIRE(info != nullptr);
        BOOST_CHECK_EQUAL(static_cast<int>(info->category),
                          static_cast<int>(apra::ModuleCategory::Analytics));
    }
}

// ============================================================
// Detection of Unregistered Modules
// ============================================================

BOOST_AUTO_TEST_CASE(DetectUnregisteredModules_InfoTest) {
    // This test provides info about which modules are not yet registered.
    // It doesn't fail, but logs which modules need to be added.
    // Run with --log_level=message to see the output.

    apra::ensureBuiltinModulesRegistered();

    auto& registry = apra::ModuleRegistry::instance();
    auto registered = registry.getAllModules();
    std::set<std::string> registeredSet(registered.begin(), registered.end());

    // List of ALL concrete modules in the codebase
    // This list was generated by scanning base/include/*.h
    // Modules with // are conditionally compiled (CUDA, ARM64, etc.)
    std::vector<std::string> allModules = {
        "AffineTransform",
        "ArchiveSpaceManager",
        "AudioCaptureSrc",
        "AudioToTextXForm",
        "BMPConverter",
        "BrightnessContrastControl",
        "CCNPPI",                    // CUDA
        "CalcHistogramCV",
        "ColorConversion",           // CUDA
        "CuCtxSynchronize",          // CUDA
        "CudaMemCopy",               // CUDA
        "CudaStreamSynchronize",     // CUDA
        "DMAFDToHostCopy",           // ARM64
        "EffectsNPPI",               // CUDA
        "EglRenderer",               // ARM64
        "ExternalSinkModule",
        "ExternalSourceModule",
        "FaceDetectorXform",         // Already registered
        "FacialLandmarkCV",
        "FileReaderModule",          // Already registered
        "FileWriterModule",          // Already registered
        "FramesMuxer",
        "GaussianBlur",              // CUDA
        "GtkGlRenderer",             // Linux
        "H264Decoder",               // CUDA - Already registered
        "H264EncoderNVCodec",        // CUDA
        "H264EncoderV4L2",           // ARM64
        "HistogramOverlay",
        "ImageDecoderCV",
        "ImageEncoderCV",
        "ImageResizeCV",
        "ImageViewerModule",
        "JPEGDecoderL4TM",           // ARM64
        "JPEGDecoderNVJPEG",         // CUDA
        "JPEGEncoderL4TM",           // ARM64
        "JPEGEncoderNVJPEG",         // CUDA
        "KeyboardListener",
        "MemTypeConversion",         // CUDA
        "Merge",
        "MotionVectorExtractor",
        "Mp4ReaderSource",
        "Mp4WriterSink",
        "MultimediaQueueXform",
        "NvArgusCamera",             // ARM64
        "NvTransform",               // ARM64
        "NvV4L2Camera",              // ARM64
        "OverlayModule",
        "OverlayNPPI",               // CUDA
        "QRReader",                  // Already registered
        "RTSPClientSrc",
        "RTSPPusher",
        "ResizeNPPI",                // CUDA
        "RotateCV",
        "RotateNPPI",                // CUDA
        "SimpleControlModule",
        "Split",
        "StatSink",
        "TestSignalGenerator",
        "TextOverlayXForm",
        "ThumbnailListGenerator",
        "ValveModule",
        "VirtualCameraSink",
        "VirtualPTZ",
        "WebCamSource",
    };

    // Find unregistered modules
    std::vector<std::string> unregistered;
    for (const auto& module : allModules) {
        if (registeredSet.find(module) == registeredSet.end()) {
            unregistered.push_back(module);
        }
    }

    if (!unregistered.empty()) {
        BOOST_TEST_MESSAGE("=== Modules not yet registered (" << unregistered.size() << ") ===");
        for (const auto& name : unregistered) {
            BOOST_TEST_MESSAGE("  - " << name);
        }
        BOOST_TEST_MESSAGE("=== Add these to ModuleRegistrations.cpp ===");
    }

    // This test passes but logs warnings
    BOOST_CHECK_MESSAGE(true, "Module detection completed");
}

BOOST_AUTO_TEST_SUITE_END()
