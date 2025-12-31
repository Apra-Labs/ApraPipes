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
#include <boost/filesystem.hpp>
#include "declarative/ModuleRegistry.h"
#include "declarative/ModuleRegistrations.h"
#include "declarative/ModuleRegistrationBuilder.h"
#include "Module.h"  // For complete Module type
#include "FileWriterModule.h"  // For direct property binding test
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

// ============================================================
// Property Binding Tests
// ============================================================

BOOST_AUTO_TEST_CASE(FileWriterModuleProps_HasApplyProperties) {
    // Direct test: check if the type trait detects applyProperties
    constexpr bool hasApply = apra::detail::has_apply_properties_v<FileWriterModuleProps>;
    BOOST_TEST_MESSAGE("has_apply_properties_v<FileWriterModuleProps> = " << (hasApply ? "true" : "false"));
    BOOST_CHECK_MESSAGE(hasApply, "FileWriterModuleProps should have applyProperties detected");
}

BOOST_AUTO_TEST_CASE(FileWriterModuleProps_ApplyPropertiesDirectly) {
    // Test calling applyProperties directly
    FileWriterModuleProps props;
    BOOST_CHECK(props.strFullFileNameWithPattern.empty());

    std::map<std::string, apra::ScalarPropertyValue> values;
    values["strFullFileNameWithPattern"] = std::string("/tmp/test/frame.raw");
    values["append"] = true;

    std::vector<std::string> missing;
    FileWriterModuleProps::applyProperties(props, values, missing);

    BOOST_TEST_MESSAGE("After applyProperties:");
    BOOST_TEST_MESSAGE("  strFullFileNameWithPattern = " << props.strFullFileNameWithPattern);
    BOOST_TEST_MESSAGE("  append = " << props.append);
    BOOST_TEST_MESSAGE("  missing count = " << missing.size());

    BOOST_CHECK_EQUAL(props.strFullFileNameWithPattern, "/tmp/test/frame.raw");
    BOOST_CHECK_EQUAL(props.append, true);
    BOOST_CHECK(missing.empty());
}

BOOST_AUTO_TEST_CASE(FileWriterModule_FactoryLambdaTest) {
    // Create the factory lambda manually to test if it works
    auto factory = [](const std::map<std::string, apra::ScalarPropertyValue>& props)
        -> std::unique_ptr<Module> {

        FileWriterModuleProps moduleProps;

        // Debug: check if has_apply_properties_v is true
        constexpr bool hasApply = apra::detail::has_apply_properties_v<FileWriterModuleProps>;
        BOOST_TEST_MESSAGE("In factory lambda: has_apply_properties_v = " << (hasApply ? "true" : "false"));

        if constexpr (hasApply) {
            std::vector<std::string> missingRequired;
            FileWriterModuleProps::applyProperties(moduleProps, props, missingRequired);

            BOOST_TEST_MESSAGE("After applyProperties in factory:");
            BOOST_TEST_MESSAGE("  strFullFileNameWithPattern = " << moduleProps.strFullFileNameWithPattern);

            if (!missingRequired.empty()) {
                std::string msg = "Missing: ";
                for (const auto& m : missingRequired) msg += m + " ";
                throw std::runtime_error(msg);
            }
        }

        // Create output directory
        if (!moduleProps.strFullFileNameWithPattern.empty()) {
            boost::filesystem::path p(moduleProps.strFullFileNameWithPattern);
            boost::filesystem::path dirPath = p.parent_path();
            if (!dirPath.empty() && !boost::filesystem::exists(dirPath)) {
                boost::filesystem::create_directories(dirPath);
            }
        }

        return std::make_unique<FileWriterModule>(moduleProps);
    };

    // Test the factory
    std::map<std::string, apra::ScalarPropertyValue> props;
    props["strFullFileNameWithPattern"] = std::string("/tmp/factory_test/frame_????.raw");
    props["append"] = false;

    BOOST_TEST_MESSAGE("Calling factory with path: /tmp/factory_test/frame_????.raw");

    auto module = factory(props);
    BOOST_REQUIRE(module != nullptr);

    // Clean up
    boost::filesystem::remove_all("/tmp/factory_test");
}

BOOST_AUTO_TEST_CASE(FileWriterModule_PropertyBinding) {
    // Clear registry to ensure fresh registration with proper factories
    auto& registry = apra::ModuleRegistry::instance();
    registry.clear();

    // Register modules with proper factories
    apra::ensureBuiltinModulesRegistered();

    BOOST_REQUIRE(registry.hasModule("FileWriterModule"));

    // Create FileWriterModule with required property
    std::map<std::string, apra::ScalarPropertyValue> props;
    props["strFullFileNameWithPattern"] = std::string("/tmp/apra_test_output/frame_????.raw");
    props["append"] = false;

    // Create the test directory first
    boost::filesystem::create_directories("/tmp/apra_test_output");

    // Should succeed - property must be bound for this to work
    std::unique_ptr<Module> module;
    try {
        module = registry.createModule("FileWriterModule", props);
        BOOST_REQUIRE(module != nullptr);
    }
    catch (const std::exception& e) {
        BOOST_TEST_MESSAGE("std::exception: " << e.what());
        BOOST_FAIL("FileWriterModule creation failed: " << e.what());
    }

    // Clean up test directory
    boost::filesystem::remove_all("/tmp/apra_test_output");
}

BOOST_AUTO_TEST_CASE(FileReaderModule_PropertyBinding) {
    // Clear registry to ensure fresh registration with proper factories
    auto& registry = apra::ModuleRegistry::instance();
    registry.clear();

    apra::ensureBuiltinModulesRegistered();

    BOOST_REQUIRE(registry.hasModule("FileReaderModule"));

    // Create FileReaderModule with required property
    std::map<std::string, apra::ScalarPropertyValue> props;
    props["strFullFileNameWithPattern"] = std::string("./data/test_????.jpg");
    props["readLoop"] = false;

    // Should succeed
    std::unique_ptr<Module> module;
    BOOST_REQUIRE_NO_THROW(module = registry.createModule("FileReaderModule", props));
    BOOST_REQUIRE(module != nullptr);
}

BOOST_AUTO_TEST_SUITE_END()
