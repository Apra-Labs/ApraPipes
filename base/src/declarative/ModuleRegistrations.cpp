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

    // Register built-in modules that don't have REGISTER_MODULE yet
    {
        // Note: Most core modules now use REGISTER_MODULE macro
        // This section is for modules being migrated or special cases

        // TestSignalGenerator - might not have REGISTER_MODULE yet
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
    }
}

} // namespace apra
