// ============================================================
// File: declarative/ModuleFactory.h
// Module Factory for building PipeLine from PipelineDescription
// Task D1: Module Factory
// ============================================================

#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <boost/shared_ptr.hpp>

#include "declarative/PipelineDescription.h"
#include "declarative/ModuleRegistry.h"
#include "PipeLine.h"

// Forward declaration
class Module;

namespace apra {

// ============================================================
// BuildIssue - Represents an issue encountered during build
// ============================================================
struct BuildIssue {
    enum class Level {
        Error,      // Fatal - stops build
        Warning,    // Non-fatal - build continues
        Info        // Informational - for logging
    };

    Level level = Level::Error;
    std::string code;           // e.g., "E001", "W002", "I001"
    std::string location;       // e.g., "modules.decoder.props.device_id"
    std::string message;        // Human-readable description

    // Factory methods for common issues
    static BuildIssue error(const std::string& code,
                           const std::string& location,
                           const std::string& message) {
        return {Level::Error, code, location, message};
    }

    static BuildIssue warning(const std::string& code,
                             const std::string& location,
                             const std::string& message) {
        return {Level::Warning, code, location, message};
    }

    static BuildIssue info(const std::string& code,
                          const std::string& location,
                          const std::string& message) {
        return {Level::Info, code, location, message};
    }

    // Common error codes
    static constexpr const char* UNKNOWN_MODULE = "E001";
    static constexpr const char* MODULE_CREATION_FAILED = "E002";
    static constexpr const char* UNKNOWN_PROPERTY = "E010";
    static constexpr const char* PROPERTY_TYPE_MISMATCH = "E011";
    static constexpr const char* UNKNOWN_SOURCE_MODULE = "E020";
    static constexpr const char* UNKNOWN_SOURCE_PIN = "E021";
    static constexpr const char* UNKNOWN_DEST_MODULE = "E022";
    static constexpr const char* UNKNOWN_DEST_PIN = "E023";
    static constexpr const char* CONNECTION_FAILED = "E025";
    static constexpr const char* EMPTY_PIPELINE = "E030";
    static constexpr const char* INIT_FAILED = "E031";
    static constexpr const char* MISSING_REQUIRED_PROP = "E012";
    static constexpr const char* PROP_TYPE_CONVERSION = "W002";
    static constexpr const char* MODULE_CREATED = "I001";
    static constexpr const char* CONNECTION_ESTABLISHED = "I002";
};

// ============================================================
// ModuleFactory - Builds PipeLine from PipelineDescription
// ============================================================
class ModuleFactory {
public:
    // Build options
    struct Options {
        bool auto_insert_converters = false;  // Future: auto-insert frame converters
        bool strict_mode = false;              // Fail on warnings in strict mode
        bool collect_info_messages = false;    // Include info messages in result

        Options() = default;  // Explicit default constructor for C++ compatibility
    };

    // Build result
    struct BuildResult {
        std::unique_ptr<PipeLine> pipeline;
        std::vector<BuildIssue> issues;

        // Check if build succeeded (no errors)
        bool success() const {
            if (!pipeline) return false;
            return !std::any_of(issues.begin(), issues.end(),
                [](const BuildIssue& i) { return i.level == BuildIssue::Level::Error; });
        }

        // Convenience methods
        bool hasErrors() const {
            return std::any_of(issues.begin(), issues.end(),
                [](const BuildIssue& i) { return i.level == BuildIssue::Level::Error; });
        }

        bool hasWarnings() const {
            return std::any_of(issues.begin(), issues.end(),
                [](const BuildIssue& i) { return i.level == BuildIssue::Level::Warning; });
        }

        // Get issues filtered by level
        std::vector<BuildIssue> getErrors() const {
            std::vector<BuildIssue> result;
            std::copy_if(issues.begin(), issues.end(), std::back_inserter(result),
                [](const BuildIssue& i) { return i.level == BuildIssue::Level::Error; });
            return result;
        }

        std::vector<BuildIssue> getWarnings() const {
            std::vector<BuildIssue> result;
            std::copy_if(issues.begin(), issues.end(), std::back_inserter(result),
                [](const BuildIssue& i) { return i.level == BuildIssue::Level::Warning; });
            return result;
        }

        // Format all issues as string for display
        std::string formatIssues() const;
    };

    // Constructor
    explicit ModuleFactory(Options opts = {});

    // Main build method - takes PipelineDescription, returns running pipeline
    BuildResult build(const PipelineDescription& desc);

    // Get current options
    const Options& options() const { return options_; }

    // Set options
    void setOptions(Options opts) { options_ = std::move(opts); }

private:
    Options options_;

    // Internal helpers

    // Create a single module from instance description
    boost::shared_ptr<Module> createModule(
        const ModuleInstance& instance,
        std::vector<BuildIssue>& issues
    );

    // Apply properties from description to module
    void applyProperties(
        Module* module,
        const ModuleInstance& instance,
        const ModuleInfo* info,
        std::vector<BuildIssue>& issues
    );

    // Connect all modules according to connections list
    bool connectModules(
        const std::vector<Connection>& connections,
        const std::map<std::string, boost::shared_ptr<Module>>& moduleMap,
        std::vector<BuildIssue>& issues
    );

    // Convert PropertyValue from PipelineDescription to ScalarPropertyValue
    // (handles array types by extracting first element or using default)
    std::optional<ScalarPropertyValue> convertPropertyValue(
        const PropertyValue& value,
        const ModuleInfo::PropInfo& propInfo,
        std::vector<BuildIssue>& issues,
        const std::string& location
    );
};

} // namespace apra
