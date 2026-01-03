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
#include <optional>
#include <boost/shared_ptr.hpp>

#include "declarative/PipelineDescription.h"
#include "declarative/ModuleRegistry.h"
#include "declarative/Issue.h"
#include "PipeLine.h"

// Forward declaration
class Module;

namespace apra {

// ============================================================
// ModuleFactory - Builds PipeLine from PipelineDescription
// ============================================================
class ModuleFactory {
public:
    // Build options
    struct Options {
        bool auto_insert_converters;    // Future: auto-insert frame converters
        bool strict_mode;               // Fail on warnings in strict mode
        bool collect_info_messages;     // Include info messages in result

        Options() : auto_insert_converters(false), strict_mode(false),
                    collect_info_messages(false) {}
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

    // ============================================================
    // ModuleContext - Stores per-instance pin mappings
    // ============================================================
    struct ModuleContext {
        boost::shared_ptr<Module> module;
        std::string moduleType;                              // e.g., "FileReaderModule"
        std::string instanceId;                              // e.g., "source"
        std::map<std::string, std::string> outputPinMap;     // TOML name → internal pin ID
        std::map<std::string, std::string> inputPinMap;      // TOML name → internal pin ID
        std::vector<std::string> connectedInputs;            // Track which inputs are connected
    };

    // Constructor
    explicit ModuleFactory(Options opts = Options());

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

    // Set up output pins for a module based on registry info
    // Returns map of TOML pin name → internal pin ID
    // Uses outputFrameType property from instance if present to override default
    std::map<std::string, std::string> setupOutputPins(
        Module* module,
        const ModuleInfo& info,
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
    // Uses ModuleContext map for pin name resolution
    bool connectModules(
        const std::vector<Connection>& connections,
        std::map<std::string, ModuleContext>& contextMap,
        std::vector<BuildIssue>& issues
    );

    // Validate that all required inputs are connected
    // Called after connectModules() to check input satisfaction
    void validateInputConnections(
        const std::map<std::string, ModuleContext>& contextMap,
        std::vector<BuildIssue>& issues
    );

public:
    // Parse connection endpoint "instance.pin" into parts
    // Made public for testing
    static std::pair<std::string, std::string> parseConnectionEndpoint(
        const std::string& endpoint
    );

private:

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
