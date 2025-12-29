// ============================================================
// File: declarative/PipelineValidator.h
// Pipeline Validator Shell (Framework)
// Task C1: Validator Shell
// ============================================================

#pragma once

#include "declarative/PipelineDescription.h"
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

namespace apra {

// ============================================================
// ValidationIssue - Represents a validation problem or info
// ============================================================
struct ValidationIssue {
    enum class Level {
        Error,      // Fatal - pipeline cannot be built
        Warning,    // Non-fatal - pipeline may work but has issues
        Info        // Informational - for logging
    };

    Level level = Level::Error;
    std::string code;           // "E001", "W010", "I001"
    std::string location;       // "modules.decoder.props.device_id"
    std::string message;        // Human-readable description
    std::string suggestion;     // Optional fix suggestion

    // Factory methods
    static ValidationIssue error(const std::string& code,
                                 const std::string& location,
                                 const std::string& message,
                                 const std::string& suggestion = "") {
        return {Level::Error, code, location, message, suggestion};
    }

    static ValidationIssue warning(const std::string& code,
                                   const std::string& location,
                                   const std::string& message,
                                   const std::string& suggestion = "") {
        return {Level::Warning, code, location, message, suggestion};
    }

    static ValidationIssue info(const std::string& code,
                                const std::string& location,
                                const std::string& message) {
        return {Level::Info, code, location, message, ""};
    }

    // Error codes for future validation phases
    // Module validation (C2)
    static constexpr const char* UNKNOWN_MODULE_TYPE = "E100";
    static constexpr const char* MODULE_VERSION_MISMATCH = "W100";

    // Property validation (C3)
    static constexpr const char* UNKNOWN_PROPERTY = "E200";
    static constexpr const char* PROPERTY_TYPE_MISMATCH = "E201";
    static constexpr const char* PROPERTY_OUT_OF_RANGE = "E202";
    static constexpr const char* PROPERTY_INVALID_ENUM = "E203";
    static constexpr const char* PROPERTY_REGEX_MISMATCH = "E204";
    static constexpr const char* MISSING_REQUIRED_PROPERTY = "W200";

    // Connection validation (C4)
    static constexpr const char* UNKNOWN_SOURCE_MODULE = "E300";
    static constexpr const char* UNKNOWN_DEST_MODULE = "E301";
    static constexpr const char* UNKNOWN_SOURCE_PIN = "E302";
    static constexpr const char* UNKNOWN_DEST_PIN = "E303";
    static constexpr const char* FRAME_TYPE_INCOMPATIBLE = "E304";
    static constexpr const char* DUPLICATE_INPUT_CONNECTION = "E305";
    static constexpr const char* REQUIRED_PIN_UNCONNECTED = "W300";

    // Graph validation (C5)
    static constexpr const char* NO_SOURCE_MODULE = "E400";
    static constexpr const char* GRAPH_HAS_CYCLE = "E401";
    static constexpr const char* ORPHAN_MODULE = "W400";
};

// ============================================================
// PipelineValidator - Validates PipelineDescription
// ============================================================
class PipelineValidator {
public:
    // Validation result
    struct Result {
        std::vector<ValidationIssue> issues;

        bool hasErrors() const {
            return std::any_of(issues.begin(), issues.end(),
                [](const ValidationIssue& i) { return i.level == ValidationIssue::Level::Error; });
        }

        bool hasWarnings() const {
            return std::any_of(issues.begin(), issues.end(),
                [](const ValidationIssue& i) { return i.level == ValidationIssue::Level::Warning; });
        }

        std::vector<ValidationIssue> errors() const {
            std::vector<ValidationIssue> result;
            std::copy_if(issues.begin(), issues.end(), std::back_inserter(result),
                [](const ValidationIssue& i) { return i.level == ValidationIssue::Level::Error; });
            return result;
        }

        std::vector<ValidationIssue> warnings() const {
            std::vector<ValidationIssue> result;
            std::copy_if(issues.begin(), issues.end(), std::back_inserter(result),
                [](const ValidationIssue& i) { return i.level == ValidationIssue::Level::Warning; });
            return result;
        }

        // Merge results from multiple phases
        void merge(const Result& other) {
            issues.insert(issues.end(), other.issues.begin(), other.issues.end());
        }

        // Format for display
        std::string format() const;
    };

    // Validation options
    struct Options {
        bool stopOnFirstError;          // Stop validation on first error
        bool includeInfoMessages;       // Include info-level messages
        bool validateConnections;       // Run connection validation
        bool validateGraph;             // Run graph validation

        Options() : stopOnFirstError(false), includeInfoMessages(false),
                    validateConnections(true), validateGraph(true) {}
    };

    // Constructor
    explicit PipelineValidator(Options opts = Options());

    // Main entry point - runs all validation phases
    Result validate(const PipelineDescription& desc) const;

    // Individual phases (for tooling that wants partial validation)
    Result validateModules(const PipelineDescription& desc) const;
    Result validateProperties(const PipelineDescription& desc) const;
    Result validateConnections(const PipelineDescription& desc) const;
    Result validateGraph(const PipelineDescription& desc) const;

    // Get/set options
    const Options& options() const { return options_; }
    void setOptions(Options opts) { options_ = std::move(opts); }

private:
    Options options_;
};

} // namespace apra
