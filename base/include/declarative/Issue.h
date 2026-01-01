// ============================================================
// File: declarative/Issue.h
// Common Issue struct for validation and build phases
// ============================================================

#pragma once

#include <string>
#include <vector>
#include <algorithm>

namespace apra {

// ============================================================
// Issue - Represents a validation/build problem or info message
// Used by both PipelineValidator and ModuleFactory
// ============================================================
struct Issue {
    enum class Level {
        Error,      // Fatal - stops validation/build
        Warning,    // Non-fatal - continues with warning
        Info        // Informational - for logging
    };

    Level level = Level::Error;
    std::string code;           // e.g., "E001", "W002", "I001"
    std::string location;       // e.g., "modules.decoder.props.device_id"
    std::string message;        // Human-readable description
    std::string suggestion;     // Optional fix suggestion with TOML snippet

    // Factory methods for common issues
    static Issue error(const std::string& code,
                      const std::string& location,
                      const std::string& message,
                      const std::string& suggestion = "") {
        return {Level::Error, code, location, message, suggestion};
    }

    static Issue warning(const std::string& code,
                        const std::string& location,
                        const std::string& message,
                        const std::string& suggestion = "") {
        return {Level::Warning, code, location, message, suggestion};
    }

    static Issue info(const std::string& code,
                     const std::string& location,
                     const std::string& message) {
        return {Level::Info, code, location, message, ""};
    }

    // ============================================================
    // Error codes - organized by phase
    // ============================================================

    // Module validation (E1xx)
    static constexpr const char* UNKNOWN_MODULE = "E100";
    static constexpr const char* MODULE_VERSION_MISMATCH = "W100";
    static constexpr const char* MODULE_CREATION_FAILED = "E101";

    // Property validation (E2xx)
    static constexpr const char* UNKNOWN_PROPERTY = "E200";
    static constexpr const char* PROPERTY_TYPE_MISMATCH = "E201";
    static constexpr const char* PROPERTY_OUT_OF_RANGE = "E202";
    static constexpr const char* PROPERTY_INVALID_ENUM = "E203";
    static constexpr const char* PROPERTY_REGEX_MISMATCH = "E204";
    static constexpr const char* MISSING_REQUIRED_PROPERTY = "E205";
    static constexpr const char* PROP_TYPE_CONVERSION = "W201";

    // Connection validation (E3xx)
    static constexpr const char* UNKNOWN_SOURCE_MODULE = "E300";
    static constexpr const char* UNKNOWN_DEST_MODULE = "E301";
    static constexpr const char* UNKNOWN_SOURCE_PIN = "E302";
    static constexpr const char* UNKNOWN_DEST_PIN = "E303";
    static constexpr const char* FRAME_TYPE_INCOMPATIBLE = "E304";
    static constexpr const char* DUPLICATE_INPUT_CONNECTION = "E305";
    static constexpr const char* CONNECTION_FAILED = "E306";
    static constexpr const char* MISSING_REQUIRED_INPUT = "E307";
    static constexpr const char* REQUIRED_PIN_UNCONNECTED = "W300";

    // Graph validation (E4xx)
    static constexpr const char* NO_SOURCE_MODULE = "E400";
    static constexpr const char* GRAPH_HAS_CYCLE = "E401";
    static constexpr const char* EMPTY_PIPELINE = "E402";
    static constexpr const char* ORPHAN_MODULE = "W400";

    // Build/Runtime (E5xx)
    static constexpr const char* INIT_FAILED = "E500";
    static constexpr const char* RUN_FAILED = "E501";

    // Info messages (I0xx)
    static constexpr const char* INFO_VALIDATING = "I000";
    static constexpr const char* INFO_MODULE_FOUND = "I010";
    static constexpr const char* INFO_MODULE_CREATED = "I011";
    static constexpr const char* INFO_CONNECTION_ESTABLISHED = "I020";

    // Aliases for backwards compatibility
    static constexpr const char* MODULE_CREATED = "I011";
    static constexpr const char* CONNECTION_ESTABLISHED = "I020";
    static constexpr const char* MISSING_REQUIRED_PROP = "E205";  // Alias for MISSING_REQUIRED_PROPERTY
    static constexpr const char* UNKNOWN_MODULE_TYPE = "E100";    // Alias for UNKNOWN_MODULE
};

// Type alias for backwards compatibility
using ValidationIssue = Issue;
using BuildIssue = Issue;

} // namespace apra
