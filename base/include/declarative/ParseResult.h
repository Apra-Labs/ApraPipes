// ============================================================
// File: declarative/ParseResult.h
// Common parse result type for all pipeline parsers
// ============================================================

#pragma once

#include "PipelineDescription.h"
#include <string>

namespace apra {

// ============================================================
// ParseResult - Result of parsing a pipeline configuration
// Used by JsonParser for pipeline configuration parsing
// ============================================================
struct ParseResult {
    bool success = false;
    PipelineDescription description;

    // Error information (populated when success == false)
    std::string error;
    int error_line = 0;
    int error_column = 0;
};

} // namespace apra
