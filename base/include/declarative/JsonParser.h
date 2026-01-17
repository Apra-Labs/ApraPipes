// ============================================================
// File: declarative/JsonParser.h
// JSON Parser for Pipeline Descriptions
// Task J1: JSON Parser Implementation
// ============================================================

#pragma once

#include "ParseResult.h"
#include <string>
#include <vector>

namespace apra {

// ============================================================
// JsonParser - JSON parser for pipeline descriptions
// ============================================================
class JsonParser {
public:
    // Parse from file path
    static ParseResult parseFile(const std::string& filepath);

    // Parse from string content
    static ParseResult parseString(const std::string& content,
                                   const std::string& source_name = "<inline>");

    // Get the format name
    static std::string formatName() { return "json"; }

    // Get supported file extensions
    static std::vector<std::string> fileExtensions() {
        return {".json"};
    }
};

} // namespace apra
