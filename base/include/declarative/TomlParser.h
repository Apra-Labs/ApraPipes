#pragma once

#include "PipelineDescription.h"
#include <string>
#include <vector>

namespace apra {

// Result of parsing a pipeline configuration
struct ParseResult {
    bool success = false;
    PipelineDescription description;

    // Error information (populated when success == false)
    std::string error;
    int error_line = 0;
    int error_column = 0;
};

// Abstract base class for pipeline parsers
class PipelineParser {
public:
    virtual ~PipelineParser() = default;

    // Parse from file path
    virtual ParseResult parseFile(const std::string& filepath) = 0;

    // Parse from string content
    virtual ParseResult parseString(const std::string& content,
                                    const std::string& source_name = "<inline>") = 0;

    // Get the format name (e.g., "toml", "yaml", "json")
    virtual std::string formatName() const = 0;

    // Get supported file extensions
    virtual std::vector<std::string> fileExtensions() const = 0;
};

// TOML parser implementation
class TomlParser : public PipelineParser {
public:
    ParseResult parseFile(const std::string& filepath) override;
    ParseResult parseString(const std::string& content,
                           const std::string& source_name = "<inline>") override;

    std::string formatName() const override { return "toml"; }
    std::vector<std::string> fileExtensions() const override {
        return {".toml"};
    }
};

} // namespace apra
