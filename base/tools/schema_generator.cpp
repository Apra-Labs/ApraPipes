// ============================================================
// File: tools/schema_generator.cpp
// Schema Generator Tool for declarative pipeline
// Task E2: Schema Generator
//
// Exports ModuleRegistry and FrameTypeRegistry to JSON/Markdown
// for LLM integration, documentation, and IDE tooling.
//
// Usage:
//   apra_schema_generator --modules-json modules.json
//   apra_schema_generator --frame-types-json frame_types.json
//   apra_schema_generator --modules-md MODULES.md
//   apra_schema_generator --all --output-dir ./schema
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <nlohmann/json.hpp>

#include "declarative/ModuleRegistry.h"
#include "declarative/FrameTypeRegistry.h"

using json = nlohmann::json;
using namespace apra;

// ============================================================
// JSON Generation Functions
// ============================================================

std::string getCurrentTimestamp() {
    auto now = std::time(nullptr);
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

json generateModulesJson() {
    json root;
    root["version"] = "1.0";
    root["generated"] = getCurrentTimestamp();
    root["generator"] = "apra_schema_generator";

    auto& registry = ModuleRegistry::instance();
    const auto allModules = registry.getAllModules();

    root["module_count"] = allModules.size();
    root["modules"] = json::object();

    for (const auto& name : allModules) {
        const auto* info = registry.getModule(name);
        if (!info) continue;

        json module;
        module["category"] = detail::categoryToString(info->category);
        module["version"] = info->version;
        module["description"] = info->description;
        module["tags"] = info->tags;

        // Inputs
        json inputs = json::array();
        for (const auto& pin : info->inputs) {
            json pinJson;
            pinJson["name"] = pin.name;
            pinJson["frame_types"] = pin.frame_types;
            pinJson["required"] = pin.required;
            pinJson["description"] = pin.description;
            inputs.push_back(pinJson);
        }
        module["inputs"] = inputs;

        // Outputs
        json outputs = json::array();
        for (const auto& pin : info->outputs) {
            json pinJson;
            pinJson["name"] = pin.name;
            pinJson["frame_types"] = pin.frame_types;
            pinJson["required"] = pin.required;
            pinJson["description"] = pin.description;
            outputs.push_back(pinJson);
        }
        module["outputs"] = outputs;

        // Properties
        json properties = json::object();
        for (const auto& prop : info->properties) {
            json propJson;
            propJson["type"] = prop.type;
            propJson["mutability"] = prop.mutability;
            propJson["description"] = prop.description;

            if (!prop.default_value.empty()) {
                propJson["default"] = prop.default_value;
            }
            if (!prop.min_value.empty()) {
                propJson["min"] = prop.min_value;
            }
            if (!prop.max_value.empty()) {
                propJson["max"] = prop.max_value;
            }
            if (!prop.regex_pattern.empty()) {
                propJson["regex"] = prop.regex_pattern;
            }
            if (!prop.enum_values.empty()) {
                propJson["enum_values"] = prop.enum_values;
            }
            if (!prop.unit.empty()) {
                propJson["unit"] = prop.unit;
            }

            properties[prop.name] = propJson;
        }
        module["properties"] = properties;

        root["modules"][name] = module;
    }

    return root;
}

json generateFrameTypesJson() {
    json root;
    root["version"] = "1.0";
    root["generated"] = getCurrentTimestamp();
    root["generator"] = "apra_schema_generator";

    auto& registry = FrameTypeRegistry::instance();
    const auto allTypes = registry.getAllFrameTypes();

    root["frame_type_count"] = allTypes.size();
    root["frame_types"] = json::object();

    for (const auto& name : allTypes) {
        const auto* info = registry.getFrameType(name);
        if (!info) continue;

        json typeJson;
        typeJson["parent"] = info->parent;
        typeJson["description"] = info->description;
        typeJson["tags"] = info->tags;

        // Attributes
        if (!info->attributes.empty()) {
            json attrs = json::object();
            for (const auto& attr : info->attributes) {
                json attrJson;
                attrJson["type"] = attr.type;
                attrJson["required"] = attr.required;
                attrJson["description"] = attr.description;
                if (!attr.enum_values.empty()) {
                    attrJson["enum_values"] = attr.enum_values;
                }
                attrs[attr.name] = attrJson;
            }
            typeJson["attributes"] = attrs;
        }

        // Add computed hierarchy info
        auto ancestors = registry.getAncestors(name);
        if (!ancestors.empty()) {
            typeJson["ancestors"] = ancestors;
        }

        auto subtypes = registry.getSubtypes(name);
        if (!subtypes.empty()) {
            typeJson["subtypes"] = subtypes;
        }

        root["frame_types"][name] = typeJson;
    }

    return root;
}

// ============================================================
// Markdown Generation Functions
// ============================================================

std::string join(const std::vector<std::string>& vec, const std::string& delimiter) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << delimiter;
        oss << vec[i];
    }
    return oss.str();
}

std::string generateModulesMarkdown() {
    std::ostringstream md;
    auto& registry = ModuleRegistry::instance();

    md << "# ApraPipes Modules Reference\n\n";
    md << "> Generated automatically from C++ metadata\n";
    md << "> Generated: " << getCurrentTimestamp() << "\n\n";

    // Table of contents
    md << "## Table of Contents\n\n";

    // Group by category
    std::map<std::string, std::vector<std::string>> byCategory;
    for (const auto& name : registry.getAllModules()) {
        const auto* info = registry.getModule(name);
        if (info) {
            byCategory[detail::categoryToString(info->category)].push_back(name);
        }
    }

    for (const auto& [category, modules] : byCategory) {
        std::string catTitle = category;
        catTitle[0] = std::toupper(catTitle[0]);
        md << "- [" << catTitle << " Modules](#" << category << "-modules)\n";
    }
    md << "\n---\n\n";

    // Module details by category
    for (const auto& [category, modules] : byCategory) {
        std::string catTitle = category;
        catTitle[0] = std::toupper(catTitle[0]);
        md << "## " << catTitle << " Modules\n\n";

        for (const auto& name : modules) {
            const auto* info = registry.getModule(name);
            if (!info) continue;

            md << "### " << info->name << "\n\n";
            md << info->description << "\n\n";

            if (!info->tags.empty()) {
                md << "**Tags:** `" << join(info->tags, "`, `") << "`\n\n";
            }
            md << "**Version:** " << info->version << "\n\n";

            // Inputs
            if (!info->inputs.empty()) {
                md << "#### Inputs\n\n";
                md << "| Pin | Frame Types | Required | Description |\n";
                md << "|-----|-------------|----------|-------------|\n";
                for (const auto& pin : info->inputs) {
                    md << "| `" << pin.name << "` | "
                       << join(pin.frame_types, ", ") << " | "
                       << (pin.required ? "Yes" : "No") << " | "
                       << pin.description << " |\n";
                }
                md << "\n";
            } else {
                md << "#### Inputs\n\nNone (source module)\n\n";
            }

            // Outputs
            if (!info->outputs.empty()) {
                md << "#### Outputs\n\n";
                md << "| Pin | Frame Types | Description |\n";
                md << "|-----|-------------|-------------|\n";
                for (const auto& pin : info->outputs) {
                    md << "| `" << pin.name << "` | "
                       << join(pin.frame_types, ", ") << " | "
                       << pin.description << " |\n";
                }
                md << "\n";
            } else {
                md << "#### Outputs\n\nNone (sink module)\n\n";
            }

            // Properties
            if (!info->properties.empty()) {
                md << "#### Properties\n\n";
                md << "| Property | Type | Default | Description |\n";
                md << "|----------|------|---------|-------------|\n";
                for (const auto& prop : info->properties) {
                    std::string typeInfo = prop.type;
                    if (!prop.enum_values.empty()) {
                        typeInfo += " (" + join(prop.enum_values, "/") + ")";
                    }
                    md << "| `" << prop.name << "` | "
                       << typeInfo << " | "
                       << (prop.default_value.empty() ? "-" : prop.default_value) << " | "
                       << prop.description << " |\n";
                }
                md << "\n";
            }

            md << "---\n\n";
        }
    }

    return md.str();
}

std::string generateFrameTypesMarkdown() {
    std::ostringstream md;
    auto& registry = FrameTypeRegistry::instance();

    md << "# ApraPipes Frame Type Hierarchy\n\n";
    md << "> Generated automatically from C++ metadata\n";
    md << "> Generated: " << getCurrentTimestamp() << "\n\n";

    // Use the registry's toMarkdown if available, or generate our own
    md << registry.toMarkdown();

    return md.str();
}

// ============================================================
// File Writing
// ============================================================

bool writeFile(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << path << "\n";
        return false;
    }
    file << content;
    file.close();
    std::cout << "Generated: " << path << "\n";
    return true;
}

// ============================================================
// Command Line Parsing
// ============================================================

void printUsage(const char* programName) {
    std::cout << "ApraPipes Schema Generator\n\n";
    std::cout << "Usage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --modules-json <path>      Export modules to JSON file\n";
    std::cout << "  --frame-types-json <path>  Export frame types to JSON file\n";
    std::cout << "  --modules-md <path>        Export modules to Markdown file\n";
    std::cout << "  --frame-types-md <path>    Export frame types to Markdown file\n";
    std::cout << "  --all                      Generate all outputs\n";
    std::cout << "  --output-dir <path>        Output directory (default: current)\n";
    std::cout << "  --pretty                   Pretty-print JSON (default: yes)\n";
    std::cout << "  --compact                  Compact JSON output\n";
    std::cout << "  --help                     Show this help message\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " --modules-json modules.json\n";
    std::cout << "  " << programName << " --all --output-dir ./schema\n";
    std::cout << "  " << programName << " --modules-json - --compact   # Output to stdout\n";
}

// ============================================================
// Main Entry Point
// ============================================================

int main(int argc, char* argv[]) {
    std::string modulesJson;
    std::string frameTypesJson;
    std::string modulesMd;
    std::string frameTypesMd;
    std::string outputDir = ".";
    bool generateAll = false;
    bool prettyPrint = true;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--modules-json" && i + 1 < argc) {
            modulesJson = argv[++i];
        } else if (arg == "--frame-types-json" && i + 1 < argc) {
            frameTypesJson = argv[++i];
        } else if (arg == "--modules-md" && i + 1 < argc) {
            modulesMd = argv[++i];
        } else if (arg == "--frame-types-md" && i + 1 < argc) {
            frameTypesMd = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--all") {
            generateAll = true;
        } else if (arg == "--pretty") {
            prettyPrint = true;
        } else if (arg == "--compact") {
            prettyPrint = false;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use --help for usage information.\n";
            return 1;
        }
    }

    // If --all is specified, set default output paths
    if (generateAll) {
        if (modulesJson.empty()) {
            modulesJson = outputDir + "/modules.json";
        }
        if (frameTypesJson.empty()) {
            frameTypesJson = outputDir + "/frame_types.json";
        }
        if (modulesMd.empty()) {
            modulesMd = outputDir + "/MODULES.md";
        }
        if (frameTypesMd.empty()) {
            frameTypesMd = outputDir + "/FRAME_TYPES.md";
        }
    }

    // Check if any output is requested
    if (modulesJson.empty() && frameTypesJson.empty() &&
        modulesMd.empty() && frameTypesMd.empty()) {
        std::cerr << "No output specified. Use --help for usage information.\n";
        return 1;
    }

    bool success = true;
    int jsonIndent = prettyPrint ? 2 : -1;

    // Generate modules JSON
    if (!modulesJson.empty()) {
        try {
            json modulesData = generateModulesJson();
            std::string content = modulesData.dump(jsonIndent);

            if (modulesJson == "-") {
                std::cout << content << "\n";
            } else {
                success = writeFile(modulesJson, content) && success;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error generating modules JSON: " << e.what() << "\n";
            success = false;
        }
    }

    // Generate frame types JSON
    if (!frameTypesJson.empty()) {
        try {
            json frameTypesData = generateFrameTypesJson();
            std::string content = frameTypesData.dump(jsonIndent);

            if (frameTypesJson == "-") {
                std::cout << content << "\n";
            } else {
                success = writeFile(frameTypesJson, content) && success;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error generating frame types JSON: " << e.what() << "\n";
            success = false;
        }
    }

    // Generate modules Markdown
    if (!modulesMd.empty()) {
        try {
            std::string content = generateModulesMarkdown();

            if (modulesMd == "-") {
                std::cout << content << "\n";
            } else {
                success = writeFile(modulesMd, content) && success;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error generating modules Markdown: " << e.what() << "\n";
            success = false;
        }
    }

    // Generate frame types Markdown
    if (!frameTypesMd.empty()) {
        try {
            std::string content = generateFrameTypesMarkdown();

            if (frameTypesMd == "-") {
                std::cout << content << "\n";
            } else {
                success = writeFile(frameTypesMd, content) && success;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error generating frame types Markdown: " << e.what() << "\n";
            success = false;
        }
    }

    if (success) {
        std::cout << "\nSchema generation complete.\n";
        std::cout << "Modules registered: " << ModuleRegistry::instance().size() << "\n";
        std::cout << "Frame types registered: " << FrameTypeRegistry::instance().size() << "\n";
    }

    return success ? 0 : 1;
}
