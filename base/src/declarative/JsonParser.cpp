// ============================================================
// File: declarative/JsonParser.cpp
// JSON Parser Implementation for Pipeline Descriptions
// Task J1: JSON Parser Implementation
// ============================================================

#include "declarative/JsonParser.h"
#include "declarative/ModuleRegistrations.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace apra {

using json = nlohmann::json;

namespace {

// Convert a JSON value to a PropertyValue
PropertyValue toPropertyValue(const json& node) {
    if (node.is_number_integer()) {
        return static_cast<int64_t>(node.get<int64_t>());
    }
    if (node.is_number_float()) {
        return node.get<double>();
    }
    if (node.is_boolean()) {
        return node.get<bool>();
    }
    if (node.is_string()) {
        return node.get<std::string>();
    }
    if (node.is_array()) {
        // Handle empty arrays as int64_t arrays by default
        if (node.empty()) {
            return std::vector<int64_t>{};
        }

        // Detect array type from first element
        const auto& first = node[0];
        if (first.is_number_integer()) {
            std::vector<int64_t> result;
            result.reserve(node.size());
            for (const auto& elem : node) {
                if (elem.is_number_integer()) {
                    result.push_back(elem.get<int64_t>());
                } else {
                    throw std::runtime_error("Mixed types in integer array");
                }
            }
            return result;
        }
        if (first.is_number_float() || first.is_number()) {
            std::vector<double> result;
            result.reserve(node.size());
            for (const auto& elem : node) {
                if (elem.is_number()) {
                    result.push_back(elem.get<double>());
                } else {
                    throw std::runtime_error("Mixed types in float array");
                }
            }
            return result;
        }
        if (first.is_string()) {
            std::vector<std::string> result;
            result.reserve(node.size());
            for (const auto& elem : node) {
                if (elem.is_string()) {
                    result.push_back(elem.get<std::string>());
                } else {
                    throw std::runtime_error("Mixed types in string array");
                }
            }
            return result;
        }
        throw std::runtime_error("Unsupported array element type");
    }

    throw std::runtime_error("Unsupported JSON value type");
}

// Parse "pipeline" section
void parsePipelineSection(const json& root, PipelineDescription& desc) {
    if (!root.contains("pipeline")) {
        return;  // Optional section
    }

    const auto& pipeline = root["pipeline"];
    if (!pipeline.is_object()) {
        return;
    }

    // Parse basic fields
    if (pipeline.contains("name") && pipeline["name"].is_string()) {
        desc.settings.name = pipeline["name"].get<std::string>();
    }
    if (pipeline.contains("version") && pipeline["version"].is_string()) {
        desc.settings.version = pipeline["version"].get<std::string>();
    }
    if (pipeline.contains("description") && pipeline["description"].is_string()) {
        desc.settings.description = pipeline["description"].get<std::string>();
    }
}

// Parse "settings" section
void parseSettingsSection(const json& root, PipelineDescription& desc) {
    if (!root.contains("settings")) {
        return;  // Optional section
    }

    const auto& settings = root["settings"];
    if (!settings.is_object()) {
        return;
    }

    if (settings.contains("queueSize") && settings["queueSize"].is_number_integer()) {
        desc.settings.queue_size = settings["queueSize"].get<int>();
    }
    // Also support snake_case for consistency with TOML
    if (settings.contains("queue_size") && settings["queue_size"].is_number_integer()) {
        desc.settings.queue_size = settings["queue_size"].get<int>();
    }
    if (settings.contains("onError") && settings["onError"].is_string()) {
        desc.settings.on_error = settings["onError"].get<std::string>();
    }
    if (settings.contains("on_error") && settings["on_error"].is_string()) {
        desc.settings.on_error = settings["on_error"].get<std::string>();
    }
    if (settings.contains("autoStart") && settings["autoStart"].is_boolean()) {
        desc.settings.auto_start = settings["autoStart"].get<bool>();
    }
    if (settings.contains("auto_start") && settings["auto_start"].is_boolean()) {
        desc.settings.auto_start = settings["auto_start"].get<bool>();
    }
}

// Parse "modules" section
void parseModulesSection(const json& root, PipelineDescription& desc) {
    if (!root.contains("modules")) {
        return;  // Optional section (though a pipeline without modules isn't useful)
    }

    const auto& modules = root["modules"];
    if (!modules.is_object()) {
        throw std::runtime_error("'modules' must be an object");
    }

    for (auto it = modules.begin(); it != modules.end(); ++it) {
        ModuleInstance instance;
        instance.instance_id = it.key();

        const auto& mod_obj = it.value();
        if (!mod_obj.is_object()) {
            throw std::runtime_error("Module '" + instance.instance_id + "' must be an object");
        }

        // Get type (required)
        if (!mod_obj.contains("type") || !mod_obj["type"].is_string()) {
            throw std::runtime_error("Module '" + instance.instance_id + "' missing required 'type' field");
        }
        instance.module_type = mod_obj["type"].get<std::string>();

        // Get props (optional)
        if (mod_obj.contains("props") && mod_obj["props"].is_object()) {
            const auto& props = mod_obj["props"];
            for (auto pit = props.begin(); pit != props.end(); ++pit) {
                // Skip comment fields (starting with #)
                if (pit.key()[0] == '#') {
                    continue;
                }
                instance.properties[pit.key()] = toPropertyValue(pit.value());
            }
        }

        desc.modules.push_back(std::move(instance));
    }
}

// Parse "connections" section
void parseConnectionsSection(const json& root, PipelineDescription& desc) {
    if (!root.contains("connections")) {
        return;  // Optional section
    }

    const auto& connections = root["connections"];
    if (!connections.is_array()) {
        throw std::runtime_error("'connections' must be an array");
    }

    for (const auto& conn_obj : connections) {
        if (!conn_obj.is_object()) {
            throw std::runtime_error("Connection entry must be an object");
        }

        if (!conn_obj.contains("from") || !conn_obj["from"].is_string()) {
            throw std::runtime_error("Connection missing 'from' field");
        }
        if (!conn_obj.contains("to") || !conn_obj["to"].is_string()) {
            throw std::runtime_error("Connection missing 'to' field");
        }

        std::string from = conn_obj["from"].get<std::string>();
        std::string to = conn_obj["to"].get<std::string>();

        if (from.empty()) {
            throw std::runtime_error("Connection 'from' field is empty");
        }
        if (to.empty()) {
            throw std::runtime_error("Connection 'to' field is empty");
        }

        desc.connections.push_back(Connection::parse(from, to));
    }
}

// Common parsing logic for both file and string input
ParseResult parse(const json& root, const std::string& source_path) {
    ParseResult result;
    result.description.source_format = "json";
    result.description.source_path = source_path;

    try {
        parsePipelineSection(root, result.description);
        parseSettingsSection(root, result.description);
        parseModulesSection(root, result.description);
        parseConnectionsSection(root, result.description);
        result.success = true;
    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
    }

    return result;
}

} // anonymous namespace

ParseResult JsonParser::parseFile(const std::string& filepath) {
    // Ensure all modules are registered before parsing
    ensureBuiltinModulesRegistered();

    ParseResult result;
    result.description.source_format = "json";
    result.description.source_path = filepath;

    try {
        // Check if file exists and is readable
        std::ifstream file(filepath);
        if (!file.is_open()) {
            result.success = false;
            result.error = "Could not open file: " + filepath;
            return result;
        }

        // Parse JSON from file
        json root;
        try {
            file >> root;
        } catch (const json::parse_error& e) {
            result.success = false;
            result.error = e.what();
            // nlohmann::json provides byte position, not line/column
            // We could calculate line/column from byte position if needed
            return result;
        }

        return parse(root, filepath);
    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
        return result;
    }
}

ParseResult JsonParser::parseString(const std::string& content,
                                    const std::string& source_name) {
    // Ensure all modules are registered before parsing
    ensureBuiltinModulesRegistered();

    ParseResult result;
    result.description.source_format = "json";
    result.description.source_path = source_name;

    try {
        // Parse JSON from string
        json root;
        try {
            root = json::parse(content);
        } catch (const json::parse_error& e) {
            result.success = false;
            result.error = e.what();
            return result;
        }

        return parse(root, source_name);
    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
        return result;
    }
}

} // namespace apra
