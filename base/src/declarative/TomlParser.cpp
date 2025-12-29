#include "declarative/TomlParser.h"

#define TOML_EXCEPTIONS 1
#include <toml++/toml.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace apra {

namespace {

// Convert a TOML node to a PropertyValue
PropertyValue toPropertyValue(const toml::node& node) {
    if (auto* i = node.as_integer()) {
        return static_cast<int64_t>(i->get());
    }
    if (auto* f = node.as_floating_point()) {
        return f->get();
    }
    if (auto* b = node.as_boolean()) {
        return b->get();
    }
    if (auto* s = node.as_string()) {
        return std::string(s->get());
    }
    if (auto* arr = node.as_array()) {
        // Handle empty arrays as int64_t arrays by default
        if (arr->empty()) {
            return std::vector<int64_t>{};
        }

        // Detect array type from first element
        const auto& first = arr->front();
        if (first.is_integer()) {
            std::vector<int64_t> result;
            result.reserve(arr->size());
            for (const auto& elem : *arr) {
                if (auto* ival = elem.as_integer()) {
                    result.push_back(static_cast<int64_t>(ival->get()));
                } else {
                    throw std::runtime_error("Mixed types in integer array");
                }
            }
            return result;
        }
        if (first.is_floating_point()) {
            std::vector<double> result;
            result.reserve(arr->size());
            for (const auto& elem : *arr) {
                if (auto* fval = elem.as_floating_point()) {
                    result.push_back(fval->get());
                } else if (auto* ival = elem.as_integer()) {
                    // Allow integers in float arrays
                    result.push_back(static_cast<double>(ival->get()));
                } else {
                    throw std::runtime_error("Mixed types in float array");
                }
            }
            return result;
        }
        if (first.is_string()) {
            std::vector<std::string> result;
            result.reserve(arr->size());
            for (const auto& elem : *arr) {
                if (auto* sval = elem.as_string()) {
                    result.push_back(std::string(sval->get()));
                } else {
                    throw std::runtime_error("Mixed types in string array");
                }
            }
            return result;
        }
        throw std::runtime_error("Unsupported array element type");
    }

    throw std::runtime_error("Unsupported TOML value type");
}

// Parse [pipeline] section
void parsePipelineSection(const toml::table& root, PipelineDescription& desc) {
    auto pipeline = root["pipeline"].as_table();
    if (!pipeline) {
        return;  // Optional section
    }

    // Parse basic fields
    if (auto name = (*pipeline)["name"].value<std::string>()) {
        desc.settings.name = *name;
    }
    if (auto version = (*pipeline)["version"].value<std::string>()) {
        desc.settings.version = *version;
    }
    if (auto description = (*pipeline)["description"].value<std::string>()) {
        desc.settings.description = *description;
    }

    // Parse [pipeline.settings] subsection
    auto settings = (*pipeline)["settings"].as_table();
    if (settings) {
        if (auto queue_size = (*settings)["queue_size"].value<int64_t>()) {
            desc.settings.queue_size = static_cast<int>(*queue_size);
        }
        if (auto on_error = (*settings)["on_error"].value<std::string>()) {
            desc.settings.on_error = *on_error;
        }
        if (auto auto_start = (*settings)["auto_start"].value<bool>()) {
            desc.settings.auto_start = *auto_start;
        }
    }
}

// Parse [modules.*] sections
void parseModulesSection(const toml::table& root, PipelineDescription& desc) {
    auto modules = root["modules"].as_table();
    if (!modules) {
        return;  // Optional section (though a pipeline without modules isn't useful)
    }

    for (const auto& [id, value] : *modules) {
        ModuleInstance instance;
        instance.instance_id = std::string(id.str());

        auto* mod_table = value.as_table();
        if (!mod_table) {
            throw std::runtime_error("Module '" + instance.instance_id + "' must be a table");
        }

        // Get type (required)
        auto type_val = (*mod_table)["type"].value<std::string>();
        if (!type_val) {
            throw std::runtime_error("Module '" + instance.instance_id + "' missing required 'type' field");
        }
        instance.module_type = *type_val;

        // Get props (optional)
        auto props = (*mod_table)["props"].as_table();
        if (props) {
            for (const auto& [key, val] : *props) {
                instance.properties[std::string(key.str())] = toPropertyValue(val);
            }
        }

        desc.modules.push_back(std::move(instance));
    }
}

// Parse [[connections]] array
void parseConnectionsSection(const toml::table& root, PipelineDescription& desc) {
    auto connections = root["connections"].as_array();
    if (!connections) {
        return;  // Optional section
    }

    for (const auto& conn_node : *connections) {
        auto* conn_table = conn_node.as_table();
        if (!conn_table) {
            throw std::runtime_error("Connection entry must be a table");
        }

        auto from = (*conn_table)["from"].value<std::string>();
        auto to = (*conn_table)["to"].value<std::string>();

        if (!from || from->empty()) {
            throw std::runtime_error("Connection missing 'from' field");
        }
        if (!to || to->empty()) {
            throw std::runtime_error("Connection missing 'to' field");
        }

        desc.connections.push_back(Connection::parse(*from, *to));
    }
}

// Common parsing logic for both file and string input
ParseResult parse(const toml::table& root, const std::string& source_format,
                  const std::string& source_path) {
    ParseResult result;
    result.description.source_format = source_format;
    result.description.source_path = source_path;

    try {
        parsePipelineSection(root, result.description);
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

ParseResult TomlParser::parseFile(const std::string& filepath) {
    ParseResult result;
    result.description.source_format = "toml";
    result.description.source_path = filepath;

    try {
        // Check if file exists and is readable
        std::ifstream file(filepath);
        if (!file.is_open()) {
            result.success = false;
            result.error = "Could not open file: " + filepath;
            return result;
        }
        file.close();

        auto root = toml::parse_file(filepath);
        return parse(root, "toml", filepath);
    } catch (const toml::parse_error& err) {
        result.success = false;
        result.error = err.what();
        if (err.source().begin.line > 0) {
            result.error_line = static_cast<int>(err.source().begin.line);
        }
        if (err.source().begin.column > 0) {
            result.error_column = static_cast<int>(err.source().begin.column);
        }
        return result;
    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
        return result;
    }
}

ParseResult TomlParser::parseString(const std::string& content,
                                    const std::string& source_name) {
    ParseResult result;
    result.description.source_format = "toml";
    result.description.source_path = source_name;

    try {
        auto root = toml::parse(content, source_name);
        return parse(root, "toml", source_name);
    } catch (const toml::parse_error& err) {
        result.success = false;
        result.error = err.what();
        if (err.source().begin.line > 0) {
            result.error_line = static_cast<int>(err.source().begin.line);
        }
        if (err.source().begin.column > 0) {
            result.error_column = static_cast<int>(err.source().begin.column);
        }
        return result;
    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
        return result;
    }
}

} // namespace apra
