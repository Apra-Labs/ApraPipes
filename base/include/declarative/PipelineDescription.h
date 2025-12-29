// ============================================================
// File: declarative/PipelineDescription.h
// Pipeline Description Intermediate Representation
// Task B1: Pipeline Description IR
// ============================================================

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <variant>

namespace apra {

// ============================================================
// PropertyValue - Variant type for pipeline configuration values
// Extended with array types for TOML/YAML/JSON parsing
// ============================================================
using PropertyValue = std::variant<
    int64_t,
    double,
    bool,
    std::string,
    std::vector<int64_t>,
    std::vector<double>,
    std::vector<std::string>
>;

// ============================================================
// ModuleInstance - A module instance in the pipeline
// ============================================================
struct ModuleInstance {
    std::string instance_id;      // User-defined: "my_decoder"
    std::string module_type;      // Registry name: "H264DecoderNvCodec"
    std::map<std::string, PropertyValue> properties;

    // Source location for error messages
    int source_line = -1;
    int source_column = -1;
};

// ============================================================
// Connection - Links output pin to input pin
// ============================================================
struct Connection {
    std::string from_module;      // "source"
    std::string from_pin;         // "output"
    std::string to_module;        // "decoder"
    std::string to_pin;           // "input"

    // Source location for error messages
    int source_line = -1;
    int source_column = -1;

    // Helper to parse "module.pin" format
    // Returns empty strings on parse error
    static Connection parse(const std::string& from, const std::string& to);

    // Check if connection is valid (all fields non-empty)
    bool isValid() const;
};

// ============================================================
// PipelineSettings - Global pipeline configuration
// ============================================================
struct PipelineSettings {
    std::string name;
    std::string version = "1.0";
    std::string description;

    int queue_size = 10;
    std::string on_error = "restart_module";  // "stop_pipeline" | "skip_frame"
    bool auto_start = false;
};

// ============================================================
// PipelineDescription - Complete pipeline definition
// This is what parsers produce and validators consume
// ============================================================
struct PipelineDescription {
    PipelineSettings settings;
    std::vector<ModuleInstance> modules;
    std::vector<Connection> connections;

    // Source tracking for error messages
    std::string source_format;    // "toml", "yaml", "json", "programmatic"
    std::string source_path;      // File path or "<inline>"

    // ========================================================
    // Helpers
    // ========================================================

    // Find module by instance_id, returns nullptr if not found
    const ModuleInstance* findModule(const std::string& id) const;
    ModuleInstance* findModule(const std::string& id);

    // Add a module instance
    void addModule(ModuleInstance module);

    // Add a connection
    void addConnection(Connection conn);

    // Add connection using "module.pin" syntax
    // Returns false if parsing fails
    bool addConnection(const std::string& from, const std::string& to);

    // Check if description is empty (no modules or connections)
    bool isEmpty() const;

    // Serialize to JSON (for debugging)
    std::string toJson() const;
};

// ============================================================
// Helper: Get property as specific type with default value
// ============================================================
template<typename T>
T getProperty(const std::map<std::string, PropertyValue>& props,
              const std::string& key,
              const T& defaultValue) {
    auto it = props.find(key);
    if (it == props.end()) {
        return defaultValue;
    }
    try {
        return std::get<T>(it->second);
    } catch (const std::bad_variant_access&) {
        return defaultValue;
    }
}

// ============================================================
// Helper: Convert PropertyValue to string for display
// ============================================================
std::string propertyValueToString(const PropertyValue& value);

} // namespace apra
