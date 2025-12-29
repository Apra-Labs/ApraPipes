#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace apra {

// Property values that can appear in TOML/YAML/JSON
using PropertyValue = std::variant<
    int64_t,
    double,
    bool,
    std::string,
    std::vector<int64_t>,
    std::vector<double>,
    std::vector<std::string>
>;

// A single module instance in the pipeline
struct ModuleInstance {
    std::string instance_id;      // User-defined: "my_decoder"
    std::string module_type;      // Registry name: "H264DecoderNvCodec"
    std::map<std::string, PropertyValue> properties;
};

// A connection between two module pins
struct Connection {
    std::string from_module;      // "source"
    std::string from_pin;         // "output"
    std::string to_module;        // "decoder"
    std::string to_pin;           // "input"

    // Helper to parse "module.pin" format
    // e.g., parse("source.output", "decoder.input")
    static Connection parse(const std::string& from, const std::string& to);
};

// Pipeline-level settings
struct PipelineSettings {
    std::string name;
    std::string version = "1.0";
    std::string description;

    int queue_size = 10;
    std::string on_error = "restart_module";  // "stop_pipeline" | "skip_frame"
    bool auto_start = false;
};

// The complete pipeline description IR
struct PipelineDescription {
    PipelineSettings settings;
    std::vector<ModuleInstance> modules;
    std::vector<Connection> connections;

    // Source tracking for error messages
    std::string source_format;    // "toml", "yaml", "json"
    std::string source_path;      // File path or "<inline>"

    // Find a module by its instance_id
    const ModuleInstance* findModule(const std::string& id) const;

    // Serialize to JSON string for debugging
    std::string toJson() const;
};

} // namespace apra
