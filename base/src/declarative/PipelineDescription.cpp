// ============================================================
// File: declarative/PipelineDescription.cpp
// Pipeline Description implementation
// Task B1: Pipeline Description IR
// ============================================================

#include "declarative/PipelineDescription.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace apra {

// ============================================================
// Connection implementation
// ============================================================
Connection Connection::parse(const std::string& from, const std::string& to) {
    Connection c;

    auto dot1 = from.find('.');
    auto dot2 = to.find('.');

    if (dot1 != std::string::npos) {
        c.from_module = from.substr(0, dot1);
        c.from_pin = from.substr(dot1 + 1);
    } else {
        // No dot: entire string is module name, pin is empty (use default)
        c.from_module = from;
        c.from_pin = "";  // Will use default output pin
    }

    if (dot2 != std::string::npos) {
        c.to_module = to.substr(0, dot2);
        c.to_pin = to.substr(dot2 + 1);
    } else {
        // No dot: entire string is module name, pin is empty (use default)
        c.to_module = to;
        c.to_pin = "";  // Will use default input pin
    }

    return c;
}

bool Connection::isValid() const {
    // Pins can be empty (use defaults), but modules must be specified
    return !from_module.empty() && !to_module.empty();
}

// ============================================================
// PipelineDescription implementation
// ============================================================
const ModuleInstance* PipelineDescription::findModule(const std::string& id) const {
    auto it = std::find_if(modules.begin(), modules.end(),
        [&id](const ModuleInstance& m) { return m.instance_id == id; });
    return (it != modules.end()) ? &(*it) : nullptr;
}

ModuleInstance* PipelineDescription::findModule(const std::string& id) {
    auto it = std::find_if(modules.begin(), modules.end(),
        [&id](const ModuleInstance& m) { return m.instance_id == id; });
    return (it != modules.end()) ? &(*it) : nullptr;
}

void PipelineDescription::addModule(ModuleInstance module) {
    modules.push_back(std::move(module));
}

void PipelineDescription::addConnection(Connection conn) {
    connections.push_back(std::move(conn));
}

bool PipelineDescription::addConnection(const std::string& from, const std::string& to) {
    Connection conn = Connection::parse(from, to);
    if (!conn.isValid()) {
        return false;
    }
    connections.push_back(std::move(conn));
    return true;
}

bool PipelineDescription::isEmpty() const {
    return modules.empty() && connections.empty();
}

// ============================================================
// JSON serialization helpers
// ============================================================
namespace {

std::string escapeJson(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    o << c;
                }
        }
    }
    return o.str();
}

} // anonymous namespace

std::string propertyValueToString(const PropertyValue& value) {
    return std::visit([](auto&& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int64_t>) {
            return std::to_string(arg);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::to_string(arg);
        } else if constexpr (std::is_same_v<T, bool>) {
            return arg ? "true" : "false";
        } else if constexpr (std::is_same_v<T, std::string>) {
            return "\"" + escapeJson(arg) + "\"";
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            std::ostringstream ss;
            ss << "[";
            for (size_t i = 0; i < arg.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << arg[i];
            }
            ss << "]";
            return ss.str();
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
            std::ostringstream ss;
            ss << "[";
            for (size_t i = 0; i < arg.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << arg[i];
            }
            ss << "]";
            return ss.str();
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            std::ostringstream ss;
            ss << "[";
            for (size_t i = 0; i < arg.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << "\"" << escapeJson(arg[i]) << "\"";
            }
            ss << "]";
            return ss.str();
        }
        return "null";
    }, value);
}

std::string PipelineDescription::toJson() const {
    std::ostringstream json;

    json << "{\n";

    // Settings
    json << "  \"settings\": {\n";
    json << "    \"name\": \"" << escapeJson(settings.name) << "\",\n";
    json << "    \"version\": \"" << escapeJson(settings.version) << "\",\n";
    json << "    \"description\": \"" << escapeJson(settings.description) << "\",\n";
    json << "    \"queue_size\": " << settings.queue_size << ",\n";
    json << "    \"on_error\": \"" << escapeJson(settings.on_error) << "\",\n";
    json << "    \"auto_start\": " << (settings.auto_start ? "true" : "false") << "\n";
    json << "  },\n";

    // Source info
    json << "  \"source\": {\n";
    json << "    \"format\": \"" << escapeJson(source_format) << "\",\n";
    json << "    \"path\": \"" << escapeJson(source_path) << "\"\n";
    json << "  },\n";

    // Modules
    json << "  \"modules\": [\n";
    for (size_t i = 0; i < modules.size(); ++i) {
        const auto& m = modules[i];
        json << "    {\n";
        json << "      \"instance_id\": \"" << escapeJson(m.instance_id) << "\",\n";
        json << "      \"module_type\": \"" << escapeJson(m.module_type) << "\",\n";
        json << "      \"properties\": {";

        bool firstProp = true;
        for (const auto& [key, val] : m.properties) {
            if (!firstProp) json << ",";
            json << "\n        \"" << escapeJson(key) << "\": " << propertyValueToString(val);
            firstProp = false;
        }

        if (!m.properties.empty()) json << "\n      ";
        json << "}\n";
        json << "    }";
        if (i + 1 < modules.size()) json << ",";
        json << "\n";
    }
    json << "  ],\n";

    // Connections
    json << "  \"connections\": [\n";
    for (size_t i = 0; i < connections.size(); ++i) {
        const auto& c = connections[i];
        json << "    {\n";
        json << "      \"from\": \"" << escapeJson(c.from_module) << "." << escapeJson(c.from_pin) << "\",\n";
        json << "      \"to\": \"" << escapeJson(c.to_module) << "." << escapeJson(c.to_pin) << "\"\n";
        json << "    }";
        if (i + 1 < connections.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n";

    json << "}\n";

    return json.str();
}

} // namespace apra
