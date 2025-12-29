#include "PipelineDescription.h"
#include <sstream>
#include <stdexcept>

namespace apra {

Connection Connection::parse(const std::string& from, const std::string& to) {
    Connection c;

    auto dot1 = from.find('.');
    if (dot1 == std::string::npos) {
        throw std::invalid_argument("Invalid 'from' format: expected 'module.pin', got '" + from + "'");
    }

    auto dot2 = to.find('.');
    if (dot2 == std::string::npos) {
        throw std::invalid_argument("Invalid 'to' format: expected 'module.pin', got '" + to + "'");
    }

    c.from_module = from.substr(0, dot1);
    c.from_pin = from.substr(dot1 + 1);
    c.to_module = to.substr(0, dot2);
    c.to_pin = to.substr(dot2 + 1);

    return c;
}

const ModuleInstance* PipelineDescription::findModule(const std::string& id) const {
    for (const auto& module : modules) {
        if (module.instance_id == id) {
            return &module;
        }
    }
    return nullptr;
}

namespace {
// Helper to escape strings for JSON
std::string escapeJson(const std::string& s) {
    std::string result;
    result.reserve(s.size() + 10);
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b";  break;
            case '\f': result += "\\f";  break;
            case '\n': result += "\\n";  break;
            case '\r': result += "\\r";  break;
            case '\t': result += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    // Control character - output as unicode escape
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
                break;
        }
    }
    return result;
}

// Visitor to convert PropertyValue to JSON
struct PropertyValueToJson {
    std::string operator()(int64_t v) const {
        return std::to_string(v);
    }

    std::string operator()(double v) const {
        std::ostringstream oss;
        oss.precision(17);
        oss << v;
        return oss.str();
    }

    std::string operator()(bool v) const {
        return v ? "true" : "false";
    }

    std::string operator()(const std::string& v) const {
        return "\"" + escapeJson(v) + "\"";
    }

    std::string operator()(const std::vector<int64_t>& v) const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) oss << ",";
            oss << v[i];
        }
        oss << "]";
        return oss.str();
    }

    std::string operator()(const std::vector<double>& v) const {
        std::ostringstream oss;
        oss.precision(17);
        oss << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) oss << ",";
            oss << v[i];
        }
        oss << "]";
        return oss.str();
    }

    std::string operator()(const std::vector<std::string>& v) const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "\"" << escapeJson(v[i]) << "\"";
        }
        oss << "]";
        return oss.str();
    }
};
} // anonymous namespace

std::string PipelineDescription::toJson() const {
    std::ostringstream oss;
    PropertyValueToJson valueVisitor;

    oss << "{";

    // Settings
    oss << "\"settings\":{";
    oss << "\"name\":\"" << escapeJson(settings.name) << "\",";
    oss << "\"version\":\"" << escapeJson(settings.version) << "\",";
    oss << "\"description\":\"" << escapeJson(settings.description) << "\",";
    oss << "\"queue_size\":" << settings.queue_size << ",";
    oss << "\"on_error\":\"" << escapeJson(settings.on_error) << "\",";
    oss << "\"auto_start\":" << (settings.auto_start ? "true" : "false");
    oss << "},";

    // Modules
    oss << "\"modules\":[";
    for (size_t i = 0; i < modules.size(); ++i) {
        if (i > 0) oss << ",";
        const auto& mod = modules[i];
        oss << "{";
        oss << "\"instance_id\":\"" << escapeJson(mod.instance_id) << "\",";
        oss << "\"module_type\":\"" << escapeJson(mod.module_type) << "\",";
        oss << "\"properties\":{";
        bool first = true;
        for (const auto& [key, value] : mod.properties) {
            if (!first) oss << ",";
            first = false;
            oss << "\"" << escapeJson(key) << "\":" << std::visit(valueVisitor, value);
        }
        oss << "}";
        oss << "}";
    }
    oss << "],";

    // Connections
    oss << "\"connections\":[";
    for (size_t i = 0; i < connections.size(); ++i) {
        if (i > 0) oss << ",";
        const auto& conn = connections[i];
        oss << "{";
        oss << "\"from_module\":\"" << escapeJson(conn.from_module) << "\",";
        oss << "\"from_pin\":\"" << escapeJson(conn.from_pin) << "\",";
        oss << "\"to_module\":\"" << escapeJson(conn.to_module) << "\",";
        oss << "\"to_pin\":\"" << escapeJson(conn.to_pin) << "\"";
        oss << "}";
    }
    oss << "],";

    // Source tracking
    oss << "\"source_format\":\"" << escapeJson(source_format) << "\",";
    oss << "\"source_path\":\"" << escapeJson(source_path) << "\"";

    oss << "}";

    return oss.str();
}

} // namespace apra
