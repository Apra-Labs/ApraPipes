// ============================================================
// File: declarative/PipelineValidator.cpp
// Pipeline Validator Shell implementation
// Task C1: Validator Shell
// ============================================================

#include "declarative/PipelineValidator.h"
#include "declarative/ModuleRegistry.h"
#include <sstream>
#include <set>
#include <queue>
#include <algorithm>

namespace apra {

// ============================================================
// Type Conversion Registry - Known conversions between frame types
// ============================================================

struct TypeConversion {
    std::string fromType;
    std::string toType;
    std::string moduleName;
    std::map<std::string, std::string> props;
};

// Static registry of known type conversions
static const std::vector<TypeConversion> KNOWN_CONVERSIONS = {
    // Planar to Packed conversions
    {"RawImagePlanar", "RawImage", "ColorConversion", {{"conversionType", "YUV420PLANAR_TO_RGB"}}},

    // Packed to Planar conversions
    {"RawImage", "RawImagePlanar", "ColorConversion", {{"conversionType", "RGB_TO_YUV420PLANAR"}}},

    // Decode conversions
    {"EncodedImage", "RawImage", "ImageDecoderCV", {}},
    {"H264Data", "RawImagePlanar", "H264Decoder", {}},

    // Encode conversions
    {"RawImage", "EncodedImage", "ImageEncoderCV", {}},
};

// Find a bridge module that converts from srcType to dstType
static const TypeConversion* findTypeConversion(const std::string& srcType, const std::string& dstType) {
    for (const auto& conv : KNOWN_CONVERSIONS) {
        if (conv.fromType == srcType && conv.toType == dstType) {
            return &conv;
        }
    }
    return nullptr;
}

// Generate a suggested module instance name based on conversion
static std::string generateBridgeModuleName(const std::string& fromModule, const std::string& toModule, const TypeConversion& conv) {
    // Create a descriptive name like "convert_generator_to_ptz"
    std::string name = "convert_" + fromModule + "_to_" + toModule;
    // Sanitize: replace any non-alphanumeric chars with underscore
    for (char& c : name) {
        if (!std::isalnum(c) && c != '_') {
            c = '_';
        }
    }
    return name;
}

// Generate TOML snippet for inserting a bridge module
static std::string generateTomlSnippet(
    const std::string& fromModule,
    const std::string& toModule,
    const TypeConversion& conv
) {
    std::string bridgeName = generateBridgeModuleName(fromModule, toModule, conv);
    std::ostringstream oss;

    oss << "\n  Add this module:\n\n";
    oss << "  [modules." << bridgeName << "]\n";
    oss << "  type = \"" << conv.moduleName << "\"\n";

    if (!conv.props.empty()) {
        oss << "    [modules." << bridgeName << ".props]\n";
        for (const auto& [key, value] : conv.props) {
            oss << "    " << key << " = \"" << value << "\"\n";
        }
    }

    oss << "\n  And update connections:\n\n";
    oss << "  [[connections]]\n";
    oss << "  from = \"" << fromModule << "\"\n";
    oss << "  to = \"" << bridgeName << "\"\n";
    oss << "\n";
    oss << "  [[connections]]\n";
    oss << "  from = \"" << bridgeName << "\"\n";
    oss << "  to = \"" << toModule << "\"\n";

    return oss.str();
}

// ============================================================
// Result formatting
// ============================================================

std::string PipelineValidator::Result::format() const {
    std::ostringstream oss;

    for (const auto& issue : issues) {
        switch (issue.level) {
            case Issue::Level::Error:
                oss << "[ERROR] ";
                break;
            case Issue::Level::Warning:
                oss << "[WARN]  ";
                break;
            case Issue::Level::Info:
                oss << "[INFO]  ";
                break;
        }

        oss << issue.code << " @ " << issue.location << ": " << issue.message;

        if (!issue.suggestion.empty()) {
            oss << "\n        Suggestion: " << issue.suggestion;
        }

        oss << "\n";
    }

    return oss.str();
}

// ============================================================
// Constructor
// ============================================================

PipelineValidator::PipelineValidator(Options opts)
    : options_(std::move(opts)) {}

// ============================================================
// Main validation entry point
// ============================================================

PipelineValidator::Result PipelineValidator::validate(const PipelineDescription& desc) const {
    Result result;

    // Add info about what we're validating
    if (options_.includeInfoMessages) {
        result.issues.push_back(Issue::info(
            "I000",
            "pipeline",
            "Validating pipeline: " + desc.settings.name +
            " (" + std::to_string(desc.modules.size()) + " modules, " +
            std::to_string(desc.connections.size()) + " connections)"
        ));
    }

    // Phase 1: Module validation
    auto moduleResult = validateModules(desc);
    result.merge(moduleResult);

    if (options_.stopOnFirstError && result.hasErrors()) {
        return result;
    }

    // Phase 2: Property validation
    auto propResult = validateProperties(desc);
    result.merge(propResult);

    if (options_.stopOnFirstError && result.hasErrors()) {
        return result;
    }

    // Phase 3: Connection validation
    if (options_.validateConnections) {
        auto connResult = validateConnections(desc);
        result.merge(connResult);

        if (options_.stopOnFirstError && result.hasErrors()) {
            return result;
        }
    }

    // Phase 4: Graph validation
    if (options_.validateGraph) {
        auto graphResult = validateGraph(desc);
        result.merge(graphResult);
    }

    // Summary
    if (options_.includeInfoMessages) {
        result.issues.push_back(Issue::info(
            "I001",
            "pipeline",
            "Validation complete: " +
            std::to_string(result.errors().size()) + " errors, " +
            std::to_string(result.warnings().size()) + " warnings"
        ));
    }

    return result;
}

// ============================================================
// Phase 1: Module validation (C2)
// ============================================================

PipelineValidator::Result PipelineValidator::validateModules(const PipelineDescription& desc) const {
    Result result;
    auto& registry = ModuleRegistry::instance();

    for (const auto& module : desc.modules) {
        const std::string location = "modules." + module.instance_id;

        if (options_.includeInfoMessages) {
            result.issues.push_back(Issue::info(
                "I010",
                location,
                "Found module: " + module.module_type
            ));
        }

        // C2: Check module type exists in registry
        if (!registry.hasModule(module.module_type)) {
            // Try to find similar module names for suggestion
            std::string suggestion;
            auto allModules = registry.getAllModules();
            for (const auto& name : allModules) {
                // Simple substring match for suggestions
                if (name.find(module.module_type.substr(0, 4)) != std::string::npos ||
                    module.module_type.find(name.substr(0, 4)) != std::string::npos) {
                    if (suggestion.empty()) {
                        suggestion = "Did you mean: " + name;
                    } else {
                        suggestion += ", " + name;
                    }
                }
            }
            if (suggestion.empty() && !allModules.empty()) {
                suggestion = "Available modules: ";
                for (size_t i = 0; i < std::min(allModules.size(), size_t(5)); ++i) {
                    if (i > 0) suggestion += ", ";
                    suggestion += allModules[i];
                }
                if (allModules.size() > 5) suggestion += "...";
            }

            result.issues.push_back(Issue::error(
                Issue::UNKNOWN_MODULE,
                location,
                "Unknown module type '" + module.module_type + "'",
                suggestion
            ));
        }
    }

    return result;
}

// ============================================================
// Phase 2: Property validation (C3)
// ============================================================

PipelineValidator::Result PipelineValidator::validateProperties(const PipelineDescription& desc) const {
    Result result;
    auto& registry = ModuleRegistry::instance();

    for (const auto& module : desc.modules) {
        const std::string moduleLocation = "modules." + module.instance_id;

        // Skip property validation if module type is unknown (already caught in C2)
        const auto* moduleInfo = registry.getModule(module.module_type);
        if (!moduleInfo) {
            continue;
        }

        if (options_.includeInfoMessages && !module.properties.empty()) {
            result.issues.push_back(Issue::info(
                "I020",
                moduleLocation,
                "Module has " + std::to_string(module.properties.size()) + " properties"
            ));
        }

        // Build map of known properties for this module type
        std::map<std::string, const ModuleInfo::PropInfo*> knownProps;
        for (const auto& prop : moduleInfo->properties) {
            knownProps[prop.name] = &prop;
        }

        // If module has no property metadata, skip property validation for this module
        // (properties will be passed through to the module's applyProperties)
        if (knownProps.empty() && !module.properties.empty()) {
            if (options_.includeInfoMessages) {
                result.issues.push_back(Issue::info(
                    "I021",
                    moduleLocation,
                    "Module type '" + module.module_type + "' has no property metadata; skipping property validation"
                ));
            }
            continue;
        }

        // C3: Check each property in the description
        for (const auto& [propName, propValue] : module.properties) {
            const std::string propLocation = moduleLocation + ".props." + propName;

            auto it = knownProps.find(propName);
            if (it == knownProps.end()) {
                // Unknown property
                std::string suggestion;
                if (!moduleInfo->properties.empty()) {
                    suggestion = "Known properties: ";
                    for (size_t i = 0; i < std::min(moduleInfo->properties.size(), size_t(5)); ++i) {
                        if (i > 0) suggestion += ", ";
                        suggestion += moduleInfo->properties[i].name;
                    }
                }
                result.issues.push_back(Issue::error(
                    Issue::UNKNOWN_PROPERTY,
                    propLocation,
                    "Unknown property '" + propName + "' for module type '" + module.module_type + "'",
                    suggestion
                ));
                continue;
            }

            const auto& propInfo = *it->second;

            // C3: Type checking
            bool typeMatch = false;
            std::string expectedType = propInfo.type;
            std::string actualType;

            std::visit([&](auto&& val) {
                using T = std::decay_t<decltype(val)>;
                if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, std::vector<int64_t>>) {
                    actualType = "int";
                    typeMatch = (propInfo.type == "int");
                } else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::vector<double>>) {
                    actualType = "float";
                    typeMatch = (propInfo.type == "float" || propInfo.type == "int"); // Allow int for float
                } else if constexpr (std::is_same_v<T, bool>) {
                    actualType = "bool";
                    typeMatch = (propInfo.type == "bool");
                } else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::vector<std::string>>) {
                    actualType = "string";
                    typeMatch = (propInfo.type == "string" || propInfo.type == "enum");
                }
            }, propValue);

            if (!typeMatch) {
                result.issues.push_back(Issue::error(
                    Issue::PROPERTY_TYPE_MISMATCH,
                    propLocation,
                    "Type mismatch: expected " + expectedType + ", got " + actualType,
                    ""
                ));
                continue;
            }

            // C3: Range checking for numeric types
            if (propInfo.type == "int") {
                int64_t intVal = 0;
                if (std::holds_alternative<int64_t>(propValue)) {
                    intVal = std::get<int64_t>(propValue);
                } else if (std::holds_alternative<std::vector<int64_t>>(propValue)) {
                    const auto& vec = std::get<std::vector<int64_t>>(propValue);
                    if (!vec.empty()) intVal = vec[0];
                }

                if (!propInfo.min_value.empty() && !propInfo.max_value.empty()) {
                    try {
                        int64_t minVal = std::stoll(propInfo.min_value);
                        int64_t maxVal = std::stoll(propInfo.max_value);
                        if (intVal < minVal || intVal > maxVal) {
                            result.issues.push_back(Issue::error(
                                Issue::PROPERTY_OUT_OF_RANGE,
                                propLocation,
                                "Value " + std::to_string(intVal) + " out of range [" +
                                    propInfo.min_value + ", " + propInfo.max_value + "]",
                                ""
                            ));
                        }
                    } catch (...) {
                        // Ignore parse errors in metadata
                    }
                }
            }

            // C3: Enum value checking
            if (propInfo.type == "enum" && !propInfo.enum_values.empty()) {
                std::string strVal;
                if (std::holds_alternative<std::string>(propValue)) {
                    strVal = std::get<std::string>(propValue);
                }

                bool validEnum = false;
                for (const auto& ev : propInfo.enum_values) {
                    if (ev == strVal) {
                        validEnum = true;
                        break;
                    }
                }

                if (!validEnum) {
                    std::string validValues;
                    for (size_t i = 0; i < propInfo.enum_values.size(); ++i) {
                        if (i > 0) validValues += ", ";
                        validValues += propInfo.enum_values[i];
                    }
                    result.issues.push_back(Issue::error(
                        Issue::PROPERTY_INVALID_ENUM,
                        propLocation,
                        "Invalid enum value '" + strVal + "'",
                        "Valid values: " + validValues
                    ));
                }
            }
        }

        // C3: Check for missing required properties
        for (const auto& propInfo : moduleInfo->properties) {
            if (propInfo.required) {
                if (module.properties.find(propInfo.name) == module.properties.end()) {
                    result.issues.push_back(Issue::warning(
                        Issue::MISSING_REQUIRED_PROPERTY,
                        moduleLocation + ".props",
                        "Missing required property '" + propInfo.name + "'",
                        "Default value: " + propInfo.default_value
                    ));
                }
            }
        }
    }

    return result;
}

// ============================================================
// Phase 3: Connection validation (C4)
// ============================================================

PipelineValidator::Result PipelineValidator::validateConnections(const PipelineDescription& desc) const {
    Result result;
    auto& registry = ModuleRegistry::instance();

    if (options_.includeInfoMessages && !desc.connections.empty()) {
        result.issues.push_back(Issue::info(
            "I030",
            "connections",
            "Pipeline has " + std::to_string(desc.connections.size()) + " connections"
        ));
    }

    // Build module lookup maps
    std::map<std::string, const ModuleInstance*> moduleMap;
    std::map<std::string, const ModuleInfo*> moduleInfoMap;
    for (const auto& module : desc.modules) {
        moduleMap[module.instance_id] = &module;
        moduleInfoMap[module.instance_id] = registry.getModule(module.module_type);
    }

    // Track connected inputs for duplicate detection
    std::set<std::string> connectedInputs;  // "module.pin"

    size_t connIdx = 0;
    for (const auto& conn : desc.connections) {
        const std::string location = "connections[" + std::to_string(connIdx++) + "]";

        // C4: Check source module exists
        auto srcIt = moduleMap.find(conn.from_module);
        if (srcIt == moduleMap.end()) {
            result.issues.push_back(Issue::error(
                Issue::UNKNOWN_SOURCE_MODULE,
                location,
                "Unknown source module '" + conn.from_module + "' in connection",
                ""
            ));
            continue;
        }

        // C4: Check dest module exists
        auto dstIt = moduleMap.find(conn.to_module);
        if (dstIt == moduleMap.end()) {
            result.issues.push_back(Issue::error(
                Issue::UNKNOWN_DEST_MODULE,
                location,
                "Unknown destination module '" + conn.to_module + "' in connection",
                ""
            ));
            continue;
        }

        const auto* srcInfo = moduleInfoMap[conn.from_module];
        const auto* dstInfo = moduleInfoMap[conn.to_module];

        // C4: Compute effective source output types
        // With sieve=false (default), Transform modules pass through their input types
        // as well as their declared output types
        std::vector<std::string> srcPinTypes;

        if (srcInfo && !conn.from_pin.empty()) {
            // Specific pin requested - find it
            bool foundInOutputs = false;
            bool foundInInputs = false;

            // First check output pins (always valid as source)
            for (const auto& pin : srcInfo->outputs) {
                if (pin.name == conn.from_pin) {
                    foundInOutputs = true;
                    for (const auto& ft : pin.frame_types) {
                        srcPinTypes.push_back(ft);
                    }
                    break;
                }
            }

            // If not in outputs, check input pins (valid only with sieve=false)
            if (!foundInOutputs) {
                for (const auto& pin : srcInfo->inputs) {
                    if (pin.name == conn.from_pin) {
                        foundInInputs = true;
                        if (conn.sieve) {
                            // Input pin used as source with sieve=true - ERROR
                            result.issues.push_back(Issue::error(
                                Issue::UNKNOWN_SOURCE_PIN,
                                location,
                                "Input pin '" + conn.from_pin + "' cannot be used as source with sieve=true. "
                                "Input pins only pass through when sieve=false (default).",
                                "Either remove 'sieve: true' from this connection, or use an output pin."
                            ));
                        } else {
                            // sieve=false: input pin passes through, valid as source
                            for (const auto& ft : pin.frame_types) {
                                srcPinTypes.push_back(ft);
                            }
                        }
                        break;
                    }
                }
            }

            if (!foundInOutputs && !foundInInputs) {
                // Pin not found in either outputs or inputs
                std::string suggestion;
                if (!srcInfo->outputs.empty()) {
                    suggestion = "Available output pins: ";
                    for (size_t i = 0; i < srcInfo->outputs.size(); ++i) {
                        if (i > 0) suggestion += ", ";
                        suggestion += srcInfo->outputs[i].name;
                    }
                }
                if (!conn.sieve && !srcInfo->inputs.empty()) {
                    if (!suggestion.empty()) suggestion += ". ";
                    suggestion += "With sieve=false, input pins also available: ";
                    for (size_t i = 0; i < srcInfo->inputs.size(); ++i) {
                        if (i > 0) suggestion += ", ";
                        suggestion += srcInfo->inputs[i].name;
                    }
                }
                result.issues.push_back(Issue::error(
                    Issue::UNKNOWN_SOURCE_PIN,
                    location,
                    "Unknown pin '" + conn.from_pin + "' on module '" + conn.from_module + "'",
                    suggestion
                ));
            }
        } else if (srcInfo) {
            // No specific pin - collect all declared output types
            for (const auto& pin : srcInfo->outputs) {
                for (const auto& ft : pin.frame_types) {
                    srcPinTypes.push_back(ft);
                }
            }

            // Sieve passthrough: with sieve=false (default), Transform modules also
            // pass through their input types as effective outputs
            if (!conn.sieve && srcInfo->category == ModuleCategory::Transform) {
                for (const auto& inputPin : srcInfo->inputs) {
                    for (const auto& ft : inputPin.frame_types) {
                        // Avoid duplicates
                        if (std::find(srcPinTypes.begin(), srcPinTypes.end(), ft) == srcPinTypes.end()) {
                            srcPinTypes.push_back(ft);
                        }
                    }
                }
            }
        }

        // C4: Check dest pin exists (if specified and module info available)
        std::vector<std::string> dstPinTypes;
        if (dstInfo && !conn.to_pin.empty()) {
            bool foundPin = false;
            for (const auto& pin : dstInfo->inputs) {
                if (pin.name == conn.to_pin) {
                    foundPin = true;
                    dstPinTypes = pin.frame_types;
                    break;
                }
            }
            if (!foundPin) {
                std::string suggestion;
                if (!dstInfo->inputs.empty()) {
                    suggestion = "Available input pins: ";
                    for (size_t i = 0; i < dstInfo->inputs.size(); ++i) {
                        if (i > 0) suggestion += ", ";
                        suggestion += dstInfo->inputs[i].name;
                    }
                }
                result.issues.push_back(Issue::error(
                    Issue::UNKNOWN_DEST_PIN,
                    location,
                    "Unknown input pin '" + conn.to_pin + "' on module '" + conn.to_module + "'",
                    suggestion
                ));
            }
        } else if (dstInfo && !dstInfo->inputs.empty()) {
            // Use first input pin types if pin not specified
            dstPinTypes = dstInfo->inputs[0].frame_types;
        }

        // C4: Check frame type compatibility
        // Check if ANY source output type is compatible with ANY destination input type
        if (!srcPinTypes.empty() && !dstPinTypes.empty()) {
            bool compatible = false;
            for (const auto& srcType : srcPinTypes) {
                for (const auto& dstType : dstPinTypes) {
                    if (srcType == dstType || dstType == "*" || srcType == "*" ||
                        dstType == "Frame" || srcType == "Frame") {
                        compatible = true;
                        break;
                    }
                }
                if (compatible) break;
            }
            if (!compatible) {
                // Format source types for error message
                std::string srcTypesStr;
                for (size_t i = 0; i < srcPinTypes.size(); ++i) {
                    if (i > 0) srcTypesStr += ", ";
                    srcTypesStr += srcPinTypes[i];
                }

                std::string dstTypesStr;
                for (size_t i = 0; i < dstPinTypes.size(); ++i) {
                    if (i > 0) dstTypesStr += ", ";
                    dstTypesStr += dstPinTypes[i];
                }

                // Look for a known type conversion (using first source type)
                std::string suggestion;
                if (!srcPinTypes.empty()) {
                    for (const auto& dstType : dstPinTypes) {
                        const TypeConversion* conv = findTypeConversion(srcPinTypes[0], dstType);
                        if (conv) {
                            suggestion = "Insert " + conv->moduleName + " module to convert " +
                                         srcPinTypes[0] + " → " + dstType + ":" +
                                         generateTomlSnippet(conn.from_module, conn.to_module, *conv);
                            break;
                        }
                    }
                }

                result.issues.push_back(Issue::error(
                    Issue::FRAME_TYPE_INCOMPATIBLE,
                    location,
                    "Frame type mismatch: output produces [" + srcTypesStr +
                        "], input expects [" + dstTypesStr + "]",
                    suggestion
                ));
            }
        }

        // C4: Check for duplicate connections to same input
        std::string inputKey = conn.to_module + "." + (conn.to_pin.empty() ? "input" : conn.to_pin);
        if (connectedInputs.find(inputKey) != connectedInputs.end()) {
            result.issues.push_back(Issue::error(
                Issue::DUPLICATE_INPUT_CONNECTION,
                location,
                "Duplicate connection to input '" + inputKey + "'",
                "Each input pin can only have one connection"
            ));
        } else {
            connectedInputs.insert(inputKey);
        }
    }

    // C4: Check required pins are connected
    for (const auto& module : desc.modules) {
        const auto* info = moduleInfoMap[module.instance_id];
        if (!info) continue;

        for (const auto& inputPin : info->inputs) {
            if (inputPin.required) {
                std::string inputKey = module.instance_id + "." + inputPin.name;
                // Also check default "input" key for single-input modules
                bool connected = connectedInputs.find(inputKey) != connectedInputs.end();
                if (!connected && info->inputs.size() == 1) {
                    connected = connectedInputs.find(module.instance_id + ".input") != connectedInputs.end();
                }
                if (!connected) {
                    result.issues.push_back(Issue::warning(
                        Issue::REQUIRED_PIN_UNCONNECTED,
                        "modules." + module.instance_id,
                        "Required input pin '" + inputPin.name + "' is not connected",
                        ""
                    ));
                }
            }
        }
    }

    return result;
}

// ============================================================
// Phase 4: Graph validation (C5)
// ============================================================

PipelineValidator::Result PipelineValidator::validateGraph(const PipelineDescription& desc) const {
    Result result;
    auto& registry = ModuleRegistry::instance();

    if (desc.modules.empty()) {
        if (options_.includeInfoMessages) {
            result.issues.push_back(Issue::info(
                "I040",
                "pipeline",
                "Pipeline has no modules"
            ));
        }
        return result;
    }

    // C5: Check pipeline has at least one source module
    bool hasSource = false;
    for (const auto& module : desc.modules) {
        const auto* info = registry.getModule(module.module_type);
        if (info && info->category == ModuleCategory::Source) {
            hasSource = true;
            break;
        }
    }

    if (!hasSource) {
        result.issues.push_back(Issue::error(
            Issue::NO_SOURCE_MODULE,
            "pipeline",
            "Pipeline has no Source modules (nothing will produce data)",
            "Add a module with category 'Source' like FileReaderModule or Mp4ReaderSource"
        ));
    }

    // Build adjacency list for cycle detection and orphan detection
    std::map<std::string, std::vector<std::string>> adjacency;  // from -> [to, to, ...]
    std::set<std::string> allModules;
    std::set<std::string> hasIncoming;
    std::set<std::string> hasOutgoing;

    for (const auto& module : desc.modules) {
        allModules.insert(module.instance_id);
        adjacency[module.instance_id] = {};  // Initialize empty
    }

    for (const auto& conn : desc.connections) {
        if (allModules.find(conn.from_module) != allModules.end() &&
            allModules.find(conn.to_module) != allModules.end()) {
            adjacency[conn.from_module].push_back(conn.to_module);
            hasOutgoing.insert(conn.from_module);
            hasIncoming.insert(conn.to_module);
        }
    }

    // C5: Check for cycles using DFS with coloring
    // White = 0 (unvisited), Gray = 1 (in current path), Black = 2 (done)
    std::map<std::string, int> color;
    for (const auto& module : desc.modules) {
        color[module.instance_id] = 0;  // White
    }

    std::vector<std::string> cyclePath;
    std::function<bool(const std::string&, std::vector<std::string>&)> hasCycle =
        [&](const std::string& node, std::vector<std::string>& path) -> bool {
            color[node] = 1;  // Gray - in current path
            path.push_back(node);

            for (const auto& neighbor : adjacency[node]) {
                if (color[neighbor] == 1) {
                    // Found cycle - neighbor is in current path
                    path.push_back(neighbor);
                    return true;
                }
                if (color[neighbor] == 0 && hasCycle(neighbor, path)) {
                    return true;
                }
            }

            path.pop_back();
            color[node] = 2;  // Black - done
            return false;
        };

    for (const auto& module : desc.modules) {
        if (color[module.instance_id] == 0) {
            std::vector<std::string> path;
            if (hasCycle(module.instance_id, path)) {
                // Format cycle path
                std::string cycleStr;
                bool inCycle = false;
                std::string cycleStart = path.back();
                for (const auto& node : path) {
                    if (node == cycleStart) inCycle = true;
                    if (inCycle) {
                        if (!cycleStr.empty()) cycleStr += " -> ";
                        cycleStr += node;
                    }
                }
                result.issues.push_back(Issue::error(
                    Issue::GRAPH_HAS_CYCLE,
                    "pipeline",
                    "Cycle detected in pipeline: " + cycleStr,
                    "Remove one of the connections to break the cycle"
                ));
                break;  // Only report first cycle
            }
        }
    }

    // C5: Warn on orphan modules (not connected to anything)
    for (const auto& module : desc.modules) {
        bool isOrphan = (hasIncoming.find(module.instance_id) == hasIncoming.end() &&
                         hasOutgoing.find(module.instance_id) == hasOutgoing.end());

        // Source modules without outgoing connections are also orphans
        // Sink modules without incoming connections are also orphans
        const auto* info = registry.getModule(module.module_type);
        if (info) {
            if (info->category == ModuleCategory::Source &&
                hasOutgoing.find(module.instance_id) == hasOutgoing.end()) {
                isOrphan = true;
            }
            if (info->category == ModuleCategory::Sink &&
                hasIncoming.find(module.instance_id) == hasIncoming.end()) {
                isOrphan = true;
            }
        }

        // Single-module pipelines are not orphans
        if (isOrphan && desc.modules.size() > 1) {
            result.issues.push_back(Issue::warning(
                Issue::ORPHAN_MODULE,
                "modules." + module.instance_id,
                "Module is not connected to any other module (orphan)",
                "Either connect it to the pipeline or remove it"
            ));
        }
    }

    return result;
}

} // namespace apra
