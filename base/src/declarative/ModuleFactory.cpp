// ============================================================
// File: declarative/ModuleFactory.cpp
// Module Factory implementation
// Task D1: Module Factory
// ============================================================

#include "declarative/ModuleFactory.h"
#include "declarative/ModuleRegistrations.h"
#include "Module.h"
#include <sstream>

namespace apra {

// ============================================================
// BuildResult methods
// ============================================================

std::string ModuleFactory::BuildResult::formatIssues() const {
    std::ostringstream oss;
    for (const auto& issue : issues) {
        switch (issue.level) {
            case BuildIssue::Level::Error:
                oss << "[ERROR] ";
                break;
            case BuildIssue::Level::Warning:
                oss << "[WARN]  ";
                break;
            case BuildIssue::Level::Info:
                oss << "[INFO]  ";
                break;
        }
        oss << issue.code << " @ " << issue.location << ": " << issue.message << "\n";
    }
    return oss.str();
}

// ============================================================
// ModuleFactory constructor
// ============================================================

ModuleFactory::ModuleFactory(Options opts) : options_(std::move(opts)) {}

// ============================================================
// Main build method
// ============================================================

ModuleFactory::BuildResult ModuleFactory::build(const PipelineDescription& desc) {
    // Ensure all modules are registered before building
    ensureBuiltinModulesRegistered();

    BuildResult result;

    // Validate we have something to build
    if (desc.modules.empty()) {
        result.issues.push_back(BuildIssue::error(
            BuildIssue::EMPTY_PIPELINE,
            "pipeline",
            "Pipeline has no modules"
        ));
        return result;
    }

    // Generate pipeline name from settings or use default
    std::string pipelineName = desc.settings.name;
    if (pipelineName.empty()) {
        pipelineName = "declarative_pipeline";
    }

    // Create pipeline
    result.pipeline = std::make_unique<PipeLine>(pipelineName);

    // Map of instance_id -> Module for connection phase
    std::map<std::string, boost::shared_ptr<Module>> moduleMap;

    // Phase 1: Create all modules
    for (const auto& instance : desc.modules) {
        auto module = createModule(instance, result.issues);
        if (module) {
            moduleMap[instance.instance_id] = module;
            result.pipeline->appendModule(module);

            if (options_.collect_info_messages) {
                result.issues.push_back(BuildIssue::info(
                    BuildIssue::MODULE_CREATED,
                    "modules." + instance.instance_id,
                    "Created module: " + instance.module_type
                ));
            }
        }
    }

    // If any critical errors during module creation, stop
    if (result.hasErrors()) {
        result.pipeline.reset();
        return result;
    }

    // Phase 2: Connect modules
    if (!desc.connections.empty()) {
        connectModules(desc.connections, moduleMap, result.issues);
    }

    // If errors during connection, pipeline is invalid
    if (result.hasErrors()) {
        result.pipeline.reset();
        return result;
    }

    // In strict mode, treat warnings as errors
    if (options_.strict_mode && result.hasWarnings()) {
        result.pipeline.reset();
        result.issues.push_back(BuildIssue::error(
            "E099",
            "pipeline",
            "Build failed due to warnings in strict mode"
        ));
    }

    // Remove info messages if not requested
    if (!options_.collect_info_messages) {
        auto& issues = result.issues;
        issues.erase(
            std::remove_if(issues.begin(), issues.end(),
                [](const BuildIssue& i) { return i.level == BuildIssue::Level::Info; }),
            issues.end()
        );
    }

    return result;
}

// ============================================================
// Create a single module
// ============================================================

boost::shared_ptr<Module> ModuleFactory::createModule(
    const ModuleInstance& instance,
    std::vector<BuildIssue>& issues
) {
    auto& registry = ModuleRegistry::instance();

    // Check module exists in registry
    if (!registry.hasModule(instance.module_type)) {
        issues.push_back(BuildIssue::error(
            BuildIssue::UNKNOWN_MODULE,
            "modules." + instance.instance_id,
            "Unknown module type: " + instance.module_type
        ));
        return nullptr;
    }

    // Get module info for property validation
    const ModuleInfo* info = registry.getModule(instance.module_type);

    // Convert PipelineDescription properties to ModuleRegistry format
    // PipelineDescription::PropertyValue includes array types, ScalarPropertyValue doesn't
    std::map<std::string, ScalarPropertyValue> convertedProps;

    for (const auto& [propName, propValue] : instance.properties) {
        // Find property info for type conversion
        const ModuleInfo::PropInfo* propInfo = nullptr;
        if (info) {
            for (const auto& pi : info->properties) {
                if (pi.name == propName) {
                    propInfo = &pi;
                    break;
                }
            }
        }

        std::string location = "modules." + instance.instance_id + ".props." + propName;

        // Convert based on variant type
        std::visit([&](auto&& val) {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, int64_t>) {
                convertedProps[propName] = val;
            }
            else if constexpr (std::is_same_v<T, double>) {
                convertedProps[propName] = val;
            }
            else if constexpr (std::is_same_v<T, bool>) {
                convertedProps[propName] = val;
            }
            else if constexpr (std::is_same_v<T, std::string>) {
                convertedProps[propName] = val;
            }
            else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                if (!val.empty()) {
                    convertedProps[propName] = val[0];
                    issues.push_back(BuildIssue::warning(
                        BuildIssue::PROP_TYPE_CONVERSION,
                        location,
                        "Array property converted to scalar (using first element)"
                    ));
                }
            }
            else if constexpr (std::is_same_v<T, std::vector<double>>) {
                if (!val.empty()) {
                    convertedProps[propName] = val[0];
                    issues.push_back(BuildIssue::warning(
                        BuildIssue::PROP_TYPE_CONVERSION,
                        location,
                        "Array property converted to scalar (using first element)"
                    ));
                }
            }
            else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                if (!val.empty()) {
                    convertedProps[propName] = val[0];
                    issues.push_back(BuildIssue::warning(
                        BuildIssue::PROP_TYPE_CONVERSION,
                        location,
                        "Array property converted to scalar (using first element)"
                    ));
                }
            }
        }, propValue);
    }

    // Validate required properties are provided
    if (info) {
        for (const auto& propInfo : info->properties) {
            if (propInfo.required) {
                auto it = convertedProps.find(propInfo.name);
                if (it == convertedProps.end()) {
                    issues.push_back(BuildIssue::error(
                        BuildIssue::MISSING_REQUIRED_PROP,
                        "modules." + instance.instance_id + ".props." + propInfo.name,
                        "Required property '" + propInfo.name + "' not provided for module '" +
                        instance.module_type + "'"
                    ));
                }
            }
        }
    }

    // If we have errors from missing required properties, don't try to create the module
    bool hasRequiredPropErrors = std::any_of(issues.begin(), issues.end(),
        [&instance](const BuildIssue& i) {
            return i.level == BuildIssue::Level::Error &&
                   i.location.find("modules." + instance.instance_id) != std::string::npos;
        });
    if (hasRequiredPropErrors) {
        return nullptr;
    }

    // Create via registry factory
    try {
        auto modulePtr = registry.createModule(instance.module_type, convertedProps);
        if (!modulePtr) {
            issues.push_back(BuildIssue::error(
                BuildIssue::MODULE_CREATION_FAILED,
                "modules." + instance.instance_id,
                "Factory returned null for module: " + instance.module_type
            ));
            return nullptr;
        }

        // Convert unique_ptr to shared_ptr for ApraPipes compatibility
        return boost::shared_ptr<Module>(modulePtr.release());
    }
    catch (const std::exception& e) {
        issues.push_back(BuildIssue::error(
            BuildIssue::MODULE_CREATION_FAILED,
            "modules." + instance.instance_id,
            "Failed to create module: " + std::string(e.what())
        ));
        return nullptr;
    }
    catch (...) {
        issues.push_back(BuildIssue::error(
            BuildIssue::MODULE_CREATION_FAILED,
            "modules." + instance.instance_id,
            "Unknown exception while creating module"
        ));
        return nullptr;
    }
}

// ============================================================
// Apply properties to module (future use - properties are
// currently passed to factory function)
// ============================================================

void ModuleFactory::applyProperties(
    Module* module,
    const ModuleInstance& instance,
    const ModuleInfo* info,
    std::vector<BuildIssue>& issues
) {
    // Properties are currently passed to the factory function during module creation.
    // This method is here for future use if we need to apply dynamic properties
    // after module creation.
}

// ============================================================
// Connect modules according to connections list
// ============================================================

bool ModuleFactory::connectModules(
    const std::vector<Connection>& connections,
    const std::map<std::string, boost::shared_ptr<Module>>& moduleMap,
    std::vector<BuildIssue>& issues
) {
    bool allSuccess = true;

    for (const auto& conn : connections) {
        // Find source module
        auto srcIt = moduleMap.find(conn.from_module);
        if (srcIt == moduleMap.end()) {
            issues.push_back(BuildIssue::error(
                BuildIssue::UNKNOWN_SOURCE_MODULE,
                "connections",
                "Unknown source module: " + conn.from_module
            ));
            allSuccess = false;
            continue;
        }

        // Find destination module
        auto dstIt = moduleMap.find(conn.to_module);
        if (dstIt == moduleMap.end()) {
            issues.push_back(BuildIssue::error(
                BuildIssue::UNKNOWN_DEST_MODULE,
                "connections",
                "Unknown destination module: " + conn.to_module
            ));
            allSuccess = false;
            continue;
        }

        // Connect using ApraPipes API
        // Note: ApraPipes setNext() connects all output pins by default
        // Pin-specific connections could be implemented in the future
        try {
            bool connected = srcIt->second->setNext(dstIt->second);
            if (!connected) {
                issues.push_back(BuildIssue::error(
                    BuildIssue::CONNECTION_FAILED,
                    "connections",
                    "setNext() returned false for: " + conn.from_module + " -> " + conn.to_module
                ));
                allSuccess = false;
            } else if (options_.collect_info_messages) {
                issues.push_back(BuildIssue::info(
                    BuildIssue::CONNECTION_ESTABLISHED,
                    "connections",
                    "Connected: " + conn.from_module + "." + conn.from_pin +
                    " -> " + conn.to_module + "." + conn.to_pin
                ));
            }
        }
        catch (const std::exception& e) {
            issues.push_back(BuildIssue::error(
                BuildIssue::CONNECTION_FAILED,
                "connections",
                "Connection failed: " + std::string(e.what())
            ));
            allSuccess = false;
        }
        catch (...) {
            issues.push_back(BuildIssue::error(
                BuildIssue::CONNECTION_FAILED,
                "connections",
                "Unknown exception during connection"
            ));
            allSuccess = false;
        }
    }

    return allSuccess;
}

// ============================================================
// Convert PropertyValue from PipelineDescription format
// ============================================================

std::optional<ScalarPropertyValue> ModuleFactory::convertPropertyValue(
    const PropertyValue& value,
    const ModuleInfo::PropInfo& propInfo,
    std::vector<BuildIssue>& issues,
    const std::string& location
) {
    // This is handled inline in createModule for now
    // Keeping this method for potential future use with more complex conversions
    return std::nullopt;
}

} // namespace apra
