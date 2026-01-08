// ============================================================
// File: declarative/ModuleRegistry.cpp
// Module Registry implementation
// Task A2: Module Registry
// ============================================================

#include "declarative/ModuleRegistry.h"
#include "Module.h"  // Need complete type for unique_ptr<Module>
#include <sstream>
#include <algorithm>

namespace apra {

// ============================================================
// Singleton implementation
// ============================================================
ModuleRegistry& ModuleRegistry::instance() {
    static ModuleRegistry registry;
    return registry;
}

// ============================================================
// Registration
// ============================================================
void ModuleRegistry::registerModule(ModuleInfo info) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (modules_.count(info.name)) {
        // Module already registered - warn but don't fail
        // In production, you might want to log this
        return;
    }

    modules_[info.name] = std::move(info);
}

void ModuleRegistry::addRegistrationCallback(RegistrationCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    registrationCallbacks_.push_back(std::move(callback));
}

void ModuleRegistry::rerunRegistrations() {
    // Copy callbacks to avoid holding lock during calls
    std::vector<RegistrationCallback> callbacks;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        callbacks = registrationCallbacks_;
    }

    // Run all registration callbacks (they check hasModule internally)
    for (const auto& callback : callbacks) {
        callback();
    }
}

// ============================================================
// Queries
// ============================================================
bool ModuleRegistry::hasModule(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return modules_.count(name) > 0;
}

const ModuleInfo* ModuleRegistry::getModule(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = modules_.find(name);
    if (it == modules_.end()) {
        return nullptr;
    }
    return &it->second;
}

std::vector<std::string> ModuleRegistry::getAllModules() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(modules_.size());
    for (const auto& [name, _] : modules_) {
        names.push_back(name);
    }
    return names;
}

std::vector<std::string> ModuleRegistry::getModulesByCategory(ModuleCategory cat) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    for (const auto& [name, info] : modules_) {
        if (info.category == cat) {
            names.push_back(name);
        }
    }
    return names;
}

std::vector<std::string> ModuleRegistry::getModulesByTag(const std::string& tag) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    for (const auto& [name, info] : modules_) {
        if (std::find(info.tags.begin(), info.tags.end(), tag) != info.tags.end()) {
            names.push_back(name);
        }
    }
    return names;
}

// ============================================================
// Factory
// ============================================================
std::unique_ptr<Module> ModuleRegistry::createModule(
    const std::string& name,
    const std::map<std::string, ScalarPropertyValue>& props
) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = modules_.find(name);
    if (it == modules_.end()) {
        return nullptr;
    }
    if (!it->second.factory) {
        return nullptr;
    }
    return it->second.factory(props);
}

// ============================================================
// CUDA Factory
// ============================================================
std::unique_ptr<Module> ModuleRegistry::createCudaModule(
    const std::string& name,
    const std::map<std::string, ScalarPropertyValue>& props,
    void* cudaStreamPtr
) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = modules_.find(name);
    if (it == modules_.end()) {
        return nullptr;
    }
    if (!it->second.cudaFactory) {
        // Fall back to regular factory if no CUDA factory
        if (it->second.factory) {
            return it->second.factory(props);
        }
        return nullptr;
    }
    return it->second.cudaFactory(props, cudaStreamPtr);
}

bool ModuleRegistry::setCudaFactory(const std::string& name, ModuleInfo::CudaFactoryFn factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = modules_.find(name);
    if (it == modules_.end()) {
        return false;
    }
    it->second.cudaFactory = std::move(factory);
    it->second.requiresCudaStream = true;
    return true;
}

bool ModuleRegistry::moduleRequiresCudaStream(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = modules_.find(name);
    if (it == modules_.end()) {
        return false;
    }
    return it->second.requiresCudaStream;
}

// ============================================================
// Export - JSON
// ============================================================
std::string ModuleRegistry::toJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream json;

    json << "{\n";
    json << "  \"modules\": [\n";

    bool firstModule = true;
    for (const auto& [name, info] : modules_) {
        if (!firstModule) json << ",\n";
        firstModule = false;

        json << "    {\n";
        json << "      \"name\": \"" << info.name << "\",\n";
        json << "      \"category\": \"" << detail::categoryToString(info.category) << "\",\n";
        json << "      \"version\": \"" << info.version << "\",\n";
        json << "      \"description\": \"" << info.description << "\",\n";

        // Tags
        json << "      \"tags\": [";
        for (size_t i = 0; i < info.tags.size(); ++i) {
            if (i > 0) json << ", ";
            json << "\"" << info.tags[i] << "\"";
        }
        json << "],\n";

        // Inputs
        json << "      \"inputs\": [";
        for (size_t i = 0; i < info.inputs.size(); ++i) {
            if (i > 0) json << ", ";
            const auto& pin = info.inputs[i];
            json << "{\n";
            json << "        \"name\": \"" << pin.name << "\",\n";
            json << "        \"required\": " << (pin.required ? "true" : "false") << ",\n";
            json << "        \"frame_types\": [";
            for (size_t j = 0; j < pin.frame_types.size(); ++j) {
                if (j > 0) json << ", ";
                json << "\"" << pin.frame_types[j] << "\"";
            }
            json << "],\n";
            json << "        \"description\": \"" << pin.description << "\"\n";
            json << "      }";
        }
        json << "],\n";

        // Outputs
        json << "      \"outputs\": [";
        for (size_t i = 0; i < info.outputs.size(); ++i) {
            if (i > 0) json << ", ";
            const auto& pin = info.outputs[i];
            json << "{\n";
            json << "        \"name\": \"" << pin.name << "\",\n";
            json << "        \"required\": " << (pin.required ? "true" : "false") << ",\n";
            json << "        \"frame_types\": [";
            for (size_t j = 0; j < pin.frame_types.size(); ++j) {
                if (j > 0) json << ", ";
                json << "\"" << pin.frame_types[j] << "\"";
            }
            json << "],\n";
            json << "        \"description\": \"" << pin.description << "\"\n";
            json << "      }";
        }
        json << "],\n";

        // Properties
        json << "      \"properties\": [";
        for (size_t i = 0; i < info.properties.size(); ++i) {
            if (i > 0) json << ", ";
            const auto& prop = info.properties[i];
            json << "{\n";
            json << "        \"name\": \"" << prop.name << "\",\n";
            json << "        \"type\": \"" << prop.type << "\",\n";
            json << "        \"mutability\": \"" << prop.mutability << "\",\n";
            json << "        \"default\": \"" << prop.default_value << "\"";
            if (!prop.min_value.empty()) {
                json << ",\n        \"min\": \"" << prop.min_value << "\"";
            }
            if (!prop.max_value.empty()) {
                json << ",\n        \"max\": \"" << prop.max_value << "\"";
            }
            if (!prop.enum_values.empty()) {
                json << ",\n        \"enum_values\": [";
                for (size_t j = 0; j < prop.enum_values.size(); ++j) {
                    if (j > 0) json << ", ";
                    json << "\"" << prop.enum_values[j] << "\"";
                }
                json << "]";
            }
            if (!prop.description.empty()) {
                json << ",\n        \"description\": \"" << prop.description << "\"";
            }
            json << "\n      }";
        }
        json << "]\n";

        json << "    }";
    }

    json << "\n  ]\n";
    json << "}\n";

    return json.str();
}

// ============================================================
// Export - TOML
// ============================================================
std::string ModuleRegistry::toToml() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream toml;

    toml << "# ApraPipes Module Registry\n";
    toml << "# Auto-generated - do not edit manually\n\n";

    for (const auto& [name, info] : modules_) {
        toml << "[[module]]\n";
        toml << "name = \"" << info.name << "\"\n";
        toml << "category = \"" << detail::categoryToString(info.category) << "\"\n";
        toml << "version = \"" << info.version << "\"\n";
        toml << "description = \"" << info.description << "\"\n";

        // Tags
        toml << "tags = [";
        for (size_t i = 0; i < info.tags.size(); ++i) {
            if (i > 0) toml << ", ";
            toml << "\"" << info.tags[i] << "\"";
        }
        toml << "]\n";

        // Inputs
        for (const auto& pin : info.inputs) {
            toml << "\n[[module.input]]\n";
            toml << "name = \"" << pin.name << "\"\n";
            toml << "required = " << (pin.required ? "true" : "false") << "\n";
            toml << "frame_types = [";
            for (size_t j = 0; j < pin.frame_types.size(); ++j) {
                if (j > 0) toml << ", ";
                toml << "\"" << pin.frame_types[j] << "\"";
            }
            toml << "]\n";
            if (!pin.description.empty()) {
                toml << "description = \"" << pin.description << "\"\n";
            }
        }

        // Outputs
        for (const auto& pin : info.outputs) {
            toml << "\n[[module.output]]\n";
            toml << "name = \"" << pin.name << "\"\n";
            toml << "required = " << (pin.required ? "true" : "false") << "\n";
            toml << "frame_types = [";
            for (size_t j = 0; j < pin.frame_types.size(); ++j) {
                if (j > 0) toml << ", ";
                toml << "\"" << pin.frame_types[j] << "\"";
            }
            toml << "]\n";
            if (!pin.description.empty()) {
                toml << "description = \"" << pin.description << "\"\n";
            }
        }

        // Properties
        for (const auto& prop : info.properties) {
            toml << "\n[[module.property]]\n";
            toml << "name = \"" << prop.name << "\"\n";
            toml << "type = \"" << prop.type << "\"\n";
            toml << "mutability = \"" << prop.mutability << "\"\n";
            toml << "default = \"" << prop.default_value << "\"\n";
            if (!prop.min_value.empty()) {
                toml << "min = \"" << prop.min_value << "\"\n";
            }
            if (!prop.max_value.empty()) {
                toml << "max = \"" << prop.max_value << "\"\n";
            }
            if (!prop.enum_values.empty()) {
                toml << "enum_values = [";
                for (size_t j = 0; j < prop.enum_values.size(); ++j) {
                    if (j > 0) toml << ", ";
                    toml << "\"" << prop.enum_values[j] << "\"";
                }
                toml << "]\n";
            }
            if (!prop.description.empty()) {
                toml << "description = \"" << prop.description << "\"\n";
            }
        }

        toml << "\n";
    }

    return toml.str();
}

// ============================================================
// Utility
// ============================================================
void ModuleRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    modules_.clear();
}

size_t ModuleRegistry::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return modules_.size();
}

} // namespace apra
