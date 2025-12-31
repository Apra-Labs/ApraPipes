// ============================================================
// File: declarative/ModuleRegistrationBuilder.h
// Task D2: Property Binding System
//
// Provides fluent builder for module registration.
// Usage:
//   registerModule<FileReaderModule, FileReaderModuleProps>()
//       .category(ModuleCategory::Source)
//       .description("Reads frames from files")
//       .tags("reader", "file")
//       .output("output", "EncodedImage");
// ============================================================

#pragma once

#include "ModuleRegistry.h"
#include "PropertyMacros.h"
#include "Module.h"  // For ModuleProps base class
#include <string>
#include <vector>
#include <type_traits>
#include <typeinfo>
#include <iostream>  // Debug output
#include <limits>    // For std::numeric_limits
#include <cstdint>   // For INT64_MIN, INT64_MAX

#ifdef __GNUC__
#include <cxxabi.h>
#endif

namespace apra {

// Type trait has_apply_properties is defined in ModuleRegistry.h

namespace detail {
    // Helper to join strings
    inline std::string joinStrings(const std::vector<std::string>& vec, const std::string& sep = ", ") {
        std::string result;
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) result += sep;
            result += vec[i];
        }
        return result;
    }
}

// ============================================================
// Helper to demangle type names (for automatic name detection)
// ============================================================
inline std::string demangleTypeName(const char* mangledName) {
#ifdef __GNUC__
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangledName, nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string result(demangled);
        free(demangled);
        return result;
    }
#endif
    return std::string(mangledName);
}

// Extract just the class name (remove namespace and MSVC prefixes)
inline std::string extractClassName(const std::string& fullName) {
    std::string name = fullName;

    // MSVC prepends "class " or "struct " to type names
    if (name.substr(0, 6) == "class ") {
        name = name.substr(6);
    } else if (name.substr(0, 7) == "struct ") {
        name = name.substr(7);
    }

    // Remove namespace prefix
    auto pos = name.rfind("::");
    if (pos != std::string::npos) {
        name = name.substr(pos + 2);
    }

    return name;
}

// ============================================================
// ModuleRegistrationBuilder - fluent builder for module registration
// ============================================================
template<typename ModuleClass, typename PropsClass>
class ModuleRegistrationBuilder {
    ModuleInfo info_;
    bool registered_ = false;

public:
    ModuleRegistrationBuilder() {
        // Automatically derive module name from class name
        std::string fullName = demangleTypeName(typeid(ModuleClass).name());
        info_.name = extractClassName(fullName);
    }

    // Explicitly set name (optional - usually auto-derived)
    ModuleRegistrationBuilder& name(const std::string& n) {
        info_.name = n;
        return *this;
    }

    // Set category (required)
    ModuleRegistrationBuilder& category(ModuleCategory cat) {
        info_.category = cat;
        return *this;
    }

    // Set description (required)
    ModuleRegistrationBuilder& description(const std::string& desc) {
        info_.description = desc;
        return *this;
    }

    // Set version (optional, defaults to "1.0")
    ModuleRegistrationBuilder& version(const std::string& ver) {
        info_.version = ver;
        return *this;
    }

    // Add single tag
    ModuleRegistrationBuilder& tag(const std::string& t) {
        info_.tags.push_back(t);
        return *this;
    }

    // Add multiple tags (variadic)
    template<typename... Tags>
    ModuleRegistrationBuilder& tags(Tags... t) {
        (info_.tags.push_back(std::string(t)), ...);
        return *this;
    }

    // Add input pin with single frame type
    ModuleRegistrationBuilder& input(const std::string& pinName, const std::string& frameType) {
        ModuleInfo::PinInfo pin;
        pin.name = pinName;
        pin.frame_types.push_back(frameType);
        pin.required = true;
        info_.inputs.push_back(std::move(pin));
        return *this;
    }

    // Add input pin with multiple frame types (variadic)
    template<typename... FrameTypes>
    ModuleRegistrationBuilder& input(const std::string& pinName, const std::string& ft1, FrameTypes... rest) {
        ModuleInfo::PinInfo pin;
        pin.name = pinName;
        pin.frame_types.push_back(ft1);
        (pin.frame_types.push_back(std::string(rest)), ...);
        pin.required = true;
        info_.inputs.push_back(std::move(pin));
        return *this;
    }

    // Add optional input pin
    ModuleRegistrationBuilder& optionalInput(const std::string& pinName, const std::string& frameType) {
        ModuleInfo::PinInfo pin;
        pin.name = pinName;
        pin.frame_types.push_back(frameType);
        pin.required = false;
        info_.inputs.push_back(std::move(pin));
        return *this;
    }

    // Add output pin with single frame type
    ModuleRegistrationBuilder& output(const std::string& pinName, const std::string& frameType) {
        ModuleInfo::PinInfo pin;
        pin.name = pinName;
        pin.frame_types.push_back(frameType);
        pin.required = true;
        info_.outputs.push_back(std::move(pin));
        return *this;
    }

    // Add output pin with multiple frame types (variadic)
    template<typename... FrameTypes>
    ModuleRegistrationBuilder& output(const std::string& pinName, const std::string& ft1, FrameTypes... rest) {
        ModuleInfo::PinInfo pin;
        pin.name = pinName;
        pin.frame_types.push_back(ft1);
        (pin.frame_types.push_back(std::string(rest)), ...);
        pin.required = true;
        info_.outputs.push_back(std::move(pin));
        return *this;
    }

    // ============================================================
    // Property definition methods
    // ============================================================

    // Add a string property
    ModuleRegistrationBuilder& stringProp(const std::string& name, const std::string& desc,
                                          bool required = false, const std::string& defaultVal = "") {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = "string";
        prop.mutability = "static";
        prop.required = required;
        prop.default_value = defaultVal;
        prop.description = desc;
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    // Add an integer property
    ModuleRegistrationBuilder& intProp(const std::string& name, const std::string& desc,
                                       bool required = false, int64_t defaultVal = 0,
                                       int64_t minVal = INT64_MIN, int64_t maxVal = INT64_MAX) {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = "int";
        prop.mutability = "static";
        prop.required = required;
        prop.default_value = std::to_string(defaultVal);
        if (minVal != INT64_MIN) prop.min_value = std::to_string(minVal);
        if (maxVal != INT64_MAX) prop.max_value = std::to_string(maxVal);
        prop.description = desc;
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    // Add a float property
    ModuleRegistrationBuilder& floatProp(const std::string& name, const std::string& desc,
                                         bool required = false, double defaultVal = 0.0,
                                         double minVal = -std::numeric_limits<double>::max(),
                                         double maxVal = std::numeric_limits<double>::max()) {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = "float";
        prop.mutability = "static";
        prop.required = required;
        prop.default_value = std::to_string(defaultVal);
        if (minVal != -std::numeric_limits<double>::max()) prop.min_value = std::to_string(minVal);
        if (maxVal != std::numeric_limits<double>::max()) prop.max_value = std::to_string(maxVal);
        prop.description = desc;
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    // Add a boolean property
    ModuleRegistrationBuilder& boolProp(const std::string& name, const std::string& desc,
                                        bool required = false, bool defaultVal = false) {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = "bool";
        prop.mutability = "static";
        prop.required = required;
        prop.default_value = defaultVal ? "true" : "false";
        prop.description = desc;
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    // Add an enum property
    template<typename... EnumValues>
    ModuleRegistrationBuilder& enumProp(const std::string& name, const std::string& desc,
                                        bool required, const std::string& defaultVal,
                                        EnumValues... values) {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = "enum";
        prop.mutability = "static";
        prop.required = required;
        prop.default_value = defaultVal;
        prop.description = desc;
        (prop.enum_values.push_back(std::string(values)), ...);
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    // Add a dynamic property (can be changed at runtime)
    ModuleRegistrationBuilder& dynamicProp(const std::string& name, const std::string& type,
                                           const std::string& desc, bool required = false,
                                           const std::string& defaultVal = "") {
        ModuleInfo::PropInfo prop;
        prop.name = name;
        prop.type = type;
        prop.mutability = "dynamic";
        prop.required = required;
        prop.default_value = defaultVal;
        prop.description = desc;
        info_.properties.push_back(std::move(prop));
        return *this;
    }

    // Destructor registers the module
    ~ModuleRegistrationBuilder() {
        if (!registered_) {
            finalize();
        }
    }

    // Explicit registration (called by destructor if not called manually)
    void finalize() {
        if (registered_) return;
        registered_ = true;

        // Set default version if not specified
        if (info_.version.empty()) {
            info_.version = "1.0";
        }

        // Create factory function that applies properties and creates module
        info_.factory = [](const std::map<std::string, ScalarPropertyValue>& props)
            -> std::unique_ptr<Module> {

            PropsClass moduleProps;

            // Apply properties using SFINAE helper (MSVC-compatible)
            // Uses function overloading instead of if constexpr
            std::vector<std::string> missingRequired;
            detail::tryApplyProperties(moduleProps, props, missingRequired);

            if (!missingRequired.empty()) {
                throw std::runtime_error(
                    "Missing required properties: " +
                    detail::joinStrings(missingRequired));
            }

            return std::make_unique<ModuleClass>(moduleProps);
        };

        // Register with the singleton registry
        ModuleRegistry::instance().registerModule(std::move(info_));
    }

    // Prevent copying
    ModuleRegistrationBuilder(const ModuleRegistrationBuilder&) = delete;
    ModuleRegistrationBuilder& operator=(const ModuleRegistrationBuilder&) = delete;

    // Allow moving
    ModuleRegistrationBuilder(ModuleRegistrationBuilder&& other) noexcept
        : info_(std::move(other.info_)), registered_(other.registered_) {
        other.registered_ = true;  // Prevent double registration
    }
};

// ============================================================
// Factory function for fluent registration
// ============================================================
template<typename ModuleClass, typename PropsClass>
ModuleRegistrationBuilder<ModuleClass, PropsClass> registerModule() {
    return ModuleRegistrationBuilder<ModuleClass, PropsClass>();
}

// Overload for modules without Props class (rare)
template<typename ModuleClass>
ModuleRegistrationBuilder<ModuleClass, ModuleProps> registerModule() {
    return ModuleRegistrationBuilder<ModuleClass, ModuleProps>();
}

} // namespace apra
