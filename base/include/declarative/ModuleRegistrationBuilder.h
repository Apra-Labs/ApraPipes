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

    // ============================================================
    // SFINAE helpers for detecting dynamic property support on Props class
    // ============================================================

    // Check if PropsClass has static dynamicPropertyNames()
    template<typename P, typename = void>
    struct has_dynamic_property_names : std::false_type {};

    template<typename P>
    struct has_dynamic_property_names<P,
        std::void_t<decltype(P::dynamicPropertyNames())>> : std::true_type {};

    // Check if PropsClass has getProperty(string) const
    template<typename P, typename = void>
    struct has_get_property : std::false_type {};

    template<typename P>
    struct has_get_property<P,
        std::void_t<decltype(std::declval<const P&>().getProperty(std::string{}))>> : std::true_type {};

    // Check if PropsClass has setProperty(string, value)
    template<typename P, typename = void>
    struct has_set_property : std::false_type {};

    template<typename P>
    struct has_set_property<P,
        std::void_t<decltype(std::declval<P&>().setProperty(std::string{}, ScalarPropertyValue{}))>> : std::true_type {};

    // Check if ModuleClass has getProps() and setProps(PropsClass&)
    template<typename M, typename P, typename = void>
    struct has_get_set_props : std::false_type {};

    template<typename M, typename P>
    struct has_get_set_props<M, P,
        std::void_t<
            decltype(std::declval<M&>().getProps()),
            decltype(std::declval<M&>().setProps(std::declval<P&>()))
        >> : std::true_type {};

    // Combined check: module supports dynamic properties if all methods exist
    template<typename M, typename P>
    constexpr bool supports_dynamic_props_v =
        has_dynamic_property_names<P>::value &&
        has_get_property<P>::value &&
        has_set_property<P>::value &&
        has_get_set_props<M, P>::value;
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
    ModuleRegistrationBuilder& input(const std::string& pinName, const std::string& frameType,
                                     MemType memType = FrameMetadata::HOST) {
        ModuleInfo::PinInfo pin;
        pin.name = pinName;
        pin.frame_types.push_back(frameType);
        pin.required = true;
        pin.memType = memType;
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

    // Add input pin with memType and multiple frame types
    ModuleRegistrationBuilder& inputWithMemType(const std::string& pinName, MemType memType,
                                                 const std::string& frameType) {
        return input(pinName, frameType, memType);
    }

    // Add CUDA input pin (convenience for CUDA_DEVICE memType)
    ModuleRegistrationBuilder& cudaInput(const std::string& pinName, const std::string& frameType) {
        return input(pinName, frameType, FrameMetadata::CUDA_DEVICE);
    }

    // Add optional input pin
    ModuleRegistrationBuilder& optionalInput(const std::string& pinName, const std::string& frameType,
                                             MemType memType = FrameMetadata::HOST) {
        ModuleInfo::PinInfo pin;
        pin.name = pinName;
        pin.frame_types.push_back(frameType);
        pin.required = false;
        pin.memType = memType;
        info_.inputs.push_back(std::move(pin));
        return *this;
    }

    // Add output pin with single frame type
    ModuleRegistrationBuilder& output(const std::string& pinName, const std::string& frameType,
                                      MemType memType = FrameMetadata::HOST) {
        ModuleInfo::PinInfo pin;
        pin.name = pinName;
        pin.frame_types.push_back(frameType);
        pin.required = true;
        pin.memType = memType;
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

    // Add output pin with memType and multiple frame types
    ModuleRegistrationBuilder& outputWithMemType(const std::string& pinName, MemType memType,
                                                  const std::string& frameType) {
        return output(pinName, frameType, memType);
    }

    // Add CUDA output pin (convenience for CUDA_DEVICE memType)
    ModuleRegistrationBuilder& cudaOutput(const std::string& pinName, const std::string& frameType) {
        return output(pinName, frameType, FrameMetadata::CUDA_DEVICE);
    }

    // ============================================================
    // Image Type methods (set allowed pixel formats on last-added pin)
    // ============================================================

    // Set image types on the last input pin
    template<typename... ImageTypes>
    ModuleRegistrationBuilder& inputImageTypes(ImageTypes... types) {
        if (!info_.inputs.empty()) {
            (info_.inputs.back().image_types.push_back(types), ...);
        }
        return *this;
    }

    // Set image types on the last output pin
    template<typename... ImageTypes>
    ModuleRegistrationBuilder& outputImageTypes(ImageTypes... types) {
        if (!info_.outputs.empty()) {
            (info_.outputs.back().image_types.push_back(types), ...);
        }
        return *this;
    }

    // Add input pin with specific image types
    template<typename... ImageTypes>
    ModuleRegistrationBuilder& inputWithImageTypes(const std::string& pinName, const std::string& frameType,
                                                    MemType memType, ImageTypes... types) {
        input(pinName, frameType, memType);
        inputImageTypes(types...);
        return *this;
    }

    // Add output pin with specific image types
    template<typename... ImageTypes>
    ModuleRegistrationBuilder& outputWithImageTypes(const std::string& pinName, const std::string& frameType,
                                                     MemType memType, ImageTypes... types) {
        output(pinName, frameType, memType);
        outputImageTypes(types...);
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

    // Mark module as managing its own output pins (creates them in addInputPin)
    // This prevents ModuleFactory from pre-creating output pins
    ModuleRegistrationBuilder& selfManagedOutputPins() {
        info_.selfManagedOutputPins = true;
        return *this;
    }

    // Mark module as requiring a CUDA stream (cudastream_sp)
    // Modules with this flag will have their cudaFactory called instead of factory
    // The cudaFactory receives a void* pointing to the cudastream_sp
    ModuleRegistrationBuilder& cudaStreamRequired() {
        info_.requiresCudaStream = true;
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
        // Note: Explicit ModuleInfo::FactoryFn cast required for GCC 9 compatibility
        // (GCC 9 has issues with lambda-to-std::function conversion in templates)
        info_.factory = ModuleInfo::FactoryFn(
            [](const std::map<std::string, ScalarPropertyValue>& props)
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
        });

        // Create property accessor factory for modules that support dynamic props
        createPropertyAccessorFactory();

        // Register with the singleton registry
        ModuleRegistry::instance().registerModule(std::move(info_));
    }

private:
    // Helper to create property accessor factory (SFINAE-enabled)
    // Note: Explicit ModuleInfo::PropertyAccessorFactoryFn cast required for GCC 9 compatibility
    template<typename M = ModuleClass, typename P = PropsClass>
    typename std::enable_if<detail::supports_dynamic_props_v<M, P>, void>::type
    createPropertyAccessorFactory() {
        info_.propertyAccessorFactory = ModuleInfo::PropertyAccessorFactoryFn(
            [](Module* rawModule) -> ModuleInfo::PropertyAccessors {
            ModuleInfo::PropertyAccessors accessors;

            // Cast to concrete module type
            auto* typedModule = static_cast<ModuleClass*>(rawModule);

            // Get list of dynamic property names (static on props class)
            accessors.getDynamicPropertyNames = []() -> std::vector<std::string> {
                return PropsClass::dynamicPropertyNames();
            };

            // Get property value from current props
            accessors.getProperty = [typedModule](const std::string& name) -> ScalarPropertyValue {
                PropsClass props = typedModule->getProps();
                return props.getProperty(name);
            };

            // Set property value: get current props, modify, apply back
            accessors.setProperty = [typedModule](const std::string& name,
                                                   const ScalarPropertyValue& value) -> bool {
                PropsClass props = typedModule->getProps();
                bool success = props.setProperty(name, value);
                if (success) {
                    typedModule->setProps(props);
                }
                return success;
            };

            return accessors;
        });
    }

    // Fallback for modules without dynamic property support
    template<typename M = ModuleClass, typename P = PropsClass>
    typename std::enable_if<!detail::supports_dynamic_props_v<M, P>, void>::type
    createPropertyAccessorFactory() {
        // No-op: leave propertyAccessorFactory as nullptr
    }

public:

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

// ============================================================
// CUDA Module Registration Helper
// Use this to set the cudaFactory for modules that require cudastream_sp
// Must be called after the module is registered via registerModule()
//
// Usage (in ModuleRegistrations.cpp, inside #ifdef ENABLE_CUDA):
//   #include "CudaCommon.h"
//   setCudaModuleFactory<GaussianBlur, GaussianBlurProps>(
//       [](const auto& props, cudastream_sp stream) {
//           GaussianBlurProps moduleProps(stream);
//           // apply props...
//           return std::make_unique<GaussianBlur>(moduleProps);
//       });
// ============================================================
template<typename ModuleClass, typename PropsClass, typename CudaFactoryLambda>
void setCudaModuleFactory(const std::string& moduleName, CudaFactoryLambda&& cudaFactory) {
    auto& registry = ModuleRegistry::instance();
    // Note: We can't directly modify the registry entry, so we need
    // a different approach. See ModuleRegistrations.cpp for the pattern.
    // This function is kept for documentation purposes.
    (void)cudaFactory;  // Suppress unused warning
}

} // namespace apra
