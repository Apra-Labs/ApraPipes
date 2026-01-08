// ============================================================
// File: declarative/ModuleRegistry.h
// Module Registry for declarative pipeline construction
// Task A2: Module Registry
// ============================================================

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <functional>
#include <variant>
#include <type_traits>
#include "declarative/Metadata.h"

// Forward declaration - real Module class from existing codebase
class Module;

namespace apra {

// ============================================================
// ScalarPropertyValue - Variant type for runtime property values
// (scalar-only version for module creation; see PipelineDescription.h
// for PropertyValue with array types used in config parsing)
// ============================================================
using ScalarPropertyValue = std::variant<
    int64_t,
    double,
    bool,
    std::string
>;

// ============================================================
// Runtime representation of module metadata
// Stores copies of metadata for runtime access
// ============================================================
struct ModuleInfo {
    std::string name;
    ModuleCategory category;
    std::string version;
    std::string description;
    std::vector<std::string> tags;

    struct PinInfo {
        std::string name;
        std::vector<std::string> frame_types;
        bool required = true;
        std::string description;
    };
    std::vector<PinInfo> inputs;
    std::vector<PinInfo> outputs;

    struct PropInfo {
        std::string name;
        std::string type;  // "int", "float", "bool", "string", "enum"
        std::string mutability;  // "static", "dynamic"
        bool required = false;  // true = mandatory (user must provide), false = optional (uses default)
        std::string default_value;
        std::string min_value;
        std::string max_value;
        std::string regex_pattern;
        std::vector<std::string> enum_values;
        std::string description;
        std::string unit;
    };
    std::vector<PropInfo> properties;

    // Factory function type - creates a module from property map
    using FactoryFn = std::function<std::unique_ptr<Module>(
        const std::map<std::string, ScalarPropertyValue>&
    )>;
    FactoryFn factory;

    // CUDA factory function type - creates a module from property map + CUDA stream
    // The void* is a pointer to cudastream_sp (type-erased to avoid CUDA header dependency)
    // Use this for modules that require cudastream_sp in their Props constructor
    using CudaFactoryFn = std::function<std::unique_ptr<Module>(
        const std::map<std::string, ScalarPropertyValue>&,
        void* cudaStreamPtr  // Points to cudastream_sp
    )>;
    CudaFactoryFn cudaFactory;

    // Flag indicating this module requires a CUDA stream
    bool requiresCudaStream = false;

    // Dynamic property accessor factory - creates type-erased accessors bound to a module instance
    // Returns {getDynamicPropertyNames, getProperty, setProperty} functions
    struct PropertyAccessors {
        std::function<std::vector<std::string>()> getDynamicPropertyNames;
        std::function<ScalarPropertyValue(const std::string&)> getProperty;
        std::function<bool(const std::string&, const ScalarPropertyValue&)> setProperty;
    };
    using PropertyAccessorFactoryFn = std::function<PropertyAccessors(Module*)>;
    PropertyAccessorFactoryFn propertyAccessorFactory;

    // Flag indicating module creates its own output pins in addInputPin()
    // When true, ModuleFactory skips output pin setup to avoid duplicates
    bool selfManagedOutputPins = false;
};

// ============================================================
// ModuleRegistry - Central singleton registry for all modules
// Thread-safe registration and queries
// ============================================================
class ModuleRegistry {
public:
    // Singleton access
    static ModuleRegistry& instance();

    // Registration (called from REGISTER_MODULE macro)
    void registerModule(ModuleInfo info);

    // Register a callback that can re-register a module after clear()
    // Called by REGISTER_MODULE macro to enable test isolation
    using RegistrationCallback = std::function<void()>;
    void addRegistrationCallback(RegistrationCallback callback);

    // Re-run all registration callbacks (call after clear() in tests)
    void rerunRegistrations();

    // Queries
    bool hasModule(const std::string& name) const;
    const ModuleInfo* getModule(const std::string& name) const;
    std::vector<std::string> getAllModules() const;
    std::vector<std::string> getModulesByCategory(ModuleCategory cat) const;
    std::vector<std::string> getModulesByTag(const std::string& tag) const;

    // Factory
    std::unique_ptr<Module> createModule(
        const std::string& name,
        const std::map<std::string, ScalarPropertyValue>& props
    ) const;

    // CUDA Factory - creates module with CUDA stream
    // cudaStreamPtr should point to a cudastream_sp
    std::unique_ptr<Module> createCudaModule(
        const std::string& name,
        const std::map<std::string, ScalarPropertyValue>& props,
        void* cudaStreamPtr
    ) const;

    // Set CUDA factory for a module (call after registerModule)
    // Used by CUDA-specific registration code
    bool setCudaFactory(const std::string& name, ModuleInfo::CudaFactoryFn factory);

    // Check if a module requires CUDA stream
    bool moduleRequiresCudaStream(const std::string& name) const;

    // Export
    std::string toJson() const;
    std::string toToml() const;

    // Clear registry (useful for testing)
    // Note: Does NOT clear registration callbacks; call rerunRegistrations() after
    void clear();

    // Get count of registered modules
    size_t size() const;

private:
    ModuleRegistry() = default;
    ~ModuleRegistry() = default;

    // Prevent copying
    ModuleRegistry(const ModuleRegistry&) = delete;
    ModuleRegistry& operator=(const ModuleRegistry&) = delete;

    std::map<std::string, ModuleInfo> modules_;
    std::vector<RegistrationCallback> registrationCallbacks_;
    mutable std::mutex mutex_;
};

// ============================================================
// Helper functions for converting Metadata types to ModuleInfo
// ============================================================
namespace detail {

// Convert PinDef to PinInfo
inline ModuleInfo::PinInfo toPinInfo(const PinDef& pin) {
    ModuleInfo::PinInfo info;
    info.name = std::string(pin.name);
    info.required = pin.required;
    info.description = std::string(pin.description);
    for (size_t i = 0; i < pin.frame_type_count; ++i) {
        info.frame_types.push_back(std::string(pin.frame_types[i]));
    }
    return info;
}

// Convert PropDef::Type to string
inline std::string propTypeToString(PropDef::Type type) {
    switch (type) {
        case PropDef::Type::Integer: return "int";
        case PropDef::Type::Floating: return "float";
        case PropDef::Type::Boolean: return "bool";
        case PropDef::Type::Text: return "string";
        case PropDef::Type::Enumeration: return "enum";
    }
    return "unknown";
}

// Convert PropDef::Mutability to string
inline std::string mutabilityToString(PropDef::Mutability mut) {
    switch (mut) {
        case PropDef::Mutability::Static: return "static";
        case PropDef::Mutability::Dynamic: return "dynamic";
    }
    return "unknown";
}

// Convert PropDef to PropInfo
inline ModuleInfo::PropInfo toPropInfo(const PropDef& prop) {
    ModuleInfo::PropInfo info;
    info.name = std::string(prop.name);
    info.type = propTypeToString(prop.type);
    info.mutability = mutabilityToString(prop.mutability);
    info.required = prop.required;
    info.description = std::string(prop.description);
    info.unit = std::string(prop.unit);
    info.regex_pattern = std::string(prop.regex_pattern);

    // Set type-specific values
    switch (prop.type) {
        case PropDef::Type::Integer:
            info.default_value = std::to_string(prop.int_default);
            info.min_value = std::to_string(prop.int_min);
            info.max_value = std::to_string(prop.int_max);
            break;
        case PropDef::Type::Floating:
            info.default_value = std::to_string(prop.float_default);
            info.min_value = std::to_string(prop.float_min);
            info.max_value = std::to_string(prop.float_max);
            break;
        case PropDef::Type::Boolean:
            info.default_value = prop.bool_default ? "true" : "false";
            break;
        case PropDef::Type::Text:
            info.default_value = std::string(prop.string_default);
            break;
        case PropDef::Type::Enumeration:
            info.default_value = std::string(prop.string_default);
            for (size_t i = 0; i < prop.enum_value_count; ++i) {
                info.enum_values.push_back(std::string(prop.enum_values[i]));
            }
            break;
    }
    return info;
}

// Convert ModuleCategory to string
inline std::string categoryToString(ModuleCategory cat) {
    switch (cat) {
        case ModuleCategory::Source: return "source";
        case ModuleCategory::Sink: return "sink";
        case ModuleCategory::Transform: return "transform";
        case ModuleCategory::Analytics: return "analytics";
        case ModuleCategory::Controller: return "controller";
        case ModuleCategory::Utility: return "utility";
    }
    return "unknown";
}

// ============================================================
// Type trait to detect if PropsClass has applyProperties method
// Used by REGISTER_MODULE macro to apply TOML properties
// ============================================================
template<typename T, typename = void>
struct has_apply_properties : std::false_type {};

template<typename T>
struct has_apply_properties<T,
    std::void_t<decltype(T::applyProperties(
        std::declval<T&>(),
        std::declval<const std::map<std::string, ScalarPropertyValue>&>(),
        std::declval<std::vector<std::string>&>()
    ))>
> : std::true_type {};

template<typename T>
inline constexpr bool has_apply_properties_v = has_apply_properties<T>::value;

// ============================================================
// Helper function template to apply properties using SFINAE
// MSVC has issues with if constexpr in lambdas within macros,
// so we use function overloading with enable_if instead.
// ============================================================
template<typename T>
inline auto tryApplyProperties(
    T& props,
    const std::map<std::string, ScalarPropertyValue>& propMap,
    std::vector<std::string>& missing
) -> std::enable_if_t<has_apply_properties_v<T>, void>
{
    T::applyProperties(props, propMap, missing);
}

template<typename T>
inline auto tryApplyProperties(
    T& props,
    const std::map<std::string, ScalarPropertyValue>&,
    std::vector<std::string>&
) -> std::enable_if_t<!has_apply_properties_v<T>, void>
{
    // No-op for types without applyProperties method
    // Module will use default-constructed props
}

} // namespace detail

// ============================================================
// REGISTER_MODULE Macro
// Registers a module at static initialization time
//
// Usage in .cpp file:
//   REGISTER_MODULE(FileReaderModule, FileReaderModuleProps)
//
// Requirements:
//   - ModuleClass must have a nested `Metadata` struct with:
//     - static constexpr std::string_view name
//     - static constexpr ModuleCategory category
//     - static constexpr std::string_view version
//     - static constexpr std::string_view description
//     - static constexpr std::array<std::string_view, N> tags
//     - static constexpr std::array<PinDef, N> inputs
//     - static constexpr std::array<PinDef, N> outputs
//     - static constexpr std::array<PropDef, N> properties
// ============================================================
#define REGISTER_MODULE(ModuleClass, PropsClass) \
    static_assert(std::is_default_constructible<PropsClass>::value, \
        "REGISTER_MODULE requires " #PropsClass " to have a default constructor. " \
        "Add a default constructor to " #PropsClass " that initializes all members " \
        "with sensible defaults. Required properties can be marked with PropDef::Required*()."); \
    namespace apra { namespace _reg_##ModuleClass { \
        inline void registerIfNeeded() { \
            auto& registry = ModuleRegistry::instance(); \
            const std::string moduleName = std::string(ModuleClass::Metadata::name); \
            if (registry.hasModule(moduleName)) { \
                return; /* Already registered */ \
            } \
            \
            ModuleInfo info; \
            info.name = moduleName; \
            info.category = ModuleClass::Metadata::category; \
            info.version = std::string(ModuleClass::Metadata::version); \
            info.description = std::string(ModuleClass::Metadata::description); \
            \
            /* Copy tags */ \
            for (const auto& tag : ModuleClass::Metadata::tags) { \
                info.tags.push_back(std::string(tag)); \
            } \
            \
            /* Copy inputs */ \
            for (const auto& pin : ModuleClass::Metadata::inputs) { \
                info.inputs.push_back(detail::toPinInfo(pin)); \
            } \
            \
            /* Copy outputs */ \
            for (const auto& pin : ModuleClass::Metadata::outputs) { \
                info.outputs.push_back(detail::toPinInfo(pin)); \
            } \
            \
            /* Copy properties */ \
            for (const auto& prop : ModuleClass::Metadata::properties) { \
                info.properties.push_back(detail::toPropInfo(prop)); \
            } \
            \
            /* Factory function - creates module with props */ \
            info.factory = [](const std::map<std::string, ScalarPropertyValue>& props) \
                -> std::unique_ptr<Module> { \
                PropsClass moduleProps; \
                /* Apply properties using SFINAE helper (MSVC-compatible) */ \
                std::vector<std::string> missingRequired; \
                detail::tryApplyProperties(moduleProps, props, missingRequired); \
                if (!missingRequired.empty()) { \
                    std::string msg = "Missing required properties: "; \
                    for (size_t i = 0; i < missingRequired.size(); ++i) { \
                        if (i > 0) msg += ", "; \
                        msg += missingRequired[i]; \
                    } \
                    throw std::runtime_error(msg); \
                } \
                return std::make_unique<ModuleClass>(moduleProps); \
            }; \
            \
            registry.registerModule(std::move(info)); \
        } \
        static const int _trigger = []() { \
            registerIfNeeded(); \
            ModuleRegistry::instance().addRegistrationCallback(registerIfNeeded); \
            return 0; \
        }(); \
    }}

} // namespace apra
