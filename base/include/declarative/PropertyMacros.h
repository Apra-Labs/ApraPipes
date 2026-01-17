// ============================================================
// File: declarative/PropertyMacros.h
// Task D2: Property Binding System
//
// Provides DECLARE_PROPS macro for DRY property definitions.
// Single definition generates: members, metadata, apply, get, set.
// ============================================================

#pragma once

#include "Metadata.h"
#include <map>
#include <string>
#include <variant>
#include <vector>
#include <type_traits>
#include <stdexcept>
#include <cstdint>

namespace apra {

// ============================================================
// ScalarPropertyValue - variant type for property values
// (Also defined in ModuleRegistry.h, but we need it here too)
// ============================================================
#ifndef APRA_SCALAR_PROPERTY_VALUE_DEFINED
#define APRA_SCALAR_PROPERTY_VALUE_DEFINED
using ScalarPropertyValue = std::variant<int64_t, double, bool, std::string>;
#endif

// ============================================================
// Type-safe property application with automatic conversion
// Applies a value from the map to a member variable
// ============================================================
template<typename T>
inline void applyProp(
    T& member,
    const char* propName,
    const std::map<std::string, ScalarPropertyValue>& values,
    bool isRequired,
    std::vector<std::string>& missingRequired
) {
    auto it = values.find(propName);
    if (it == values.end()) {
        if (isRequired) {
            missingRequired.push_back(propName);
        }
        return;  // Keep default value
    }

    std::visit([&member, propName](auto&& val) {
        using V = std::decay_t<decltype(val)>;

        if constexpr (std::is_same_v<T, std::string>) {
            if constexpr (std::is_same_v<V, std::string>) {
                member = val;
            }
        }
        else if constexpr (std::is_same_v<T, bool>) {
            if constexpr (std::is_same_v<V, bool>) {
                member = val;
            }
        }
        else if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_same_v<V, int64_t>) {
                member = static_cast<T>(val);
            }
        }
        else if constexpr (std::is_floating_point_v<T>) {
            if constexpr (std::is_same_v<V, double>) {
                member = static_cast<T>(val);
            }
            else if constexpr (std::is_same_v<V, int64_t>) {
                member = static_cast<T>(val);
            }
        }
    }, it->second);
}

// ============================================================
// Convert member to ScalarPropertyValue for getProperty()
// ============================================================
template<typename T>
inline ScalarPropertyValue toPropertyValue(const T& member) {
    if constexpr (std::is_same_v<T, std::string>) {
        return member;
    }
    else if constexpr (std::is_same_v<T, bool>) {
        return member;
    }
    else if constexpr (std::is_integral_v<T>) {
        return static_cast<int64_t>(member);
    }
    else if constexpr (std::is_floating_point_v<T>) {
        return static_cast<double>(member);
    }
    else {
        // Unsupported type - return empty string as fallback
        return std::string("");
    }
}

// ============================================================
// Apply value from variant to member for setProperty()
// Returns true if successfully applied
// ============================================================
template<typename T>
inline bool applyFromVariant(T& member, const ScalarPropertyValue& value) {
    return std::visit([&member](auto&& val) -> bool {
        using V = std::decay_t<decltype(val)>;

        if constexpr (std::is_same_v<T, std::string> && std::is_same_v<V, std::string>) {
            member = val;
            return true;
        }
        else if constexpr (std::is_same_v<T, bool> && std::is_same_v<V, bool>) {
            member = val;
            return true;
        }
        else if constexpr (std::is_integral_v<T> && std::is_same_v<V, int64_t>) {
            member = static_cast<T>(val);
            return true;
        }
        else if constexpr (std::is_floating_point_v<T> && std::is_same_v<V, double>) {
            member = static_cast<T>(val);
            return true;
        }
        else if constexpr (std::is_floating_point_v<T> && std::is_same_v<V, int64_t>) {
            member = static_cast<T>(val);
            return true;
        }
        return false;
    }, value);
}

// ============================================================
// PropertyInfo - runtime property metadata
// ============================================================
struct PropertyInfo {
    std::string name;
    std::string type;
    std::string description;
    bool required = false;
    bool dynamic = false;  // true = can change at runtime
};

} // namespace apra

// ============================================================
// X-Macro Helpers
// ============================================================

// Mutability markers (matches PropDef::Mutability in Metadata.h)
#define APRA_PROP_MUT_Static  false
#define APRA_PROP_MUT_Dynamic true

// Requirement markers
#define APRA_PROP_REQ_Required true
#define APRA_PROP_REQ_Optional false

// ============================================================
// PROP_DECL_MEMBER - generates member variable with default
// P(type, name, mutability, requirement, default, description)
// ============================================================
#define APRA_PROP_DECL_MEMBER(type, name, mut, req, def, desc) \
    type name = def;

// ============================================================
// PROP_INFO - generates PropertyInfo entry
// ============================================================
#define APRA_PROP_INFO(type, name, mut, req, def, desc) \
    { #name, #type, desc, APRA_PROP_REQ_##req, APRA_PROP_MUT_##mut },

// ============================================================
// PROP_APPLY - generates applyProp call
// ============================================================
#define APRA_PROP_APPLY(type, name, mut, req, def, desc) \
    apra::applyProp(props.name, #name, values, APRA_PROP_REQ_##req, missingRequired);

// ============================================================
// PROP_GET - generates getProperty case
// ============================================================
#define APRA_PROP_GET(type, name, mut, req, def, desc) \
    if (propName == #name) return apra::toPropertyValue(name);

// ============================================================
// PROP_SET - generates setProperty case with mutability check
// ============================================================
#define APRA_PROP_SET_IMPL_false(type, name, desc) \
    if (propName == #name) { \
        throw std::runtime_error("Cannot modify static property '" #name "' after initialization"); \
    }

#define APRA_PROP_SET_IMPL_true(type, name, desc) \
    if (propName == #name) { \
        if (apra::applyFromVariant(name, value)) return true; \
        throw std::runtime_error("Type mismatch for property '" #name "'"); \
    }

// Three-level indirection to force expansion before token pasting
#define APRA_PROP_SET_CALL(impl, type, name, desc) impl(type, name, desc)
#define APRA_PROP_SET_CONCAT(x, y) x ## y
#define APRA_PROP_SET_DISPATCH(isDynamic, type, name, desc) \
    APRA_PROP_SET_CALL(APRA_PROP_SET_CONCAT(APRA_PROP_SET_IMPL_, isDynamic), type, name, desc)

#define APRA_PROP_SET(type, name, mut, req, def, desc) \
    APRA_PROP_SET_DISPATCH(APRA_PROP_MUT_##mut, type, name, desc)

// ============================================================
// PROP_DYN_NAME - generates dynamic property name if applicable
// ============================================================
#define APRA_PROP_DYN_NAME_false(name)
#define APRA_PROP_DYN_NAME_true(name) names.push_back(#name);

// Three-level indirection to force expansion before token pasting
#define APRA_PROP_DYN_CALL(impl, name) impl(name)
#define APRA_PROP_DYN_CONCAT(x, y) x ## y
#define APRA_PROP_DYN_DISPATCH(isDynamic, name) \
    APRA_PROP_DYN_CALL(APRA_PROP_DYN_CONCAT(APRA_PROP_DYN_NAME_, isDynamic), name)

#define APRA_PROP_DYN_NAME(type, name, mut, req, def, desc) \
    APRA_PROP_DYN_DISPATCH(APRA_PROP_MUT_##mut, name)

// ============================================================
// DECLARE_PROPS - Main macro that generates everything
//
// Usage:
//   #define MY_PROPS(P) \
//       P(std::string, path,   Static,  Required, "",    "File path") \
//       P(bool,        loop,   Static,  Optional, false, "Loop playback") \
//       P(float,       scale,  Dynamic, Optional, 1.0f,  "Scale factor")
//
//   class MyModuleProps : public ModuleProps {
//   public:
//       DECLARE_PROPS(MY_PROPS)
//       MyModuleProps() : ModuleProps() {}
//   };
// ============================================================
#define DECLARE_PROPS(PROPS_MACRO) \
    /* 1. Declare member variables with defaults */ \
    PROPS_MACRO(APRA_PROP_DECL_MEMBER) \
    \
    /* 2. Get property metadata for introspection */ \
    static std::vector<apra::PropertyInfo> getPropertyInfos() { \
        return { \
            PROPS_MACRO(APRA_PROP_INFO) \
        }; \
    } \
    \
    /* 3. Apply properties from TOML/map at construction */ \
    template<typename PropsT> \
    static void applyProperties( \
        PropsT& props, \
        const std::map<std::string, apra::ScalarPropertyValue>& values, \
        std::vector<std::string>& missingRequired \
    ) { \
        PROPS_MACRO(APRA_PROP_APPLY) \
    } \
    \
    /* 4. Get property by name (runtime introspection) */ \
    apra::ScalarPropertyValue getProperty(const std::string& propName) const { \
        PROPS_MACRO(APRA_PROP_GET) \
        throw std::runtime_error("Unknown property: " + propName); \
    } \
    \
    /* 5. Set property by name (only Dynamic properties allowed) */ \
    bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) { \
        PROPS_MACRO(APRA_PROP_SET) \
        throw std::runtime_error("Unknown property: " + propName); \
    } \
    \
    /* 6. Get list of dynamic (runtime-modifiable) property names */ \
    static std::vector<std::string> dynamicPropertyNames() { \
        std::vector<std::string> names; \
        PROPS_MACRO(APRA_PROP_DYN_NAME) \
        return names; \
    } \
    \
    /* 7. Check if a property is dynamic */ \
    static bool isPropertyDynamic(const std::string& propName) { \
        auto names = dynamicPropertyNames(); \
        return std::find(names.begin(), names.end(), propName) != names.end(); \
    }

// End of PropertyMacros.h
