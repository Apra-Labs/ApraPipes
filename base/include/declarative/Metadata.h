// ============================================================
// File: declarative/Metadata.h
// Core metadata types for declarative pipeline construction
// ============================================================

#pragma once

#include <string_view>
#include <array>
#include <cstdint>
#include <cstddef>

namespace apra {

// ============================================================
// Constants for fixed-capacity arrays
// ============================================================
inline constexpr size_t MAX_FRAME_TYPES = 8;
inline constexpr size_t MAX_ENUM_VALUES = 16;

// ============================================================
// Module Category (Primary classification - single value)
// ============================================================
enum class ModuleCategory {
    Source,       // Produces frames (RTSP, file, camera)
    Sink,         // Consumes frames (file writer, display)
    Transform,    // Transforms frames (decoder, encoder, filter)
    Analytics,    // Analyzes frames (motion detection, object detection)
    Controller,   // Controls other modules (REST API, scheduler)
    Utility       // Helper modules (queue, tee, mux)
};

// ============================================================
// Pin Definition
// Describes an input or output pin on a module
//
// Usage:
//   constexpr auto pin = PinDef::create("input", {"RawImagePlanar", "RawImagePacked"}, true, "desc");
// Or for single frame type:
//   constexpr auto pin = PinDef::create("output", "H264Frame");
// ============================================================
struct PinDef {
    std::string_view name;
    std::array<std::string_view, MAX_FRAME_TYPES> frame_types{};
    size_t frame_type_count = 0;
    bool required = true;
    std::string_view description = "";

    // Default constructor
    constexpr PinDef() = default;

    // Access frame types count
    constexpr size_t frameTypeCount() const { return frame_type_count; }

    // Check if a frame type is accepted
    constexpr bool acceptsFrameType(std::string_view ft) const {
        for (size_t i = 0; i < frame_type_count; ++i) {
            if (frame_types[i] == ft) return true;
        }
        return false;
    }

    // Factory for single frame type
    static constexpr PinDef create(
        std::string_view name_,
        std::string_view frame_type,
        bool required_ = true,
        std::string_view description_ = ""
    ) {
        PinDef p;
        p.name = name_;
        p.frame_types[0] = frame_type;
        p.frame_type_count = 1;
        p.required = required_;
        p.description = description_;
        return p;
    }

    // Factory for two frame types
    static constexpr PinDef create(
        std::string_view name_,
        std::string_view ft1, std::string_view ft2,
        bool required_ = true,
        std::string_view description_ = ""
    ) {
        PinDef p;
        p.name = name_;
        p.frame_types[0] = ft1;
        p.frame_types[1] = ft2;
        p.frame_type_count = 2;
        p.required = required_;
        p.description = description_;
        return p;
    }

    // Factory for three frame types
    static constexpr PinDef create(
        std::string_view name_,
        std::string_view ft1, std::string_view ft2, std::string_view ft3,
        bool required_ = true,
        std::string_view description_ = ""
    ) {
        PinDef p;
        p.name = name_;
        p.frame_types[0] = ft1;
        p.frame_types[1] = ft2;
        p.frame_types[2] = ft3;
        p.frame_type_count = 3;
        p.required = required_;
        p.description = description_;
        return p;
    }

    // Factory for four frame types
    static constexpr PinDef create(
        std::string_view name_,
        std::string_view ft1, std::string_view ft2,
        std::string_view ft3, std::string_view ft4,
        bool required_ = true,
        std::string_view description_ = ""
    ) {
        PinDef p;
        p.name = name_;
        p.frame_types[0] = ft1;
        p.frame_types[1] = ft2;
        p.frame_types[2] = ft3;
        p.frame_types[3] = ft4;
        p.frame_type_count = 4;
        p.required = required_;
        p.description = description_;
        return p;
    }
};

// ============================================================
// Property Definition with Static/Dynamic distinction
// Describes a configurable property on a module
// ============================================================
struct PropDef {
    enum class Type { Int, Float, Bool, String, Enum };
    enum class Mutability {
        Static,   // Set at construction, cannot change
        Dynamic   // Can be modified at runtime via Controller
    };

    std::string_view name;
    Type type = Type::Int;
    Mutability mutability = Mutability::Static;

    // Numeric values stored directly for type safety
    int64_t int_default = 0;
    int64_t int_min = 0;
    int64_t int_max = 0;
    double float_default = 0.0;
    double float_min = 0.0;
    double float_max = 0.0;
    bool bool_default = false;

    // String values
    std::string_view string_default = "";
    std::string_view regex_pattern = "";

    // Enum values - fixed capacity array
    std::array<std::string_view, MAX_ENUM_VALUES> enum_values{};
    size_t enum_value_count = 0;

    // Documentation
    std::string_view description = "";
    std::string_view unit = "";  // e.g., "ms", "percent", "pixels"

    // Default constructor
    constexpr PropDef() = default;

    // ========================================================
    // Factory methods for clean declaration syntax
    // ========================================================

    static constexpr PropDef Int(
        std::string_view name,
        int64_t default_val,
        int64_t min_val,
        int64_t max_val,
        std::string_view desc = "",
        Mutability mut = Mutability::Static
    ) {
        PropDef p;
        p.name = name;
        p.type = Type::Int;
        p.mutability = mut;
        p.int_default = default_val;
        p.int_min = min_val;
        p.int_max = max_val;
        p.description = desc;
        return p;
    }

    static constexpr PropDef Float(
        std::string_view name,
        double default_val,
        double min_val,
        double max_val,
        std::string_view desc = "",
        Mutability mut = Mutability::Static
    ) {
        PropDef p;
        p.name = name;
        p.type = Type::Float;
        p.mutability = mut;
        p.float_default = default_val;
        p.float_min = min_val;
        p.float_max = max_val;
        p.description = desc;
        return p;
    }

    static constexpr PropDef Bool(
        std::string_view name,
        bool default_val,
        std::string_view desc = "",
        Mutability mut = Mutability::Static
    ) {
        PropDef p;
        p.name = name;
        p.type = Type::Bool;
        p.mutability = mut;
        p.bool_default = default_val;
        p.description = desc;
        return p;
    }

    static constexpr PropDef String(
        std::string_view name,
        std::string_view default_val,
        std::string_view desc = "",
        std::string_view regex = "",
        Mutability mut = Mutability::Static
    ) {
        PropDef p;
        p.name = name;
        p.type = Type::String;
        p.mutability = mut;
        p.string_default = default_val;
        p.regex_pattern = regex;
        p.description = desc;
        return p;
    }

    // Enum factory with up to 4 values (most common case)
    static constexpr PropDef Enum(
        std::string_view name,
        std::string_view default_val,
        std::string_view v1, std::string_view v2,
        std::string_view desc = "",
        Mutability mut = Mutability::Static
    ) {
        PropDef p;
        p.name = name;
        p.type = Type::Enum;
        p.mutability = mut;
        p.string_default = default_val;
        p.enum_values[0] = v1;
        p.enum_values[1] = v2;
        p.enum_value_count = 2;
        p.description = desc;
        return p;
    }

    static constexpr PropDef Enum(
        std::string_view name,
        std::string_view default_val,
        std::string_view v1, std::string_view v2, std::string_view v3,
        std::string_view desc = "",
        Mutability mut = Mutability::Static
    ) {
        PropDef p;
        p.name = name;
        p.type = Type::Enum;
        p.mutability = mut;
        p.string_default = default_val;
        p.enum_values[0] = v1;
        p.enum_values[1] = v2;
        p.enum_values[2] = v3;
        p.enum_value_count = 3;
        p.description = desc;
        return p;
    }

    static constexpr PropDef Enum(
        std::string_view name,
        std::string_view default_val,
        std::string_view v1, std::string_view v2,
        std::string_view v3, std::string_view v4,
        std::string_view desc = "",
        Mutability mut = Mutability::Static
    ) {
        PropDef p;
        p.name = name;
        p.type = Type::Enum;
        p.mutability = mut;
        p.string_default = default_val;
        p.enum_values[0] = v1;
        p.enum_values[1] = v2;
        p.enum_values[2] = v3;
        p.enum_values[3] = v4;
        p.enum_value_count = 4;
        p.description = desc;
        return p;
    }

    // ========================================================
    // Convenience: Dynamic variants
    // ========================================================

    static constexpr PropDef DynamicInt(
        std::string_view name,
        int64_t default_val,
        int64_t min_val,
        int64_t max_val,
        std::string_view desc = ""
    ) {
        return Int(name, default_val, min_val, max_val, desc, Mutability::Dynamic);
    }

    static constexpr PropDef DynamicFloat(
        std::string_view name,
        double default_val,
        double min_val,
        double max_val,
        std::string_view desc = ""
    ) {
        return Float(name, default_val, min_val, max_val, desc, Mutability::Dynamic);
    }

    static constexpr PropDef DynamicBool(
        std::string_view name,
        bool default_val,
        std::string_view desc = ""
    ) {
        return Bool(name, default_val, desc, Mutability::Dynamic);
    }

    static constexpr PropDef DynamicString(
        std::string_view name,
        std::string_view default_val,
        std::string_view desc = "",
        std::string_view regex = ""
    ) {
        return String(name, default_val, desc, regex, Mutability::Dynamic);
    }

    static constexpr PropDef DynamicEnum(
        std::string_view name,
        std::string_view default_val,
        std::string_view v1, std::string_view v2,
        std::string_view desc = ""
    ) {
        return Enum(name, default_val, v1, v2, desc, Mutability::Dynamic);
    }

    static constexpr PropDef DynamicEnum(
        std::string_view name,
        std::string_view default_val,
        std::string_view v1, std::string_view v2, std::string_view v3,
        std::string_view desc = ""
    ) {
        return Enum(name, default_val, v1, v2, v3, desc, Mutability::Dynamic);
    }
};

// ============================================================
// Frame Type Attribute Definition
// Describes attributes of a frame type (e.g., width, height for images)
// ============================================================
struct AttrDef {
    enum class Type { Int, Int64, Float, Bool, String, Enum, IntArray };

    std::string_view name;
    Type type = Type::Int;
    bool required = true;
    std::array<std::string_view, MAX_ENUM_VALUES> enum_values{};
    size_t enum_value_count = 0;
    std::string_view description = "";

    // Default constructor
    constexpr AttrDef() = default;

    // ========================================================
    // Factory methods
    // ========================================================

    static constexpr AttrDef Int(
        std::string_view name,
        bool req = true,
        std::string_view desc = ""
    ) {
        AttrDef a;
        a.name = name;
        a.type = Type::Int;
        a.required = req;
        a.description = desc;
        return a;
    }

    static constexpr AttrDef Int64(
        std::string_view name,
        bool req = true,
        std::string_view desc = ""
    ) {
        AttrDef a;
        a.name = name;
        a.type = Type::Int64;
        a.required = req;
        a.description = desc;
        return a;
    }

    static constexpr AttrDef Float(
        std::string_view name,
        bool req = true,
        std::string_view desc = ""
    ) {
        AttrDef a;
        a.name = name;
        a.type = Type::Float;
        a.required = req;
        a.description = desc;
        return a;
    }

    static constexpr AttrDef Bool(
        std::string_view name,
        bool req = true,
        std::string_view desc = ""
    ) {
        AttrDef a;
        a.name = name;
        a.type = Type::Bool;
        a.required = req;
        a.description = desc;
        return a;
    }

    static constexpr AttrDef String(
        std::string_view name,
        bool req = true,
        std::string_view desc = ""
    ) {
        AttrDef a;
        a.name = name;
        a.type = Type::String;
        a.required = req;
        a.description = desc;
        return a;
    }

    static constexpr AttrDef Enum(
        std::string_view name,
        std::string_view v1, std::string_view v2,
        bool req = true,
        std::string_view desc = ""
    ) {
        AttrDef a;
        a.name = name;
        a.type = Type::Enum;
        a.required = req;
        a.enum_values[0] = v1;
        a.enum_values[1] = v2;
        a.enum_value_count = 2;
        a.description = desc;
        return a;
    }

    static constexpr AttrDef Enum(
        std::string_view name,
        std::string_view v1, std::string_view v2,
        std::string_view v3, std::string_view v4,
        bool req = true,
        std::string_view desc = ""
    ) {
        AttrDef a;
        a.name = name;
        a.type = Type::Enum;
        a.required = req;
        a.enum_values[0] = v1;
        a.enum_values[1] = v2;
        a.enum_values[2] = v3;
        a.enum_values[3] = v4;
        a.enum_value_count = 4;
        a.description = desc;
        return a;
    }

    static constexpr AttrDef IntArray(
        std::string_view name,
        bool req = true,
        std::string_view desc = ""
    ) {
        AttrDef a;
        a.name = name;
        a.type = Type::IntArray;
        a.required = req;
        a.description = desc;
        return a;
    }
};

} // namespace apra
