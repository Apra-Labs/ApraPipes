// ============================================================
// File: declarative/FrameTypeRegistry.h
// Frame Type Registry for declarative pipeline validation
// Task A3: FrameType Registry
// ============================================================

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <array>
#include <mutex>

namespace apra {

// ============================================================
// FrameTypeDef - Compile-time frame type definition
// Used in REGISTER_FRAME_TYPE macro
// ============================================================
struct FrameTypeDef {
    std::string_view name;
    std::string_view parent;        // Empty for root types
    std::string_view description;

    // Tags for categorization
    static constexpr size_t MAX_TAGS = 8;
    std::array<std::string_view, MAX_TAGS> tags = {};
    size_t tag_count = 0;

    // Attributes (metadata fields on this frame type)
    struct AttrDef {
        std::string_view name;
        std::string_view type;      // "int", "float", "bool", "string"
        bool required = true;
        std::string_view description;
    };
    static constexpr size_t MAX_ATTRS = 16;
    std::array<AttrDef, MAX_ATTRS> attributes = {};
    size_t attr_count = 0;

    // Factory methods
    template<size_t N>
    static constexpr FrameTypeDef create(
        std::string_view name,
        std::string_view parent,
        std::string_view description,
        const std::array<std::string_view, N>& tagArr
    ) {
        FrameTypeDef def;
        def.name = name;
        def.parent = parent;
        def.description = description;
        def.tag_count = N;
        for (size_t i = 0; i < N && i < MAX_TAGS; ++i) {
            def.tags[i] = tagArr[i];
        }
        return def;
    }
};

// ============================================================
// FrameTypeInfo - Runtime representation of frame type
// ============================================================
struct FrameTypeInfo {
    std::string name;
    std::string parent;         // Empty for root types
    std::string description;
    std::vector<std::string> tags;

    struct AttrInfo {
        std::string name;
        std::string type;       // "int", "float", "bool", "string", "enum"
        bool required = true;
        std::vector<std::string> enum_values;
        std::string description;
    };
    std::vector<AttrInfo> attributes;
};

// ============================================================
// FrameTypeRegistry - Central registry for frame types
// Thread-safe singleton with hierarchy queries
// ============================================================
class FrameTypeRegistry {
public:
    // Singleton access
    static FrameTypeRegistry& instance();

    // Registration
    void registerFrameType(FrameTypeInfo info);

    // Basic queries
    bool hasFrameType(const std::string& name) const;
    const FrameTypeInfo* getFrameType(const std::string& name) const;
    std::vector<std::string> getAllFrameTypes() const;

    // Tag queries
    std::vector<std::string> getFrameTypesByTag(const std::string& tag) const;

    // Hierarchy queries
    std::string getParent(const std::string& name) const;
    std::vector<std::string> getSubtypes(const std::string& name) const;
    std::vector<std::string> getAncestors(const std::string& name) const;

    // Compatibility checks (for validator)
    bool isSubtype(const std::string& child, const std::string& parent) const;
    bool isCompatible(const std::string& outputType, const std::string& inputType) const;

    // Export
    std::string toJson() const;
    std::string toMarkdown() const;  // Hierarchy diagram

    // Clear registry (for testing)
    void clear();

    // Get count
    size_t size() const;

private:
    FrameTypeRegistry() = default;
    ~FrameTypeRegistry() = default;

    // Prevent copying
    FrameTypeRegistry(const FrameTypeRegistry&) = delete;
    FrameTypeRegistry& operator=(const FrameTypeRegistry&) = delete;

    std::map<std::string, FrameTypeInfo> types_;
    mutable std::mutex mutex_;

    // Cache for hierarchy queries (invalidated on registration)
    mutable std::map<std::string, std::vector<std::string>> ancestorCache_;
    mutable bool cacheValid_ = false;

    void invalidateCache() const;
};

// ============================================================
// Helper to convert FrameTypeDef to FrameTypeInfo
// ============================================================
namespace detail {

inline FrameTypeInfo toFrameTypeInfo(const FrameTypeDef& def) {
    FrameTypeInfo info;
    info.name = std::string(def.name);
    info.parent = std::string(def.parent);
    info.description = std::string(def.description);

    for (size_t i = 0; i < def.tag_count && i < FrameTypeDef::MAX_TAGS; ++i) {
        info.tags.push_back(std::string(def.tags[i]));
    }

    for (size_t i = 0; i < def.attr_count && i < FrameTypeDef::MAX_ATTRS; ++i) {
        FrameTypeInfo::AttrInfo attr;
        attr.name = std::string(def.attributes[i].name);
        attr.type = std::string(def.attributes[i].type);
        attr.required = def.attributes[i].required;
        attr.description = std::string(def.attributes[i].description);
        info.attributes.push_back(attr);
    }

    return info;
}

} // namespace detail

// ============================================================
// REGISTER_FRAME_TYPE Macro
// Registers a frame type at static initialization time
//
// Usage:
//   REGISTER_FRAME_TYPE(H264Frame)
//
// Requirements:
//   - FrameClass must have a nested `Metadata` struct with:
//     - static constexpr std::string_view name
//     - static constexpr std::string_view parent
//     - static constexpr std::string_view description
//     - static constexpr std::array<std::string_view, N> tags
// ============================================================
#define REGISTER_FRAME_TYPE(FrameClass) \
    namespace { \
        static bool _registered_ft_##FrameClass = []() { \
            apra::FrameTypeInfo info; \
            info.name = std::string(FrameClass::Metadata::name); \
            info.parent = std::string(FrameClass::Metadata::parent); \
            info.description = std::string(FrameClass::Metadata::description); \
            \
            for (const auto& tag : FrameClass::Metadata::tags) { \
                info.tags.push_back(std::string(tag)); \
            } \
            \
            apra::FrameTypeRegistry::instance().registerFrameType(std::move(info)); \
            return true; \
        }(); \
    }

} // namespace apra
