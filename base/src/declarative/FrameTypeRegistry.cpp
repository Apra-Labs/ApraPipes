// ============================================================
// File: declarative/FrameTypeRegistry.cpp
// Frame Type Registry implementation
// Task A3: FrameType Registry
// ============================================================

#include "declarative/FrameTypeRegistry.h"
#include <sstream>
#include <algorithm>
#include <set>

namespace apra {

// ============================================================
// Singleton instance
// ============================================================

FrameTypeRegistry& FrameTypeRegistry::instance() {
    static FrameTypeRegistry inst;
    return inst;
}

// ============================================================
// Registration
// ============================================================

void FrameTypeRegistry::registerFrameType(FrameTypeInfo info) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Don't overwrite existing registration
    if (types_.find(info.name) != types_.end()) {
        return;
    }

    types_[info.name] = std::move(info);
    invalidateCache();
}

// ============================================================
// Basic queries
// ============================================================

bool FrameTypeRegistry::hasFrameType(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return types_.find(name) != types_.end();
}

const FrameTypeInfo* FrameTypeRegistry::getFrameType(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = types_.find(name);
    if (it != types_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<std::string> FrameTypeRegistry::getAllFrameTypes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(types_.size());
    for (const auto& [name, info] : types_) {
        names.push_back(name);
    }
    return names;
}

// ============================================================
// Tag queries
// ============================================================

std::vector<std::string> FrameTypeRegistry::getFrameTypesByTag(const std::string& tag) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;

    for (const auto& [name, info] : types_) {
        if (std::find(info.tags.begin(), info.tags.end(), tag) != info.tags.end()) {
            result.push_back(name);
        }
    }

    return result;
}

// ============================================================
// Hierarchy queries
// ============================================================

std::string FrameTypeRegistry::getParent(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = types_.find(name);
    if (it != types_.end()) {
        return it->second.parent;
    }
    return "";
}

std::vector<std::string> FrameTypeRegistry::getSubtypes(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;

    for (const auto& [typeName, info] : types_) {
        if (info.parent == name) {
            result.push_back(typeName);
        }
    }

    return result;
}

std::vector<std::string> FrameTypeRegistry::getAncestors(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check cache
    if (cacheValid_) {
        auto it = ancestorCache_.find(name);
        if (it != ancestorCache_.end()) {
            return it->second;
        }
    }

    std::vector<std::string> ancestors;
    std::string current = name;

    while (!current.empty()) {
        auto it = types_.find(current);
        if (it == types_.end()) break;

        current = it->second.parent;
        if (!current.empty()) {
            ancestors.push_back(current);
        }
    }

    // Update cache
    ancestorCache_[name] = ancestors;
    cacheValid_ = true;

    return ancestors;
}

// ============================================================
// Compatibility checks
// ============================================================

bool FrameTypeRegistry::isSubtype(const std::string& child, const std::string& parent) const {
    if (child == parent) return true;

    // Get ancestors without lock (getAncestors handles locking)
    auto ancestors = getAncestors(child);

    return std::find(ancestors.begin(), ancestors.end(), parent) != ancestors.end();
}

bool FrameTypeRegistry::isCompatible(const std::string& outputType, const std::string& inputType) const {
    // Output type is compatible with input if:
    // 1. They are the same type
    // 2. Output is a subtype of input (more specific -> more general)
    return isSubtype(outputType, inputType);
}

// ============================================================
// Export
// ============================================================

std::string FrameTypeRegistry::toJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;

    oss << "{\n  \"frameTypes\": [\n";

    bool first = true;
    for (const auto& [name, info] : types_) {
        if (!first) oss << ",\n";
        first = false;

        oss << "    {\n";
        oss << "      \"name\": \"" << info.name << "\",\n";
        oss << "      \"parent\": \"" << info.parent << "\",\n";
        oss << "      \"description\": \"" << info.description << "\",\n";

        // Tags
        oss << "      \"tags\": [";
        for (size_t i = 0; i < info.tags.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "\"" << info.tags[i] << "\"";
        }
        oss << "],\n";

        // Attributes
        oss << "      \"attributes\": [";
        for (size_t i = 0; i < info.attributes.size(); ++i) {
            if (i > 0) oss << ", ";
            const auto& attr = info.attributes[i];
            oss << "{\"name\": \"" << attr.name << "\", \"type\": \"" << attr.type
                << "\", \"required\": " << (attr.required ? "true" : "false") << "}";
        }
        oss << "]\n";

        oss << "    }";
    }

    oss << "\n  ]\n}";
    return oss.str();
}

std::string FrameTypeRegistry::toMarkdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;

    oss << "# Frame Type Hierarchy\n\n";

    // Find root types (no parent)
    std::vector<std::string> roots;
    for (const auto& [name, info] : types_) {
        if (info.parent.empty()) {
            roots.push_back(name);
        }
    }

    // Helper to print tree
    std::function<void(const std::string&, const std::string&)> printTree;
    printTree = [&](const std::string& name, const std::string& indent) {
        auto it = types_.find(name);
        if (it == types_.end()) return;

        const auto& info = it->second;
        oss << indent << "- **" << name << "**";
        if (!info.tags.empty()) {
            oss << " `";
            for (size_t i = 0; i < info.tags.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << info.tags[i];
            }
            oss << "`";
        }
        oss << "\n";

        // Find and print children
        for (const auto& [childName, childInfo] : types_) {
            if (childInfo.parent == name) {
                printTree(childName, indent + "  ");
            }
        }
    };

    for (const auto& root : roots) {
        printTree(root, "");
    }

    return oss.str();
}

// ============================================================
// Clear and size
// ============================================================

void FrameTypeRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    types_.clear();
    invalidateCache();
}

size_t FrameTypeRegistry::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return types_.size();
}

void FrameTypeRegistry::invalidateCache() const {
    ancestorCache_.clear();
    cacheValid_ = false;
}

} // namespace apra
