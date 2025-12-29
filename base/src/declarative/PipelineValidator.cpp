// ============================================================
// File: declarative/PipelineValidator.cpp
// Pipeline Validator Shell implementation
// Task C1: Validator Shell
// ============================================================

#include "declarative/PipelineValidator.h"
#include "declarative/ModuleRegistry.h"
#include <sstream>

namespace apra {

// ============================================================
// Result formatting
// ============================================================

std::string PipelineValidator::Result::format() const {
    std::ostringstream oss;

    for (const auto& issue : issues) {
        switch (issue.level) {
            case ValidationIssue::Level::Error:
                oss << "[ERROR] ";
                break;
            case ValidationIssue::Level::Warning:
                oss << "[WARN]  ";
                break;
            case ValidationIssue::Level::Info:
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
        result.issues.push_back(ValidationIssue::info(
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
        result.issues.push_back(ValidationIssue::info(
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
// Phase 1: Module validation
// ============================================================

PipelineValidator::Result PipelineValidator::validateModules(const PipelineDescription& desc) const {
    Result result;

    // TODO (C2): Implement full module validation
    // - Check each module type exists in registry
    // - Check module category constraints
    // - Warn on version mismatch

    // MVP: Just log what we found
    for (const auto& module : desc.modules) {
        if (options_.includeInfoMessages) {
            result.issues.push_back(ValidationIssue::info(
                "I010",
                "modules." + module.instance_id,
                "Found module: " + module.module_type
            ));
        }

        // Basic check: verify module type exists in registry
        // (Note: This will become more thorough in C2)
        auto& registry = ModuleRegistry::instance();
        if (!registry.hasModule(module.module_type)) {
            // For now, just info - will be error in C2
            if (options_.includeInfoMessages) {
                result.issues.push_back(ValidationIssue::info(
                    "I011",
                    "modules." + module.instance_id,
                    "Module type not in registry (may not be registered yet): " + module.module_type
                ));
            }
        }
    }

    return result;
}

// ============================================================
// Phase 2: Property validation
// ============================================================

PipelineValidator::Result PipelineValidator::validateProperties(const PipelineDescription& desc) const {
    Result result;

    // TODO (C3): Implement full property validation
    // - Check property names exist in module metadata
    // - Check property types match
    // - Check ranges (min/max)
    // - Check regex patterns
    // - Check enum values
    // - Warn on missing optional properties

    // MVP: Just count properties
    for (const auto& module : desc.modules) {
        if (options_.includeInfoMessages && !module.properties.empty()) {
            result.issues.push_back(ValidationIssue::info(
                "I020",
                "modules." + module.instance_id,
                "Module has " + std::to_string(module.properties.size()) + " properties"
            ));
        }
    }

    return result;
}

// ============================================================
// Phase 3: Connection validation
// ============================================================

PipelineValidator::Result PipelineValidator::validateConnections(const PipelineDescription& desc) const {
    Result result;

    // TODO (C4): Implement full connection validation
    // - Check source/dest modules exist
    // - Check pins exist on modules
    // - Check frame type compatibility
    // - Check for duplicate connections to same input
    // - Check required pins are connected

    // MVP: Just count and log connections
    if (options_.includeInfoMessages && !desc.connections.empty()) {
        result.issues.push_back(ValidationIssue::info(
            "I030",
            "connections",
            "Pipeline has " + std::to_string(desc.connections.size()) + " connections"
        ));
    }

    // Basic check: verify referenced modules exist in the description
    for (const auto& conn : desc.connections) {
        bool foundSource = false;
        bool foundDest = false;

        for (const auto& module : desc.modules) {
            if (module.instance_id == conn.from_module) foundSource = true;
            if (module.instance_id == conn.to_module) foundDest = true;
        }

        if (!foundSource && options_.includeInfoMessages) {
            result.issues.push_back(ValidationIssue::info(
                "I031",
                "connections",
                "Source module not found: " + conn.from_module
            ));
        }

        if (!foundDest && options_.includeInfoMessages) {
            result.issues.push_back(ValidationIssue::info(
                "I032",
                "connections",
                "Destination module not found: " + conn.to_module
            ));
        }
    }

    return result;
}

// ============================================================
// Phase 4: Graph validation
// ============================================================

PipelineValidator::Result PipelineValidator::validateGraph(const PipelineDescription& desc) const {
    Result result;

    // TODO (C5): Implement full graph validation
    // - Check pipeline has at least one source
    // - Check graph is DAG (no cycles)
    // - Warn on orphan modules

    // MVP: Basic checks
    if (desc.modules.empty()) {
        if (options_.includeInfoMessages) {
            result.issues.push_back(ValidationIssue::info(
                "I040",
                "pipeline",
                "Pipeline has no modules"
            ));
        }
    }

    return result;
}

} // namespace apra
