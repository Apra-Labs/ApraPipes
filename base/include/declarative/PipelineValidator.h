// ============================================================
// File: declarative/PipelineValidator.h
// Pipeline Validator Shell (Framework)
// Task C1: Validator Shell
// ============================================================

#pragma once

#include "declarative/PipelineDescription.h"
#include "declarative/Issue.h"
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <iterator>

namespace apra {

// ============================================================
// PipelineValidator - Validates PipelineDescription
// ============================================================
class PipelineValidator {
public:
    // Validation result
    struct Result {
        std::vector<ValidationIssue> issues;

        bool hasErrors() const {
            return std::any_of(issues.begin(), issues.end(),
                [](const ValidationIssue& i) { return i.level == ValidationIssue::Level::Error; });
        }

        bool hasWarnings() const {
            return std::any_of(issues.begin(), issues.end(),
                [](const ValidationIssue& i) { return i.level == ValidationIssue::Level::Warning; });
        }

        std::vector<ValidationIssue> errors() const {
            std::vector<ValidationIssue> result;
            std::copy_if(issues.begin(), issues.end(), std::back_inserter(result),
                [](const ValidationIssue& i) { return i.level == ValidationIssue::Level::Error; });
            return result;
        }

        std::vector<ValidationIssue> warnings() const {
            std::vector<ValidationIssue> result;
            std::copy_if(issues.begin(), issues.end(), std::back_inserter(result),
                [](const ValidationIssue& i) { return i.level == ValidationIssue::Level::Warning; });
            return result;
        }

        // Merge results from multiple phases
        void merge(const Result& other) {
            issues.insert(issues.end(), other.issues.begin(), other.issues.end());
        }

        // Format for display
        std::string format() const;
    };

    // Validation options
    struct Options {
        bool stopOnFirstError;          // Stop validation on first error
        bool includeInfoMessages;       // Include info-level messages
        bool validateConnections;       // Run connection validation
        bool validateGraph;             // Run graph validation

        Options() : stopOnFirstError(false), includeInfoMessages(false),
                    validateConnections(true), validateGraph(true) {}
    };

    // Constructor
    explicit PipelineValidator(Options opts = Options());

    // Main entry point - runs all validation phases
    Result validate(const PipelineDescription& desc) const;

    // Individual phases (for tooling that wants partial validation)
    Result validateModules(const PipelineDescription& desc) const;
    Result validateProperties(const PipelineDescription& desc) const;
    Result validateConnections(const PipelineDescription& desc) const;
    Result validateGraph(const PipelineDescription& desc) const;

    // Get/set options
    const Options& options() const { return options_; }
    void setOptions(Options opts) { options_ = std::move(opts); }

private:
    Options options_;
};

} // namespace apra
