// ============================================================
// File: declarative/PropertyValidators.h
// Task D2: Property Binding System - Validation
//
// Provides validators for property values:
// - RangeValidator: numeric min/max bounds
// - RegexValidator: string pattern matching
// - EnumValidator: allowed string values
// ============================================================

#pragma once

#include "PropertyMacros.h"
#include <string>
#include <vector>
#include <regex>
#include <memory>
#include <map>
#include <functional>
#include <sstream>
#include <limits>
#include <cmath>

namespace apra {

// ============================================================
// ValidationResult - outcome of validation
// ============================================================
struct ValidationResult {
    bool valid = true;
    std::string error;
    std::string propertyName;

    static ValidationResult ok() { return {true, "", ""}; }
    static ValidationResult fail(const std::string& prop, const std::string& msg) {
        return {false, msg, prop};
    }
};

// ============================================================
// PropertyValidator - base class for validators
// ============================================================
class PropertyValidator {
public:
    virtual ~PropertyValidator() = default;

    // Validate a property value, return error message if invalid
    virtual ValidationResult validate(
        const std::string& propName,
        const ScalarPropertyValue& value
    ) const = 0;

    // Human-readable description of the constraint
    virtual std::string describe() const = 0;
};

// ============================================================
// RangeValidator - validates numeric values are within bounds
// ============================================================
template<typename T>
class RangeValidator : public PropertyValidator {
    T min_;
    T max_;
    bool minInclusive_;
    bool maxInclusive_;

public:
    RangeValidator(T min, T max, bool minInclusive = true, bool maxInclusive = true)
        : min_(min), max_(max), minInclusive_(minInclusive), maxInclusive_(maxInclusive) {}

    ValidationResult validate(
        const std::string& propName,
        const ScalarPropertyValue& value
    ) const override {
        T v;

        // Extract value based on variant type
        if (auto* i = std::get_if<int64_t>(&value)) {
            v = static_cast<T>(*i);
        } else if (auto* d = std::get_if<double>(&value)) {
            v = static_cast<T>(*d);
        } else {
            return ValidationResult::fail(propName,
                "Expected numeric value for range validation");
        }

        // Check bounds
        bool minOk = minInclusive_ ? (v >= min_) : (v > min_);
        bool maxOk = maxInclusive_ ? (v <= max_) : (v < max_);

        if (!minOk || !maxOk) {
            std::ostringstream oss;
            oss << "Value " << v << " out of range "
                << (minInclusive_ ? "[" : "(")
                << min_ << ", " << max_
                << (maxInclusive_ ? "]" : ")");
            return ValidationResult::fail(propName, oss.str());
        }

        return ValidationResult::ok();
    }

    std::string describe() const override {
        std::ostringstream oss;
        oss << "range" << (minInclusive_ ? "[" : "(")
            << min_ << "," << max_
            << (maxInclusive_ ? "]" : ")");
        return oss.str();
    }
};

// Convenience aliases
using IntRangeValidator = RangeValidator<int64_t>;
using FloatRangeValidator = RangeValidator<double>;

// ============================================================
// RegexValidator - validates strings match a pattern
// ============================================================
class RegexValidator : public PropertyValidator {
    std::string pattern_;
    std::regex regex_;
    std::string description_;

public:
    explicit RegexValidator(const std::string& pattern,
                           const std::string& description = "")
        : pattern_(pattern), description_(description) {
        try {
            regex_ = std::regex(pattern);
        } catch (const std::regex_error& e) {
            throw std::runtime_error("Invalid regex pattern: " + pattern + " - " + e.what());
        }
    }

    ValidationResult validate(
        const std::string& propName,
        const ScalarPropertyValue& value
    ) const override {
        auto* str = std::get_if<std::string>(&value);
        if (!str) {
            return ValidationResult::fail(propName,
                "Expected string value for regex validation");
        }

        if (!std::regex_match(*str, regex_)) {
            std::string msg = "String '" + *str + "' does not match pattern";
            if (!description_.empty()) {
                msg += " (" + description_ + ")";
            } else {
                msg += ": " + pattern_;
            }
            return ValidationResult::fail(propName, msg);
        }

        return ValidationResult::ok();
    }

    std::string describe() const override {
        if (!description_.empty()) {
            return description_;
        }
        return "regex(" + pattern_ + ")";
    }
};

// Common regex patterns
namespace patterns {
    // IPv4 address
    inline const char* IPv4 = R"(^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$)";

    // Simple email (not RFC-compliant, but practical)
    inline const char* Email = R"(^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$)";

    // File path (Unix-style)
    inline const char* UnixPath = R"(^(/[^/\0]+)+/?$|^/$)";

    // URL
    inline const char* URL = R"(^https?://[^\s/$.?#].[^\s]*$)";

    // Alphanumeric with underscores
    inline const char* Identifier = R"(^[a-zA-Z_][a-zA-Z0-9_]*$)";
}

// ============================================================
// EnumValidator - validates string is one of allowed values
// ============================================================
class EnumValidator : public PropertyValidator {
    std::vector<std::string> allowed_;
    bool caseSensitive_;

public:
    EnumValidator(std::initializer_list<std::string> values, bool caseSensitive = true)
        : allowed_(values), caseSensitive_(caseSensitive) {}

    EnumValidator(std::vector<std::string> values, bool caseSensitive = true)
        : allowed_(std::move(values)), caseSensitive_(caseSensitive) {}

    ValidationResult validate(
        const std::string& propName,
        const ScalarPropertyValue& value
    ) const override {
        auto* str = std::get_if<std::string>(&value);
        if (!str) {
            return ValidationResult::fail(propName,
                "Expected string value for enum validation");
        }

        std::string input = *str;
        if (!caseSensitive_) {
            std::transform(input.begin(), input.end(), input.begin(), ::tolower);
        }

        for (const auto& allowed : allowed_) {
            std::string cmp = allowed;
            if (!caseSensitive_) {
                std::transform(cmp.begin(), cmp.end(), cmp.begin(), ::tolower);
            }
            if (input == cmp) {
                return ValidationResult::ok();
            }
        }

        std::ostringstream oss;
        oss << "Value '" << *str << "' not in allowed values: {";
        for (size_t i = 0; i < allowed_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << allowed_[i];
        }
        oss << "}";
        return ValidationResult::fail(propName, oss.str());
    }

    std::string describe() const override {
        std::ostringstream oss;
        oss << "enum{";
        for (size_t i = 0; i < allowed_.size(); ++i) {
            if (i > 0) oss << ",";
            oss << allowed_[i];
        }
        oss << "}";
        return oss.str();
    }

    const std::vector<std::string>& allowedValues() const { return allowed_; }
};

// ============================================================
// CompositeValidator - combines multiple validators (AND logic)
// ============================================================
class CompositeValidator : public PropertyValidator {
    std::vector<std::shared_ptr<PropertyValidator>> validators_;

public:
    void add(std::shared_ptr<PropertyValidator> validator) {
        validators_.push_back(std::move(validator));
    }

    ValidationResult validate(
        const std::string& propName,
        const ScalarPropertyValue& value
    ) const override {
        for (const auto& v : validators_) {
            auto result = v->validate(propName, value);
            if (!result.valid) {
                return result;
            }
        }
        return ValidationResult::ok();
    }

    std::string describe() const override {
        std::ostringstream oss;
        for (size_t i = 0; i < validators_.size(); ++i) {
            if (i > 0) oss << " AND ";
            oss << validators_[i]->describe();
        }
        return oss.str();
    }
};

// ============================================================
// PropertyValidatorRegistry - stores validators per property
// ============================================================
class PropertyValidatorRegistry {
    std::map<std::string, std::shared_ptr<PropertyValidator>> validators_;

public:
    void registerValidator(const std::string& propName,
                          std::shared_ptr<PropertyValidator> validator) {
        validators_[propName] = std::move(validator);
    }

    ValidationResult validate(const std::string& propName,
                             const ScalarPropertyValue& value) const {
        auto it = validators_.find(propName);
        if (it == validators_.end()) {
            return ValidationResult::ok();  // No validator = always valid
        }
        return it->second->validate(propName, value);
    }

    bool hasValidator(const std::string& propName) const {
        return validators_.find(propName) != validators_.end();
    }

    std::vector<ValidationResult> validateAll(
        const std::map<std::string, ScalarPropertyValue>& props
    ) const {
        std::vector<ValidationResult> results;
        for (const auto& [name, value] : props) {
            auto result = validate(name, value);
            if (!result.valid) {
                results.push_back(result);
            }
        }
        return results;
    }
};

// ============================================================
// Helper functions to create validators
// ============================================================

template<typename T>
inline std::shared_ptr<RangeValidator<T>> makeRangeValidator(T min, T max) {
    return std::make_shared<RangeValidator<T>>(min, max);
}

inline std::shared_ptr<RegexValidator> makeRegexValidator(
    const std::string& pattern,
    const std::string& description = ""
) {
    return std::make_shared<RegexValidator>(pattern, description);
}

inline std::shared_ptr<EnumValidator> makeEnumValidator(
    std::initializer_list<std::string> values,
    bool caseSensitive = true
) {
    return std::make_shared<EnumValidator>(values, caseSensitive);
}

} // namespace apra
