# C1: Validator Shell (Minimal Framework)

**Sprint:** 1 (Week 1-2)  
**Priority:** P1 - High (but non-blocking)  
**Effort:** 1 day  
**Depends On:** A2 (Module Registry), B1 (PipelineDescription IR)  
**Blocks:** C2, C3, C4, C5 (incremental validation rules)  

## Description

Create the **skeleton** validator framework with the structure for validation phases, but minimal actual validation. The key insight: **validator should not block factory development**.

This task creates:
1. `ValidationIssue` structure (Error/Warning/Info)
2. `PipelineValidator` class with phase methods
3. Skeleton implementation that passes everything
4. Framework for adding rules incrementally

## Acceptance Criteria

### Unit Tests
- [ ] `ValidationIssue` can be created with level, code, location, message
- [ ] `PipelineValidator::validate()` returns Result
- [ ] `Result::hasErrors()` and `hasWarnings()` work correctly
- [ ] Empty pipeline returns empty result (no errors in MVP)
- [ ] Phase methods exist: `validateModules()`, `validateProperties()`, etc.

### Behavioral (Given/When/Then)

**Scenario: Validate any pipeline (MVP behavior)**
```
Given any PipelineDescription (valid or invalid)
When validator.validate(description) is called
Then result.hasErrors() == false (MVP: pass everything)
And result contains Info-level issues for logging
```

**Scenario: Framework supports future rules**
```
Given PipelineValidator with TODO placeholders for rules
When a developer adds a new rule in C2/C3/C4/C5
Then the framework supports it without structural changes
```

### Requirements
- Skeleton only - no actual validation logic
- All phase methods exist but contain only logging/TODOs
- Ready for incremental rule addition
- Does NOT block ModuleFactory development

## Implementation Notes for Claude Code Agents

### File Locations
```
base/include/core/PipelineValidator.h
base/src/core/PipelineValidator.cpp
```

### Minimal Implementation

```cpp
#pragma once
#include "PipelineDescription.h"
#include <vector>
#include <string>
#include <algorithm>

namespace apra {

struct ValidationIssue {
    enum class Level { Error, Warning, Info };
    
    Level level;
    std::string code;       // "E001", "W010", "I001"
    std::string location;   // "modules.decoder.props.device_id"
    std::string message;    // Human-readable
    std::string suggestion; // Optional fix suggestion
};

class PipelineValidator {
public:
    struct Result {
        std::vector<ValidationIssue> issues;
        
        bool hasErrors() const {
            return std::any_of(issues.begin(), issues.end(),
                [](const auto& i) { return i.level == ValidationIssue::Level::Error; });
        }
        
        bool hasWarnings() const {
            return std::any_of(issues.begin(), issues.end(),
                [](const auto& i) { return i.level == ValidationIssue::Level::Warning; });
        }
        
        std::vector<ValidationIssue> errors() const {
            std::vector<ValidationIssue> result;
            std::copy_if(issues.begin(), issues.end(), std::back_inserter(result),
                [](const auto& i) { return i.level == ValidationIssue::Level::Error; });
            return result;
        }
    };
    
    // Main entry point
    Result validate(const PipelineDescription& desc) const;
    
    // Individual phases (for tooling that wants partial validation)
    Result validateModules(const PipelineDescription& desc) const;
    Result validateProperties(const PipelineDescription& desc) const;
    Result validateConnections(const PipelineDescription& desc) const;
    Result validateGraph(const PipelineDescription& desc) const;
    
private:
    // Future: helper methods for specific checks
};

} // namespace apra
```

### Skeleton Implementation

```cpp
#include "core/PipelineValidator.h"
#include "core/ModuleRegistry.h"
#include "Logger.h"

namespace apra {

PipelineValidator::Result PipelineValidator::validate(const PipelineDescription& desc) const {
    Result result;
    
    LOG_INFO << "Validating pipeline: " << desc.settings.name;
    LOG_INFO << "  Modules: " << desc.modules.size();
    LOG_INFO << "  Connections: " << desc.connections.size();
    
    // Phase 1: Module validation
    auto moduleResult = validateModules(desc);
    result.issues.insert(result.issues.end(), 
        moduleResult.issues.begin(), moduleResult.issues.end());
    
    // Phase 2: Property validation
    auto propResult = validateProperties(desc);
    result.issues.insert(result.issues.end(),
        propResult.issues.begin(), propResult.issues.end());
    
    // Phase 3: Connection validation
    auto connResult = validateConnections(desc);
    result.issues.insert(result.issues.end(),
        connResult.issues.begin(), connResult.issues.end());
    
    // Phase 4: Graph validation
    auto graphResult = validateGraph(desc);
    result.issues.insert(result.issues.end(),
        graphResult.issues.begin(), graphResult.issues.end());
    
    // Summary
    result.issues.push_back({
        ValidationIssue::Level::Info,
        "I000",
        "pipeline",
        "Validation complete: " + std::to_string(result.errors().size()) + " errors, " +
            std::to_string(result.hasWarnings() ? 1 : 0) + " warnings"
    });
    
    return result;
}

PipelineValidator::Result PipelineValidator::validateModules(const PipelineDescription& desc) const {
    Result result;
    
    // TODO (C2): Check each module type exists in registry
    // TODO (C2): Check module category constraints
    // TODO (C2): Warn on version mismatch
    
    // MVP: Just log what we found
    for (const auto& module : desc.modules) {
        result.issues.push_back({
            ValidationIssue::Level::Info,
            "I010",
            "modules." + module.instance_id,
            "Found module: " + module.module_type
        });
    }
    
    return result;
}

PipelineValidator::Result PipelineValidator::validateProperties(const PipelineDescription& desc) const {
    Result result;
    
    // TODO (C3): Check property names exist in module metadata
    // TODO (C3): Check property types match
    // TODO (C3): Check ranges (min/max)
    // TODO (C3): Check regex patterns
    // TODO (C3): Check enum values
    // TODO (C3): Warn on missing optional properties
    
    return result;
}

PipelineValidator::Result PipelineValidator::validateConnections(const PipelineDescription& desc) const {
    Result result;
    
    // TODO (C4): Check source/dest modules exist
    // TODO (C4): Check pins exist on modules
    // TODO (C4): Check frame type compatibility
    // TODO (C4): Check for duplicate connections to same input
    // TODO (C4): Check required pins are connected
    
    return result;
}

PipelineValidator::Result PipelineValidator::validateGraph(const PipelineDescription& desc) const {
    Result result;
    
    // TODO (C5): Check pipeline has at least one source
    // TODO (C5): Check graph is DAG (no cycles)
    // TODO (C5): Warn on orphan modules
    
    return result;
}

} // namespace apra
```

### Test File Location
```
base/test/pipeline_validator_tests.cpp
```

---

## Definition of Done
- [ ] ValidationIssue and Result structures work
- [ ] validate() runs all phases
- [ ] Phase methods exist with TODO comments
- [ ] Unit tests pass
- [ ] Does NOT block Factory development
- [ ] Code reviewed and merged
