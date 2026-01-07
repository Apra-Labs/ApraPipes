# E2: Schema Generator

**Sprint:** 2 (Week 3-4)  
**Priority:** P1 - High  
**Effort:** 2 days  
**Depends On:** A2 (Module Registry)  
**Blocks:** LLM Integration (Phase 4)  

## Description

Create a build-time tool that exports the ModuleRegistry and FrameTypeRegistry to JSON/Markdown files. These artifacts are used by:
1. LLMs for pipeline generation context
2. Documentation generation
3. IDE tooling (JSON Schema for TOML validation)

This runs **parallel** to the Factory and CLI development.

## Acceptance Criteria

### Unit Tests
- [ ] Generates valid JSON from ModuleRegistry
- [ ] JSON includes all modules with pins, properties, tags
- [ ] Generates valid JSON from FrameTypeRegistry
- [ ] JSON includes frame type hierarchy
- [ ] Generates Markdown documentation
- [ ] CMake integration runs at build time
- [ ] Output files are created in build directory

### Behavioral (Given/When/Then)

**Scenario: Generate module schema**
```
Given modules are registered
When apra_schema_generator --modules-json modules.json is run
Then modules.json is created
And it contains valid JSON with all modules
And each module has: name, category, tags, inputs, outputs, properties
```

**Scenario: CMake build generates schema**
```
Given CMake project is configured
When cmake --build . is run
Then ${CMAKE_BINARY_DIR}/schema/modules.json exists
And ${CMAKE_BINARY_DIR}/schema/frame_types.json exists
```

**Scenario: Generate documentation**
```
Given modules are registered
When apra_schema_generator --modules-md modules.md is run
Then modules.md contains readable documentation
And modules are grouped by category
```

### Requirements
- Header-only nlohmann/json for JSON generation
- Standalone executable (links all modules)
- CMake custom command runs after library build
- Output formats: JSON (LLM), Markdown (docs)
- Schema files are build artifacts, NOT committed to repo

## Implementation Notes for Claude Code Agents

### File Locations
```
base/tools/schema_generator.cpp
```

### CMake Integration

```cmake
# In base/CMakeLists.txt

# Schema generator tool - links all modules to get their metadata
add_executable(apra_schema_generator
    tools/schema_generator.cpp
)
target_link_libraries(apra_schema_generator aprapipes)

# Generate schema at build time
set(SCHEMA_DIR ${CMAKE_BINARY_DIR}/schema)

add_custom_command(
    OUTPUT ${SCHEMA_DIR}/modules.json ${SCHEMA_DIR}/frame_types.json
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SCHEMA_DIR}
    COMMAND apra_schema_generator 
        --modules-json ${SCHEMA_DIR}/modules.json
        --frame-types-json ${SCHEMA_DIR}/frame_types.json
        --modules-md ${SCHEMA_DIR}/MODULES.md
    DEPENDS apra_schema_generator
    COMMENT "Generating ApraPipes schema from C++ metadata"
)

add_custom_target(generate_schema ALL 
    DEPENDS ${SCHEMA_DIR}/modules.json ${SCHEMA_DIR}/frame_types.json
)
```

### Key Implementation

```cpp
#include "core/ModuleRegistry.h"
#include "core/FrameTypeRegistry.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

json generateModulesJson() {
    json root;
    root["version"] = "1.0";
    root["generated"] = /* timestamp */;
    
    auto& registry = ModuleRegistry::instance();
    
    for (const auto& name : registry.getAllModules()) {
        const auto* info = registry.getModule(name);
        if (!info) continue;
        
        json module;
        module["category"] = categoryToString(info->category);
        module["version"] = info->version;
        module["description"] = info->description;
        module["tags"] = info->tags;
        
        // Inputs
        json inputs = json::array();
        for (const auto& pin : info->inputs) {
            inputs.push_back({
                {"name", pin.name},
                {"frame_types", pin.frame_types},
                {"required", pin.required},
                {"description", pin.description}
            });
        }
        module["inputs"] = inputs;
        
        // Outputs
        json outputs = json::array();
        for (const auto& pin : info->outputs) {
            outputs.push_back({
                {"name", pin.name},
                {"frame_types", pin.frame_types},
                {"description", pin.description}
            });
        }
        module["outputs"] = outputs;
        
        // Properties
        json properties = json::object();
        for (const auto& prop : info->properties) {
            properties[prop.name] = {
                {"type", prop.type},
                {"mutability", prop.mutability},
                {"default", prop.default_value},
                {"min", prop.min_value},
                {"max", prop.max_value},
                {"regex", prop.regex_pattern},
                {"enum_values", prop.enum_values},
                {"description", prop.description}
            };
        }
        module["properties"] = properties;
        
        root["modules"][name] = module;
    }
    
    return root;
}

json generateFrameTypesJson() {
    json root;
    auto& registry = FrameTypeRegistry::instance();
    
    for (const auto& name : registry.getAllFrameTypes()) {
        const auto* info = registry.getFrameType(name);
        if (!info) continue;
        
        root["frame_types"][name] = {
            {"parent", info->parent},
            {"description", info->description},
            {"tags", info->tags}
        };
    }
    
    return root;
}

std::string generateModulesMarkdown() {
    std::ostringstream md;
    auto& registry = ModuleRegistry::instance();
    
    md << "# ApraPipes Modules Reference\n\n";
    md << "Generated automatically from C++ metadata.\n\n";
    
    // Group by category
    std::map<std::string, std::vector<std::string>> byCategory;
    for (const auto& name : registry.getAllModules()) {
        const auto* info = registry.getModule(name);
        byCategory[categoryToString(info->category)].push_back(name);
    }
    
    for (const auto& [category, modules] : byCategory) {
        md << "## " << category << " Modules\n\n";
        
        for (const auto& name : modules) {
            const auto* info = registry.getModule(name);
            
            md << "### " << info->name << "\n\n";
            md << info->description << "\n\n";
            md << "**Tags:** " << join(info->tags, ", ") << "\n\n";
            
            md << "#### Inputs\n";
            for (const auto& pin : info->inputs) {
                md << "- `" << pin.name << "` [" << join(pin.frame_types, ", ") << "]";
                if (pin.required) md << " (required)";
                md << "\n";
            }
            
            // ... outputs, properties
            md << "\n---\n\n";
        }
    }
    
    return md.str();
}

int main(int argc, char* argv[]) {
    std::string modulesJson, frameTypesJson, modulesMd;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--modules-json" && i + 1 < argc) {
            modulesJson = argv[++i];
        } else if (arg == "--frame-types-json" && i + 1 < argc) {
            frameTypesJson = argv[++i];
        } else if (arg == "--modules-md" && i + 1 < argc) {
            modulesMd = argv[++i];
        }
    }
    
    // Generate outputs
    if (!modulesJson.empty()) {
        std::ofstream f(modulesJson);
        f << generateModulesJson().dump(2);
        std::cout << "Generated: " << modulesJson << "\n";
    }
    
    if (!frameTypesJson.empty()) {
        std::ofstream f(frameTypesJson);
        f << generateFrameTypesJson().dump(2);
        std::cout << "Generated: " << frameTypesJson << "\n";
    }
    
    if (!modulesMd.empty()) {
        std::ofstream f(modulesMd);
        f << generateModulesMarkdown();
        std::cout << "Generated: " << modulesMd << "\n";
    }
    
    return 0;
}
```

### vcpkg Dependency
Add to `base/vcpkg.json`:
```json
{
  "dependencies": [
    "nlohmann-json"
  ]
}
```

### Test Strategy
1. Build project
2. Check schema files exist in build directory
3. Validate JSON is parseable
4. Check all registered modules appear in output

---

## Definition of Done
- [ ] schema_generator builds and runs
- [ ] CMake generates schema at build time
- [ ] JSON output is valid and complete
- [ ] Markdown is readable
- [ ] Integration with vcpkg (nlohmann-json)
- [ ] Code reviewed and merged
