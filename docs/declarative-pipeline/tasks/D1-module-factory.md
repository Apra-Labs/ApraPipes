# D1: Module Factory

**Sprint:** 2 (Week 3-4)  
**Priority:** P0 - Critical Path  
**Effort:** 4 days  
**Depends On:** A2 (Module Registry), B2 (TOML Parser)  
**Blocks:** E1 (CLI Tool)  

## Description

Implement the factory that takes a `PipelineDescription` and constructs an actual running `PipeLine` with all modules instantiated and connected.

This is the **core engine** that brings declarative definitions to life.

## Acceptance Criteria

### Unit Tests
- [ ] `build()` creates Pipeline from valid PipelineDescription
- [ ] Modules are instantiated via ModuleRegistry
- [ ] Static properties are applied to module props
- [ ] Modules are connected via `setNext()` (existing ApraPipes API)
- [ ] Pipeline settings (queue_size) are applied
- [ ] BuildResult contains success flag and any issues
- [ ] Unknown module type returns error in BuildResult
- [ ] Missing required property uses default (warning in result)
- [ ] Property type mismatch is handled gracefully
- [ ] Empty pipeline (no modules) returns error

### Behavioral (Given/When/Then)

**Scenario: Build simple pipeline**
```
Given PipelineDescription with:
    - FileReaderModule (id: "source", props: {path: "/video.mp4"})
    - H264Decoder (id: "decoder", props: {device_id: 0})
    - Connection: source.output -> decoder.input

When factory.build(description) is called
Then result.success() == true
And result.pipeline is not null
And pipeline contains 2 modules
And modules are connected
```

**Scenario: Apply properties to module**
```
Given PipelineDescription with:
    - FileReaderModule with props = {path: "/test.mp4", loop: true}

When factory builds the module
Then the module's FileReaderModuleProps.path == "/test.mp4"
And the module's FileReaderModuleProps.loop == true
```

**Scenario: Handle unknown module type**
```
Given PipelineDescription with:
    - module type = "NonExistentModule"

When factory.build() is called
Then result.success() == false
And result.issues contains error with code "E001" (UNKNOWN_MODULE)
```

**Scenario: Handle property type mismatch**
```
Given PipelineDescription with:
    - H264Decoder with props = {device_id: "not_an_int"}

When factory.build() is called
Then result.issues contains warning about type conversion
And factory attempts best-effort conversion or uses default
```

### Requirements
- Use ModuleRegistry::createModule() to instantiate
- Must work with existing ApraPipes connection API (setNext, addInputPin, etc.)
- Collect all issues, don't fail on first error (report all problems)
- Support future: auto-insert converter modules (not in MVP)
- Thread-safe (factory may be reused)

## Implementation Notes for Claude Code Agents

### File Locations
```
base/include/core/ModuleFactory.h
base/src/core/ModuleFactory.cpp
```

### Key Structures

```cpp
#pragma once
#include "PipelineDescription.h"
#include "ModuleRegistry.h"
#include "PipeLine.h"
#include <memory>
#include <vector>

namespace apra {

// Reuse ValidationIssue from validator (or define here if validator not ready)
struct BuildIssue {
    enum class Level { Error, Warning, Info };
    
    Level level;
    std::string code;
    std::string location;   // "modules.decoder.props.device_id"
    std::string message;
};

class ModuleFactory {
public:
    struct Options {
        bool auto_insert_converters = false;  // Future
        bool strict_mode = false;              // Fail on warnings
    };
    
    struct BuildResult {
        std::unique_ptr<PipeLine> pipeline;
        std::vector<BuildIssue> issues;
        
        bool success() const { 
            return pipeline != nullptr && 
                   !std::any_of(issues.begin(), issues.end(), 
                       [](const auto& i) { return i.level == BuildIssue::Level::Error; });
        }
        
        bool hasErrors() const;
        bool hasWarnings() const;
    };
    
    explicit ModuleFactory(Options opts = {});
    
    BuildResult build(const PipelineDescription& desc);
    
private:
    Options options_;
    
    // Internal helpers
    boost::shared_ptr<Module> createModule(
        const ModuleInstance& instance,
        std::vector<BuildIssue>& issues
    );
    
    void applyProperties(
        Module* module,
        const ModuleInstance& instance,
        std::vector<BuildIssue>& issues
    );
    
    bool connectModules(
        PipeLine& pipeline,
        const std::vector<Connection>& connections,
        const std::map<std::string, boost::shared_ptr<Module>>& moduleMap,
        std::vector<BuildIssue>& issues
    );
};

} // namespace apra
```

### Core Build Logic

```cpp
BuildResult ModuleFactory::build(const PipelineDescription& desc) {
    BuildResult result;
    
    // Validate we have something to build
    if (desc.modules.empty()) {
        result.issues.push_back({
            BuildIssue::Level::Error,
            "E030",
            "pipeline",
            "Pipeline has no modules"
        });
        return result;
    }
    
    // Create pipeline
    result.pipeline = std::make_unique<PipeLine>("declarative_pipeline");
    
    // Map of instance_id -> Module for connection phase
    std::map<std::string, boost::shared_ptr<Module>> moduleMap;
    
    // Phase 1: Create all modules
    for (const auto& instance : desc.modules) {
        auto module = createModule(instance, result.issues);
        if (module) {
            moduleMap[instance.instance_id] = module;
            result.pipeline->appendModule(module);
        }
    }
    
    // If any critical errors, stop
    if (result.hasErrors()) {
        result.pipeline.reset();
        return result;
    }
    
    // Phase 2: Connect modules
    connectModules(*result.pipeline, desc.connections, moduleMap, result.issues);
    
    // Apply pipeline settings
    // (queue_size is typically per-module in ApraPipes, may need adaptation)
    
    return result;
}

boost::shared_ptr<Module> ModuleFactory::createModule(
    const ModuleInstance& instance,
    std::vector<BuildIssue>& issues
) {
    auto& registry = ModuleRegistry::instance();
    
    // Check module exists
    if (!registry.hasModule(instance.module_type)) {
        issues.push_back({
            BuildIssue::Level::Error,
            "E001",
            "modules." + instance.instance_id,
            "Unknown module type: " + instance.module_type
        });
        return nullptr;
    }
    
    // Create via registry factory
    try {
        auto module = registry.createModule(instance.module_type, instance.properties);
        
        issues.push_back({
            BuildIssue::Level::Info,
            "I001",
            "modules." + instance.instance_id,
            "Created module: " + instance.module_type
        });
        
        return module;
    }
    catch (const std::exception& e) {
        issues.push_back({
            BuildIssue::Level::Error,
            "E002",
            "modules." + instance.instance_id,
            "Failed to create module: " + std::string(e.what())
        });
        return nullptr;
    }
}
```

### Connection Logic

```cpp
bool ModuleFactory::connectModules(
    PipeLine& pipeline,
    const std::vector<Connection>& connections,
    const std::map<std::string, boost::shared_ptr<Module>>& moduleMap,
    std::vector<BuildIssue>& issues
) {
    for (const auto& conn : connections) {
        // Find source module
        auto srcIt = moduleMap.find(conn.from_module);
        if (srcIt == moduleMap.end()) {
            issues.push_back({
                BuildIssue::Level::Error,
                "E020",
                "connections",
                "Unknown source module: " + conn.from_module
            });
            continue;
        }
        
        // Find destination module
        auto dstIt = moduleMap.find(conn.to_module);
        if (dstIt == moduleMap.end()) {
            issues.push_back({
                BuildIssue::Level::Error,
                "E022",
                "connections",
                "Unknown destination module: " + conn.to_module
            });
            continue;
        }
        
        // Connect using ApraPipes API
        // Note: ApraPipes uses setNext() for simple connections
        // Pin-specific connections may need different API
        try {
            srcIt->second->setNext(dstIt->second);
            
            issues.push_back({
                BuildIssue::Level::Info,
                "I002",
                "connections",
                "Connected: " + conn.from_module + "." + conn.from_pin + 
                " -> " + conn.to_module + "." + conn.to_pin
            });
        }
        catch (const std::exception& e) {
            issues.push_back({
                BuildIssue::Level::Error,
                "E025",
                "connections",
                "Connection failed: " + std::string(e.what())
            });
        }
    }
    
    return !std::any_of(issues.begin(), issues.end(),
        [](const auto& i) { return i.level == BuildIssue::Level::Error; });
}
```

### Integration with ApraPipes

Review existing connection patterns in ApraPipes:
```cpp
// From existing tests (e.g., mp4_file_tests.cpp)
auto source = boost::shared_ptr<Module>(new FileReaderModule(props));
auto decoder = boost::shared_ptr<Module>(new H264Decoder(decoderProps));

source->setNext(decoder);

PipeLine p("test");
p.appendModule(source);
p.appendModule(decoder);
p.init();
p.run_all_threaded();
```

### Test File Location
```
base/test/module_factory_tests.cpp
```

### Test Strategy
1. Create test modules with Metadata (reuse from A2 tests)
2. Build PipelineDescription programmatically
3. Call factory.build()
4. Verify resulting pipeline works

---

## Definition of Done
- [ ] Factory creates working PipeLine from description
- [ ] All property types applied correctly
- [ ] Connections established via ApraPipes API
- [ ] Error handling is comprehensive
- [ ] Unit tests pass
- [ ] Integration test: TOML → Parse → Build → Run
- [ ] Code reviewed and merged
