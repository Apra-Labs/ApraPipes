# A2: Module Registry

**Sprint:** 1 (Week 1-2)  
**Priority:** P0 - Critical Path  
**Effort:** 4 days  
**Depends On:** A1 (Core Metadata Types)  
**Blocks:** D1 (Factory), E2 (Schema Generator), C1 (Validator), All module metadata tasks  

## Description

Create the central registry where all modules register their metadata at static initialization time. The registry provides runtime queries and factory methods to instantiate modules by name.

This is the **bridge** between compile-time Metadata and runtime operations.

## Acceptance Criteria

### Unit Tests
- [ ] `REGISTER_MODULE(ModuleClass, PropsClass)` macro compiles and registers at static init
- [ ] `ModuleRegistry::instance()` returns singleton
- [ ] `getModule("ModuleName")` returns ModuleInfo pointer or nullptr
- [ ] `getAllModules()` returns list of all registered module names
- [ ] `getModulesByCategory(Category::Source)` filters correctly
- [ ] `getModulesByTag("decoder")` returns modules with that tag
- [ ] `hasModule("ModuleName")` returns bool
- [ ] `createModule("ModuleName", props_map)` instantiates module with properties
- [ ] `toJson()` exports full registry as JSON
- [ ] `toToml()` exports as TOML (for documentation)
- [ ] Duplicate registration throws or warns

### Behavioral (Given/When/Then)

**Scenario: Register module via macro**
```
Given a module class with Metadata struct
When REGISTER_MODULE(MyModule, MyModuleProps) is in the .cpp file
And the program starts
Then ModuleRegistry::instance().hasModule("MyModule") == true
```

**Scenario: Query modules by category**
```
Given FileReaderModule (Source) and H264Decoder (Transform) are registered
When I call getModulesByCategory(ModuleCategory::Source)
Then result contains "FileReaderModule"
And result does NOT contain "H264Decoder"
```

**Scenario: Query modules by tag**
```
Given H264DecoderNvCodec with tags = {"decoder", "h264", "nvidia", "cuda_required"}
When I call getModulesByTag("decoder")
Then result contains "H264DecoderNvCodec"
When I call getModulesByTag("encoder")
Then result does NOT contain "H264DecoderNvCodec"
```

**Scenario: Create module instance from registry**
```
Given "FileReaderModule" is registered
When I call createModule("FileReaderModule", {{"path", "/video.mp4"}})
Then a valid Module* is returned
And the module's props.path == "/video.mp4"
```

**Scenario: Export registry to JSON**
```
Given multiple modules are registered
When I call toJson()
Then valid JSON is returned with all modules, their pins, properties, and tags
```

### Requirements
- Thread-safe singleton (modules register from multiple TUs)
- Registration happens at static init time (before main)
- Must store runtime copies of metadata (can't rely on constexpr at runtime for some ops)
- Factory function must handle property type conversion
- Export formats: JSON (for LLM), TOML (for docs), Markdown (optional)

## Implementation Notes for Claude Code Agents

### File Locations
```
base/include/core/ModuleRegistry.h
base/src/core/ModuleRegistry.cpp
```

### Key Structures

```cpp
// Runtime representation of module metadata
struct ModuleInfo {
    std::string name;
    ModuleCategory category;
    std::string version;
    std::string description;
    std::vector<std::string> tags;
    
    struct PinInfo {
        std::string name;
        std::vector<std::string> frame_types;
        bool required;
        std::string description;
    };
    std::vector<PinInfo> inputs;
    std::vector<PinInfo> outputs;
    
    struct PropInfo {
        std::string name;
        std::string type;  // "int", "float", "bool", "string", "enum"
        std::string mutability;  // "static", "dynamic"
        std::string default_value;
        std::string min_value;
        std::string max_value;
        std::string regex_pattern;
        std::vector<std::string> enum_values;
        std::string description;
    };
    std::vector<PropInfo> properties;
    
    // Factory function type
    using FactoryFn = std::function<std::unique_ptr<Module>(
        const std::map<std::string, PropertyValue>&
    )>;
    FactoryFn factory;
};

class ModuleRegistry {
public:
    static ModuleRegistry& instance();
    
    // Registration (called from REGISTER_MODULE macro)
    void registerModule(ModuleInfo info);
    
    // Queries
    bool hasModule(const std::string& name) const;
    const ModuleInfo* getModule(const std::string& name) const;
    std::vector<std::string> getAllModules() const;
    std::vector<std::string> getModulesByCategory(ModuleCategory cat) const;
    std::vector<std::string> getModulesByTag(const std::string& tag) const;
    
    // Factory
    std::unique_ptr<Module> createModule(
        const std::string& name,
        const std::map<std::string, PropertyValue>& props
    ) const;
    
    // Export
    std::string toJson() const;
    std::string toToml() const;
    
private:
    ModuleRegistry() = default;
    std::map<std::string, ModuleInfo> modules_;
    mutable std::mutex mutex_;
};
```

### REGISTER_MODULE Macro

```cpp
#define REGISTER_MODULE(ModuleClass, PropsClass) \
    namespace { \
        static bool _registered_##ModuleClass = []() { \
            ModuleInfo info; \
            /* Extract from ModuleClass::Metadata */ \
            info.name = std::string(ModuleClass::Metadata::name); \
            info.category = ModuleClass::Metadata::category; \
            info.version = std::string(ModuleClass::Metadata::version); \
            info.description = std::string(ModuleClass::Metadata::description); \
            /* Copy tags */ \
            for (const auto& tag : ModuleClass::Metadata::tags) { \
                info.tags.push_back(std::string(tag)); \
            } \
            /* Copy inputs, outputs, properties... */ \
            /* ... */ \
            /* Factory function */ \
            info.factory = [](const std::map<std::string, PropertyValue>& props) { \
                PropsClass moduleProps; \
                /* Apply props to moduleProps */ \
                return std::make_unique<ModuleClass>(moduleProps); \
            }; \
            ModuleRegistry::instance().registerModule(std::move(info)); \
            return true; \
        }(); \
    }
```

### Thread Safety
```cpp
// Use mutex for thread-safe registration
void ModuleRegistry::registerModule(ModuleInfo info) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (modules_.count(info.name)) {
        LOG_WARNING << "Module already registered: " << info.name;
        return;
    }
    modules_[info.name] = std::move(info);
}
```

### Test File Location
```
base/test/module_registry_tests.cpp
```

### Testing Strategy
Create a mock module with full Metadata for testing:
```cpp
class TestModule : public Module {
public:
    struct Metadata {
        static constexpr std::string_view name = "TestModule";
        static constexpr ModuleCategory category = ModuleCategory::Transform;
        static constexpr std::string_view version = "1.0";
        static constexpr std::string_view description = "Test module";
        static constexpr std::array<std::string_view, 2> tags = {"test", "mock"};
        // ... pins, properties
    };
};

REGISTER_MODULE(TestModule, TestModuleProps)
```

---

## Definition of Done
- [ ] REGISTER_MODULE macro works with real module classes
- [ ] All query methods return correct results
- [ ] createModule instantiates working modules
- [ ] toJson() output is valid JSON
- [ ] Thread-safe under concurrent registration
- [ ] Unit tests pass
- [ ] Code reviewed and merged
