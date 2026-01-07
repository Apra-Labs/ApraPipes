# B1: Pipeline Description IR (Intermediate Representation)

**Sprint:** 1 (Week 1-2)  
**Priority:** P0 - Critical Path  
**Effort:** 2 days  
**Depends On:** None (Start immediately, parallel with A1)  
**Blocks:** B2 (TOML Parser), C1 (Validator Shell)  

## Description

Define the intermediate representation that all frontend parsers (TOML, YAML, JSON) will produce. This is the **language-agnostic** data structure that represents a pipeline definition.

This task runs **parallel to A1** on Day 1.

## Acceptance Criteria

### Unit Tests
- [ ] `ModuleInstance` can hold instance_id, module_type, and properties map
- [ ] `PropertyValue` variant can hold: int64_t, double, bool, string, vector<int64_t>, vector<double>, vector<string>
- [ ] `Connection` stores from_module, from_pin, to_module, to_pin
- [ ] `PipelineSettings` has name, version, description, queue_size, on_error, auto_start
- [ ] `PipelineDescription` aggregates settings, modules vector, connections vector
- [ ] Source tracking fields (source_format, source_path) are populated
- [ ] Empty pipeline description is valid (for error cases)

### Behavioral (Given/When/Then)

**Scenario: Create pipeline description programmatically**
```
Given an empty PipelineDescription
When I add a ModuleInstance with id="source", type="FileReaderModule"
And I add a ModuleInstance with id="decoder", type="H264Decoder"
And I add a Connection from "source.output" to "decoder.input"
Then description.modules.size() == 2
And description.connections.size() == 1
```

**Scenario: Property values support multiple types**
```
Given a ModuleInstance
When I set properties["port"] = int64_t(8080)
And I set properties["enabled"] = true
And I set properties["url"] = std::string("rtsp://...")
Then all values are retrievable with correct types via std::get<T>
```

**Scenario: Track source format for error messages**
```
Given a PipelineDescription parsed from "pipeline.toml"
When parsing completes
Then source_format == "toml"
And source_path == "pipeline.toml"
```

### Requirements
- Header-only is acceptable but .cpp is fine too
- Must use `std::variant` for PropertyValue (C++17)
- No dependencies on Metadata.h (IR is independent)
- Should be serializable back to JSON (for debugging)
- All fields have sensible defaults

## Implementation Notes for Claude Code Agents

### File Location
```
base/include/core/PipelineDescription.h
base/src/core/PipelineDescription.cpp  (optional, for helpers)
```

### Key Implementation

```cpp
#pragma once
#include <string>
#include <vector>
#include <map>
#include <variant>

namespace apra {

// Property values that can appear in TOML/YAML/JSON
using PropertyValue = std::variant<
    int64_t,
    double,
    bool,
    std::string,
    std::vector<int64_t>,
    std::vector<double>,
    std::vector<std::string>
>;

struct ModuleInstance {
    std::string instance_id;      // User-defined: "my_decoder"
    std::string module_type;      // Registry name: "H264DecoderNvCodec"
    std::map<std::string, PropertyValue> properties;
};

struct Connection {
    std::string from_module;      // "source"
    std::string from_pin;         // "output"  
    std::string to_module;        // "decoder"
    std::string to_pin;           // "input"
    
    // Helper to parse "source.output" format
    static Connection parse(const std::string& from, const std::string& to);
};

struct PipelineSettings {
    std::string name;
    std::string version = "1.0";
    std::string description;
    
    int queue_size = 10;
    std::string on_error = "restart_module";  // "stop_pipeline" | "skip_frame"
    bool auto_start = false;
};

struct PipelineDescription {
    PipelineSettings settings;
    std::vector<ModuleInstance> modules;
    std::vector<Connection> connections;
    
    // Source tracking for error messages
    std::string source_format;    // "toml", "yaml", "json"
    std::string source_path;      // File path or "<inline>"
    
    // Helpers
    const ModuleInstance* findModule(const std::string& id) const;
    std::string toJson() const;  // For debugging
};

} // namespace apra
```

### Helper: Connection Parsing
```cpp
// Parse "module.pin" format
Connection Connection::parse(const std::string& from, const std::string& to) {
    Connection c;
    auto dot1 = from.find('.');
    auto dot2 = to.find('.');
    
    c.from_module = from.substr(0, dot1);
    c.from_pin = from.substr(dot1 + 1);
    c.to_module = to.substr(0, dot2);
    c.to_pin = to.substr(dot2 + 1);
    
    return c;
}
```

### Test File Location
```
base/test/pipeline_description_tests.cpp
```

---

## Definition of Done
- [ ] All structs compile and are usable
- [ ] Unit tests for all types pass
- [ ] `toJson()` produces valid JSON output
- [ ] `Connection::parse()` handles "module.pin" format
- [ ] Code reviewed and merged
