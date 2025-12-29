# B2: TOML Parser

**Sprint:** 1 (Week 1-2)  
**Priority:** P0 - Critical Path  
**Effort:** 4 days  
**Depends On:** B1 (PipelineDescription IR)  
**Blocks:** D1 (Factory), E1 (CLI)  

## Description

Implement the TOML parser that reads pipeline definition files and produces `PipelineDescription` IR. This is the primary frontend for the declarative pipeline system.

Uses `toml++` or `toml11` library (header-only, easy to integrate via vcpkg).

## Acceptance Criteria

### Unit Tests
- [ ] Parse minimal valid pipeline (1 module, 0 connections)
- [ ] Parse complete pipeline (multiple modules, connections, settings)
- [ ] Parse all property types: int, float, bool, string, arrays
- [ ] Parse `[pipeline]` section with name, version, description
- [ ] Parse `[pipeline.settings]` with queue_size, on_error, auto_start
- [ ] Parse `[modules.<id>]` with type and props
- [ ] Parse `[[connections]]` array of tables
- [ ] Report syntax errors with line number and column
- [ ] Report missing required fields (module type)
- [ ] Handle empty file gracefully
- [ ] Handle file not found gracefully

### Behavioral (Given/When/Then)

**Scenario: Parse complete pipeline**
```
Given TOML file:
    [pipeline]
    name = "test"
    
    [modules.source]
    type = "FileReaderModule"
        [modules.source.props]
        path = "/video.mp4"
    
    [modules.decoder]
    type = "H264Decoder"
    
    [[connections]]
    from = "source.output"
    to = "decoder.input"

When parser.parseFile("pipeline.toml") is called
Then result.success == true
And result.description.modules.size() == 2
And result.description.connections.size() == 1
And result.description.settings.name == "test"
```

**Scenario: Parse property types**
```
Given TOML with:
    [modules.test.props]
    int_val = 42
    float_val = 3.14
    bool_val = true
    string_val = "hello"
    int_array = [1, 2, 3]

When parsed
Then properties map contains all values with correct types
And std::get<int64_t>(props["int_val"]) == 42
And std::get<double>(props["float_val"]) == 3.14
```

**Scenario: Syntax error reporting**
```
Given TOML with syntax error on line 5
When parser.parseFile() is called
Then result.success == false
And result.error_line == 5
And result.error contains meaningful message
```

**Scenario: Missing module type**
```
Given TOML with:
    [modules.broken]
    # missing type field
        [modules.broken.props]
        foo = "bar"

When parsed
Then result.success == false
And result.error mentions "type" is required
```

### Requirements
- Use toml++ (header-only) or toml11 via vcpkg
- Implement `PipelineParser` interface from B1
- All errors include file location (line, column)
- Support both file and string input
- Handle TOML edge cases (inline tables, multi-line strings, etc.)

## Implementation Notes for Claude Code Agents

### File Locations
```
base/include/parsers/TomlParser.h
base/src/parsers/TomlParser.cpp
```

### vcpkg Integration
Add to `base/vcpkg.json`:
```json
{
  "dependencies": [
    "tomlplusplus"
  ]
}
```

### Key Implementation

```cpp
#pragma once
#include "core/PipelineParser.h"
#include <toml++/toml.h>

namespace apra {

class TomlParser : public PipelineParser {
public:
    ParseResult parseFile(const std::string& filepath) override;
    ParseResult parseString(const std::string& content) override;
    
    std::string formatName() const override { return "toml"; }
    std::vector<std::string> fileExtensions() const override { 
        return {".toml"}; 
    }

private:
    ParseResult parse(const toml::table& root, const std::string& source);
    
    void parsePipelineSection(const toml::table& root, PipelineDescription& desc);
    void parseModulesSection(const toml::table& root, PipelineDescription& desc);
    void parseConnectionsSection(const toml::table& root, PipelineDescription& desc);
    
    PropertyValue toPropertyValue(const toml::node& node);
};

} // namespace apra
```

### Parsing Logic

```cpp
ParseResult TomlParser::parseFile(const std::string& filepath) {
    ParseResult result;
    result.description.source_format = "toml";
    result.description.source_path = filepath;
    
    try {
        auto root = toml::parse_file(filepath);
        return parse(root, filepath);
    }
    catch (const toml::parse_error& err) {
        result.success = false;
        result.error = err.what();
        result.error_line = err.source().begin.line;
        result.error_column = err.source().begin.column;
        return result;
    }
}

void TomlParser::parseModulesSection(const toml::table& root, PipelineDescription& desc) {
    auto modules = root["modules"].as_table();
    if (!modules) return;
    
    for (const auto& [id, value] : *modules) {
        ModuleInstance instance;
        instance.instance_id = std::string(id.str());
        
        auto* mod_table = value.as_table();
        if (!mod_table) {
            throw std::runtime_error("Module '" + instance.instance_id + "' must be a table");
        }
        
        // Get type (required)
        auto type_node = (*mod_table)["type"];
        if (!type_node) {
            throw std::runtime_error("Module '" + instance.instance_id + "' missing 'type' field");
        }
        instance.module_type = type_node.value_or<std::string>("");
        
        // Get props (optional)
        auto props = (*mod_table)["props"].as_table();
        if (props) {
            for (const auto& [key, val] : *props) {
                instance.properties[std::string(key.str())] = toPropertyValue(val);
            }
        }
        
        desc.modules.push_back(std::move(instance));
    }
}

PropertyValue TomlParser::toPropertyValue(const toml::node& node) {
    if (auto* i = node.as_integer()) return i->get();
    if (auto* f = node.as_floating_point()) return f->get();
    if (auto* b = node.as_boolean()) return b->get();
    if (auto* s = node.as_string()) return std::string(s->get());
    if (auto* arr = node.as_array()) {
        // Detect array type from first element
        if (arr->empty()) return std::vector<int64_t>{};
        
        if (arr->front().is_integer()) {
            std::vector<int64_t> result;
            for (const auto& elem : *arr) {
                result.push_back(elem.as_integer()->get());
            }
            return result;
        }
        // ... handle other array types
    }
    
    throw std::runtime_error("Unsupported TOML value type");
}
```

### Connection Parsing

```cpp
void TomlParser::parseConnectionsSection(const toml::table& root, PipelineDescription& desc) {
    auto connections = root["connections"].as_array();
    if (!connections) return;
    
    for (const auto& conn_node : *connections) {
        auto* conn_table = conn_node.as_table();
        if (!conn_table) continue;
        
        auto from = (*conn_table)["from"].value_or<std::string>("");
        auto to = (*conn_table)["to"].value_or<std::string>("");
        
        if (from.empty() || to.empty()) {
            throw std::runtime_error("Connection missing 'from' or 'to' field");
        }
        
        desc.connections.push_back(Connection::parse(from, to));
    }
}
```

### Test File Location
```
base/test/toml_parser_tests.cpp
```

### Test Data
Create test TOML files in:
```
base/test/data/pipelines/
├── minimal.toml
├── complete.toml
├── all_property_types.toml
├── syntax_error.toml
└── missing_type.toml
```

---

## Definition of Done
- [ ] toml++ integrated via vcpkg
- [ ] All unit tests pass
- [ ] Parses reference pipeline.toml from RFC
- [ ] Error messages include line numbers
- [ ] PipelineParser interface implemented
- [ ] Code reviewed and merged
