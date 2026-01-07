# E1: CLI Tool

**Sprint:** 2 (Week 3-4)  
**Priority:** P0 - Critical Path  
**Effort:** 3 days  
**Depends On:** D1 (Module Factory)  
**Blocks:** None (End of critical path)  

## Description

Create the command-line interface for the declarative pipeline system. This is the primary user interface for validating and running pipelines from TOML files.

```bash
aprapipes validate pipeline.toml
aprapipes run pipeline.toml
aprapipes list-modules
aprapipes describe H264DecoderNvCodec
```

## Acceptance Criteria

### Unit Tests
- [ ] `validate` command parses TOML and reports issues
- [ ] `validate` exits 0 on success, non-zero on errors
- [ ] `run` command starts pipeline and runs until signal
- [ ] `run --set prop=value` overrides properties
- [ ] `list-modules` prints all registered modules
- [ ] `list-modules --category source` filters by category
- [ ] `list-modules --tag decoder` filters by tag
- [ ] `describe <module>` prints module details (pins, props)
- [ ] Invalid arguments show help
- [ ] Missing file shows error with path

### Behavioral (Given/When/Then)

**Scenario: Validate valid pipeline**
```
Given valid pipeline.toml
When user runs: aprapipes validate pipeline.toml
Then output shows "Validation successful"
And exit code is 0
```

**Scenario: Validate pipeline with errors**
```
Given pipeline.toml with unknown module type
When user runs: aprapipes validate pipeline.toml
Then output shows error with location and message
And exit code is 1
```

**Scenario: Run pipeline**
```
Given valid pipeline.toml
When user runs: aprapipes run pipeline.toml
Then pipeline starts
And runs until Ctrl+C or completion
And exit code is 0 on clean shutdown
```

**Scenario: Override property at runtime**
```
Given pipeline.toml with decoder.device_id = 0
When user runs: aprapipes run pipeline.toml --set decoder.device_id=1
Then pipeline runs with device_id = 1
```

**Scenario: List modules by tag**
```
Given H264DecoderNvCodec with tag "decoder"
When user runs: aprapipes list-modules --tag decoder
Then output includes H264DecoderNvCodec
And output excludes modules without "decoder" tag
```

**Scenario: Describe module**
```
Given H264DecoderNvCodec is registered
When user runs: aprapipes describe H264DecoderNvCodec
Then output shows:
    - Category
    - Tags
    - Description
    - Input pins with frame types
    - Output pins with frame types
    - Properties with types, defaults, ranges
```

### Requirements
- Use existing CLI parsing (Boost.Program_options or simple argv)
- Graceful signal handling for `run` command (SIGINT, SIGTERM)
- Machine-readable output option (--json) for tooling
- Exit codes: 0 = success, 1 = validation error, 2 = runtime error
- Help text for all commands

## Implementation Notes for Claude Code Agents

### File Locations
```
base/tools/aprapipes_cli.cpp
```

### CMake Addition
```cmake
add_executable(aprapipes
    tools/aprapipes_cli.cpp
)
target_link_libraries(aprapipes aprapipes_lib)
install(TARGETS aprapipes DESTINATION bin)
```

### Key Implementation

```cpp
#include "parsers/TomlParser.h"
#include "core/ModuleFactory.h"
#include "core/ModuleRegistry.h"
#include <iostream>
#include <csignal>

static std::atomic<bool> g_running{true};

void signalHandler(int signum) {
    std::cout << "\nShutting down...\n";
    g_running = false;
}

int cmdValidate(const std::string& filepath) {
    TomlParser parser;
    auto parseResult = parser.parseFile(filepath);
    
    if (!parseResult.success) {
        std::cerr << "Parse error at line " << parseResult.error_line 
                  << ": " << parseResult.error << "\n";
        return 1;
    }
    
    // Optional: Run through validator if available
    // PipelineValidator validator;
    // auto validationResult = validator.validate(parseResult.description);
    
    std::cout << "✓ Validation successful\n";
    std::cout << "  Modules: " << parseResult.description.modules.size() << "\n";
    std::cout << "  Connections: " << parseResult.description.connections.size() << "\n";
    
    return 0;
}

int cmdRun(const std::string& filepath, 
           const std::map<std::string, std::string>& overrides) {
    // Parse
    TomlParser parser;
    auto parseResult = parser.parseFile(filepath);
    
    if (!parseResult.success) {
        std::cerr << "Parse error: " << parseResult.error << "\n";
        return 1;
    }
    
    // Apply overrides
    for (const auto& [path, value] : overrides) {
        // Parse "module.prop=value" and update description
        applyOverride(parseResult.description, path, value);
    }
    
    // Build
    ModuleFactory factory;
    auto buildResult = factory.build(parseResult.description);
    
    // Print issues
    for (const auto& issue : buildResult.issues) {
        const char* prefix = 
            issue.level == BuildIssue::Level::Error ? "ERROR" :
            issue.level == BuildIssue::Level::Warning ? "WARN" : "INFO";
        std::cerr << "[" << prefix << "] " << issue.message << "\n";
    }
    
    if (!buildResult.success()) {
        return 1;
    }
    
    // Run
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    buildResult.pipeline->init();
    buildResult.pipeline->run_all_threaded();
    
    std::cout << "Pipeline running. Press Ctrl+C to stop.\n";
    
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    buildResult.pipeline->stop();
    buildResult.pipeline->term();
    buildResult.pipeline->wait_for_all();
    
    std::cout << "Pipeline stopped.\n";
    return 0;
}

int cmdListModules(const std::string& category, const std::string& tag) {
    auto& registry = ModuleRegistry::instance();
    
    std::vector<std::string> modules;
    
    if (!tag.empty()) {
        modules = registry.getModulesByTag(tag);
    } else if (!category.empty()) {
        modules = registry.getModulesByCategory(parseCategory(category));
    } else {
        modules = registry.getAllModules();
    }
    
    for (const auto& name : modules) {
        auto* info = registry.getModule(name);
        std::cout << name;
        if (info) {
            std::cout << " [" << categoryToString(info->category) << "]";
        }
        std::cout << "\n";
    }
    
    return 0;
}

int cmdDescribe(const std::string& moduleName) {
    auto& registry = ModuleRegistry::instance();
    auto* info = registry.getModule(moduleName);
    
    if (!info) {
        std::cerr << "Unknown module: " << moduleName << "\n";
        return 1;
    }
    
    std::cout << "# " << info->name << "\n\n";
    std::cout << "Category: " << categoryToString(info->category) << "\n";
    std::cout << "Version: " << info->version << "\n";
    std::cout << "Tags: " << join(info->tags, ", ") << "\n";
    std::cout << "\n" << info->description << "\n";
    
    std::cout << "\n## Inputs\n";
    for (const auto& pin : info->inputs) {
        std::cout << "  - " << pin.name << " [" << join(pin.frame_types, ", ") << "]";
        if (pin.required) std::cout << " (required)";
        std::cout << "\n";
    }
    
    std::cout << "\n## Outputs\n";
    for (const auto& pin : info->outputs) {
        std::cout << "  - " << pin.name << " [" << join(pin.frame_types, ", ") << "]\n";
    }
    
    std::cout << "\n## Properties\n";
    for (const auto& prop : info->properties) {
        std::cout << "  - " << prop.name << " (" << prop.type << ")";
        std::cout << " [" << prop.mutability << "]";
        std::cout << " default=" << prop.default_value;
        if (!prop.min_value.empty()) {
            std::cout << " range=[" << prop.min_value << "," << prop.max_value << "]";
        }
        std::cout << "\n";
    }
    
    return 0;
}

void printUsage() {
    std::cout << R"(
Usage: aprapipes <command> [options]

Commands:
  validate <file.toml>           Validate pipeline definition
  run <file.toml>                Run pipeline
  list-modules                   List all registered modules
  describe <module>              Show module details

Options for 'run':
  --set <module.prop>=<value>    Override property value

Options for 'list-modules':
  --category <cat>               Filter by category (source, sink, transform, analytics)
  --tag <tag>                    Filter by tag (decoder, encoder, nvidia, etc.)
)";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage();
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "validate" && argc >= 3) {
        return cmdValidate(argv[2]);
    }
    else if (command == "run" && argc >= 3) {
        std::map<std::string, std::string> overrides;
        // Parse --set arguments...
        return cmdRun(argv[2], overrides);
    }
    else if (command == "list-modules") {
        std::string category, tag;
        // Parse --category, --tag arguments...
        return cmdListModules(category, tag);
    }
    else if (command == "describe" && argc >= 3) {
        return cmdDescribe(argv[2]);
    }
    else {
        printUsage();
        return 1;
    }
}
```

### Test Strategy
Create shell script tests:
```bash
# test_cli.sh
set -e

# Test validate
./aprapipes validate test/data/pipelines/valid.toml
echo "✓ validate passed"

# Test validate with error
./aprapipes validate test/data/pipelines/invalid.toml && exit 1 || echo "✓ validate error detection passed"

# Test list-modules
./aprapipes list-modules | grep -q "FileReaderModule"
echo "✓ list-modules passed"

# Test describe
./aprapipes describe FileReaderModule | grep -q "Source"
echo "✓ describe passed"
```

### Test File Location
```
base/test/cli_tests.cpp          # Unit tests for parsing
base/test/scripts/test_cli.sh    # Integration tests
```

---

## Definition of Done
- [ ] All commands implemented and working
- [ ] Signal handling for clean shutdown
- [ ] Exit codes are correct
- [ ] Help text is clear
- [ ] Unit tests pass
- [ ] Integration test script passes
- [ ] Code reviewed and merged
