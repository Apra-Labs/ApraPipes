# C2-C5: Validator Enhancement Tasks

These are **incremental improvements** to the validator shell. They are NOT blocking and can be done in parallel with other work or deferred to later sprints.

---

# C2: Validator - Module Existence Checks

**Sprint:** 2 (Week 3-4) or later  
**Priority:** P2 - Medium  
**Effort:** 1 day  
**Depends On:** C1 (Validator Shell), A2 (Module Registry)  
**Blocks:** Nothing (quality improvement)  

## Description

Add actual module validation to the validator:
- Check if module type exists in registry
- Warn on version mismatch
- Check category constraints (future)

## Implementation

```cpp
PipelineValidator::Result PipelineValidator::validateModules(const PipelineDescription& desc) const {
    Result result;
    auto& registry = ModuleRegistry::instance();
    
    for (const auto& module : desc.modules) {
        // Check module exists
        if (!registry.hasModule(module.module_type)) {
            result.issues.push_back({
                ValidationIssue::Level::Error,
                "E001",
                "modules." + module.instance_id,
                "Unknown module type: " + module.module_type
            });
            continue;
        }
        
        // Get module info for further checks
        const auto* info = registry.getModule(module.module_type);
        
        // Check version (if specified in pipeline)
        // TODO: Add version field to ModuleInstance
        
        result.issues.push_back({
            ValidationIssue::Level::Info,
            "I010",
            "modules." + module.instance_id,
            "Module validated: " + module.module_type
        });
    }
    
    return result;
}
```

## Acceptance Criteria
- [ ] Unknown module type returns Error
- [ ] Valid modules log Info
- [ ] All existing tests still pass

---

# C3: Validator - Property Checks

**Sprint:** 2-3 (Week 3-4) or later  
**Priority:** P2 - Medium  
**Effort:** 2 days  
**Depends On:** C2  
**Blocks:** Nothing  

## Description

Add property validation:
- Check property names exist in module metadata
- Check property types match
- Check numeric ranges (min/max)
- Check regex patterns
- Check enum values
- Warn on missing optional properties

## Implementation

```cpp
PipelineValidator::Result PipelineValidator::validateProperties(const PipelineDescription& desc) const {
    Result result;
    auto& registry = ModuleRegistry::instance();
    
    for (const auto& module : desc.modules) {
        const auto* info = registry.getModule(module.module_type);
        if (!info) continue;  // Already caught in module validation
        
        // Build property lookup
        std::map<std::string, const ModuleInfo::PropInfo*> propMap;
        for (const auto& prop : info->properties) {
            propMap[prop.name] = &prop;
        }
        
        // Check each provided property
        for (const auto& [name, value] : module.properties) {
            auto it = propMap.find(name);
            if (it == propMap.end()) {
                result.issues.push_back({
                    ValidationIssue::Level::Warning,
                    "W010",
                    "modules." + module.instance_id + ".props." + name,
                    "Unknown property: " + name
                });
                continue;
            }
            
            const auto* propInfo = it->second;
            
            // Type check
            if (!checkPropertyType(value, propInfo->type)) {
                result.issues.push_back({
                    ValidationIssue::Level::Error,
                    "E011",
                    "modules." + module.instance_id + ".props." + name,
                    "Type mismatch: expected " + propInfo->type
                });
                continue;
            }
            
            // Range check for numerics
            if (propInfo->type == "int" || propInfo->type == "float") {
                if (!checkRange(value, propInfo)) {
                    result.issues.push_back({
                        ValidationIssue::Level::Error,
                        "E012",
                        "modules." + module.instance_id + ".props." + name,
                        "Value out of range [" + propInfo->min_value + ", " + 
                            propInfo->max_value + "]"
                    });
                }
            }
            
            // Regex check for strings
            if (propInfo->type == "string" && !propInfo->regex_pattern.empty()) {
                if (!checkRegex(std::get<std::string>(value), propInfo->regex_pattern)) {
                    result.issues.push_back({
                        ValidationIssue::Level::Error,
                        "E013",
                        "modules." + module.instance_id + ".props." + name,
                        "Value does not match pattern: " + propInfo->regex_pattern
                    });
                }
            }
            
            // Enum check
            if (propInfo->type == "enum") {
                if (!checkEnum(std::get<std::string>(value), propInfo->enum_values)) {
                    result.issues.push_back({
                        ValidationIssue::Level::Error,
                        "E014",
                        "modules." + module.instance_id + ".props." + name,
                        "Invalid enum value. Allowed: " + join(propInfo->enum_values, ", ")
                    });
                }
            }
        }
        
        // Check for missing required properties (that have no default)
        // For MVP, all properties have defaults, so this is Info level
        for (const auto& prop : info->properties) {
            if (module.properties.find(prop.name) == module.properties.end()) {
                result.issues.push_back({
                    ValidationIssue::Level::Info,
                    "I011",
                    "modules." + module.instance_id + ".props." + prop.name,
                    "Using default value: " + prop.default_value
                });
            }
        }
    }
    
    return result;
}
```

## Acceptance Criteria
- [ ] Unknown property returns Warning
- [ ] Type mismatch returns Error
- [ ] Out of range returns Error
- [ ] Regex mismatch returns Error
- [ ] Invalid enum returns Error
- [ ] Missing property with default returns Info

---

# C4: Validator - Connection Checks

**Sprint:** 3 (Week 4+) or later  
**Priority:** P2 - Medium  
**Effort:** 2 days  
**Depends On:** C1, A2, A3 (FrameType Registry)  
**Blocks:** Nothing  

## Description

Add connection validation:
- Check source/destination modules exist
- Check pins exist on modules
- Check frame type compatibility
- Check for duplicate connections to same input
- Check required input pins are connected

## Implementation

```cpp
PipelineValidator::Result PipelineValidator::validateConnections(const PipelineDescription& desc) const {
    Result result;
    auto& modRegistry = ModuleRegistry::instance();
    auto& ftRegistry = FrameTypeRegistry::instance();
    
    // Track connected input pins
    std::set<std::string> connectedInputs;  // "module.pin"
    
    for (const auto& conn : desc.connections) {
        std::string connStr = conn.from_module + "." + conn.from_pin + 
            " -> " + conn.to_module + "." + conn.to_pin;
        
        // Check source module exists
        auto srcIt = std::find_if(desc.modules.begin(), desc.modules.end(),
            [&](const auto& m) { return m.instance_id == conn.from_module; });
        
        if (srcIt == desc.modules.end()) {
            result.issues.push_back({
                ValidationIssue::Level::Error,
                "E020",
                "connections",
                "Unknown source module: " + conn.from_module
            });
            continue;
        }
        
        // Check destination module exists
        auto dstIt = std::find_if(desc.modules.begin(), desc.modules.end(),
            [&](const auto& m) { return m.instance_id == conn.to_module; });
        
        if (dstIt == desc.modules.end()) {
            result.issues.push_back({
                ValidationIssue::Level::Error,
                "E022",
                "connections",
                "Unknown destination module: " + conn.to_module
            });
            continue;
        }
        
        // Check pins exist and get frame types
        const auto* srcInfo = modRegistry.getModule(srcIt->module_type);
        const auto* dstInfo = modRegistry.getModule(dstIt->module_type);
        
        const ModuleInfo::PinInfo* srcPin = findPin(srcInfo->outputs, conn.from_pin);
        const ModuleInfo::PinInfo* dstPin = findPin(dstInfo->inputs, conn.to_pin);
        
        if (!srcPin) {
            result.issues.push_back({
                ValidationIssue::Level::Error,
                "E021",
                "connections",
                "Unknown output pin: " + conn.from_module + "." + conn.from_pin
            });
            continue;
        }
        
        if (!dstPin) {
            result.issues.push_back({
                ValidationIssue::Level::Error,
                "E023",
                "connections",
                "Unknown input pin: " + conn.to_module + "." + conn.to_pin
            });
            continue;
        }
        
        // Check frame type compatibility
        bool compatible = false;
        for (const auto& outType : srcPin->frame_types) {
            for (const auto& inType : dstPin->frame_types) {
                if (ftRegistry.isCompatible(outType, inType)) {
                    compatible = true;
                    break;
                }
            }
            if (compatible) break;
        }
        
        if (!compatible) {
            result.issues.push_back({
                ValidationIssue::Level::Error,
                "E024",
                "connections",
                "Frame type incompatible: " + connStr + 
                    " (output: " + join(srcPin->frame_types, ",") +
                    ", input: " + join(dstPin->frame_types, ",") + ")"
            });
        }
        
        // Check for duplicate input connections
        std::string inputKey = conn.to_module + "." + conn.to_pin;
        if (connectedInputs.count(inputKey)) {
            result.issues.push_back({
                ValidationIssue::Level::Error,
                "E025",
                "connections",
                "Duplicate connection to input: " + inputKey
            });
        }
        connectedInputs.insert(inputKey);
    }
    
    // Check required inputs are connected
    for (const auto& module : desc.modules) {
        const auto* info = modRegistry.getModule(module.module_type);
        if (!info) continue;
        
        for (const auto& pin : info->inputs) {
            if (pin.required) {
                std::string inputKey = module.instance_id + "." + pin.name;
                if (!connectedInputs.count(inputKey)) {
                    result.issues.push_back({
                        ValidationIssue::Level::Error,
                        "E026",
                        "connections",
                        "Required input not connected: " + inputKey
                    });
                }
            }
        }
    }
    
    return result;
}
```

## Acceptance Criteria
- [ ] Unknown source/dest module returns Error
- [ ] Unknown pin returns Error
- [ ] Frame type incompatibility returns Error
- [ ] Duplicate input connection returns Error
- [ ] Unconnected required input returns Error

---

# C5: Validator - Graph Checks

**Sprint:** 3 (Week 4+) or later  
**Priority:** P3 - Low  
**Effort:** 1 day  
**Depends On:** C4  
**Blocks:** Nothing  

## Description

Add graph structure validation:
- Check pipeline has at least one source
- Check graph is a DAG (no cycles)
- Warn on orphan modules (no connections)

## Implementation

```cpp
PipelineValidator::Result PipelineValidator::validateGraph(const PipelineDescription& desc) const {
    Result result;
    auto& registry = ModuleRegistry::instance();
    
    // Check for at least one source
    bool hasSource = false;
    for (const auto& module : desc.modules) {
        const auto* info = registry.getModule(module.module_type);
        if (info && info->category == ModuleCategory::Source) {
            hasSource = true;
            break;
        }
    }
    
    if (!hasSource) {
        result.issues.push_back({
            ValidationIssue::Level::Error,
            "E030",
            "pipeline",
            "Pipeline has no source module"
        });
    }
    
    // Build adjacency list
    std::map<std::string, std::vector<std::string>> graph;
    std::set<std::string> allModules;
    std::set<std::string> connectedModules;
    
    for (const auto& module : desc.modules) {
        allModules.insert(module.instance_id);
        graph[module.instance_id] = {};
    }
    
    for (const auto& conn : desc.connections) {
        graph[conn.from_module].push_back(conn.to_module);
        connectedModules.insert(conn.from_module);
        connectedModules.insert(conn.to_module);
    }
    
    // Check for cycles using DFS
    std::set<std::string> visited, inStack;
    std::function<bool(const std::string&)> hasCycle = [&](const std::string& node) -> bool {
        visited.insert(node);
        inStack.insert(node);
        
        for (const auto& neighbor : graph[node]) {
            if (!visited.count(neighbor)) {
                if (hasCycle(neighbor)) return true;
            } else if (inStack.count(neighbor)) {
                return true;
            }
        }
        
        inStack.erase(node);
        return false;
    };
    
    for (const auto& module : desc.modules) {
        if (!visited.count(module.instance_id)) {
            if (hasCycle(module.instance_id)) {
                result.issues.push_back({
                    ValidationIssue::Level::Error,
                    "E031",
                    "pipeline",
                    "Pipeline contains a cycle"
                });
                break;
            }
        }
    }
    
    // Warn on orphan modules
    for (const auto& module : allModules) {
        if (!connectedModules.count(module)) {
            result.issues.push_back({
                ValidationIssue::Level::Warning,
                "W030",
                "modules." + module,
                "Module has no connections (orphan)"
            });
        }
    }
    
    return result;
}
```

## Acceptance Criteria
- [ ] No source module returns Error
- [ ] Cycle in graph returns Error
- [ ] Orphan module returns Warning
- [ ] Valid DAG passes

---

## Definition of Done (for each validator task)
- [ ] Validation logic implemented
- [ ] Unit tests for each error case
- [ ] Integration test with real pipeline
- [ ] Does not block factory/CLI
- [ ] Code reviewed and merged
