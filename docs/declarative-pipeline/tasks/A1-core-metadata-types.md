# A1: Core Metadata Types (Metadata.h)

**Sprint:** 1 (Week 1-2)  
**Priority:** P0 - Critical Path  
**Effort:** 3 days  
**Depends On:** None (Start immediately)  
**Blocks:** A2, A3, All module/frame metadata tasks  

## Description

Create the foundational type system for declarative metadata. This defines `PinDef`, `PropDef`, `AttrDef`, `ModuleCategory`, and the tag system that all modules and frame types will use.

This is the **first task** in the critical path and unblocks everything else.

## Acceptance Criteria

### Unit Tests
- [ ] `PinDef` can be constructed with name, frame_types, required flag, description
- [ ] `PropDef::Int()` creates integer property with range validation metadata
- [ ] `PropDef::Float()` creates float property with range validation metadata
- [ ] `PropDef::Bool()` creates boolean property
- [ ] `PropDef::String()` creates string property with optional regex pattern
- [ ] `PropDef::Enum()` creates enum property with allowed values list
- [ ] `PropDef::DynamicInt/Float/Bool()` create properties with `Mutability::Dynamic`
- [ ] `AttrDef` factory methods work for all types
- [ ] `ModuleCategory` enum has all 6 values: Source, Sink, Transform, Analytics, Controller, Utility
- [ ] All types are `constexpr` constructible (compile-time validation)
- [ ] Tags can be declared as `std::array<std::string_view, N>`

### Behavioral (Given/When/Then)

**Scenario: Define module metadata with mixed properties**
```
Given a module class with nested Metadata struct
When I declare static constexpr properties array with Int, DynamicFloat, Enum
Then compilation succeeds
And all property metadata is accessible at compile time
```

**Scenario: Define pin with multiple accepted frame types**
```
Given a PinDef declaration
When I specify frame_types = {"RawImagePlanar", "RawImagePacked"}
Then both types are stored in the initializer_list
And required defaults to true
```

**Scenario: Property range validation metadata**
```
Given PropDef::Int("device_id", 0, 0, 7, "CUDA device")
When I access min_value and max_value
Then min_value == "0" and max_value == "7"
And type == Type::Int
And mutability == Mutability::Static
```

### Requirements
- All types must be header-only (no .cpp file needed for Metadata.h)
- Must compile with C++17 (ApraPipes current standard)
- All factory methods must be `constexpr`
- `string_view` used for all string data (no runtime allocation)
- Must support designated initializers for clean syntax: `PinDef{.name = "input", ...}`
- Tags declared as `std::array` not `std::vector` (constexpr compatible)

## Implementation Notes for Claude Code Agents

### File Location
```
base/include/core/Metadata.h
```

### Key Implementation Details

1. **Use string_view, not string**
```cpp
// CORRECT
static constexpr std::string_view name = "MyModule";

// WRONG - not constexpr
static constexpr std::string name = "MyModule";
```

2. **Property factory methods pattern**
```cpp
static constexpr PropDef Int(
    std::string_view name,
    int default_val,
    int min_val, 
    int max_val,
    std::string_view desc = "",
    Mutability mut = Mutability::Static
) {
    PropDef p;
    p.name = name;
    p.type = Type::Int;
    p.mutability = mut;
    // Store as string_view for uniform handling
    // Actual conversion happens at runtime in validator/factory
    p.default_value = /* need to handle int->string_view */;
    p.min_value = /* ... */;
    p.max_value = /* ... */;
    p.description = desc;
    return p;
}
```

3. **Challenge: int to string_view at constexpr**

Option A: Store numeric values separately
```cpp
struct PropDef {
    std::string_view name;
    Type type;
    // For numeric types
    int64_t int_default = 0;
    int64_t int_min = 0;
    int64_t int_max = 0;
    double float_default = 0.0;
    // ... etc
};
```

Option B: Use string literals in factory calls
```cpp
PropDef::Int("device_id", "0", "0", "7", "CUDA device")
```

**Recommend Option A** - type safety at compile time.

4. **Tags as std::array**
```cpp
// In module Metadata struct:
static constexpr std::array tags = {
    std::string_view{"decoder"},
    std::string_view{"h264"},
    std::string_view{"nvidia"}
};
// Or simpler with CTAD:
static constexpr std::array<std::string_view, 3> tags = {"decoder", "h264", "nvidia"};
```

### Test File Location
```
base/test/metadata_tests.cpp
```

### Reference
- RFC Section 2: Core C++ Infrastructure
- Existing pattern: `base/include/FrameMetadata.h` for how ApraPipes structures metadata

---

## Definition of Done
- [ ] `Metadata.h` compiles standalone
- [ ] All unit tests pass
- [ ] Can write a sample module Metadata struct using all types
- [ ] Code reviewed and merged to feature branch
