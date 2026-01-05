# ApraPipes JavaScript Interface - Implementation Plan

> **Status:** READY FOR APPROVAL
> **Created:** 2026-01-05
> **Author:** Claude Code Agent
> **Package:** `@apralabs/aprapipes`

## Executive Summary

This document outlines the plan to evolve ApraPipes from TOML-based declarative pipelines to a JSON-based approach with a native Node.js addon (`aprapipes.node`). This enables:

1. **JSON Pipeline Descriptions** - More widely understood, programmatically generated
2. **Native Node.js Integration** - Direct C++ binding via N-API
3. **JavaScript API** - Create, validate, run pipelines from JS/TS
4. **Event System** - Pipeline/module events delivered to JS callbacks
5. **npm Package** - Distributable as `aprapipes` npm package

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [JSON Schema Design](#2-json-schema-design)
3. [JavaScript API Design](#3-javascript-api-design)
4. [Node Addon Implementation](#4-node-addon-implementation)
5. [Event System](#5-event-system)
6. [Migration from TOML](#6-migration-from-toml)
7. [Implementation Tasks](#7-implementation-tasks)
8. [File Structure](#8-file-structure)

---

## 1. Architecture Overview

### Current Architecture (TOML)

```
┌─────────────┐     ┌─────────────┐     ┌───────────────┐     ┌──────────┐
│ TOML File   │────▶│ TomlParser  │────▶│ PipelineDesc  │────▶│ Factory  │────▶ Running Pipeline
└─────────────┘     └─────────────┘     └───────────────┘     └──────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ Validator   │
                                        └─────────────┘
```

### New Architecture (JSON + Node.js)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Node.js Process                                 │
│  ┌───────────────┐     ┌─────────────────┐     ┌───────────────────────┐   │
│  │ JS/TS Code    │────▶│ aprapipes.node  │────▶│ C++ Declarative API   │   │
│  │               │◀────│ (N-API Addon)   │◀────│                       │   │
│  │ - JSON config │     │                 │     │ - JsonParser          │   │
│  │ - Callbacks   │     │ - Thread-safe   │     │ - PipelineDescription │   │
│  │ - Events      │     │   callbacks     │     │ - ModuleFactory       │   │
│  └───────────────┘     └─────────────────┘     │ - PipelineValidator   │   │
│                                                 │ - Event Bridge        │   │
│                                                 └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          ┌───────────────────┐
                          │ C++ Module Graph  │
                          │ (Threaded Runtime)│
                          └───────────────────┘
```

### Key Design Decisions

1. **JSON over TOML** - Universal format, easier programmatic generation
2. **No REST API** - Direct N-API binding (like apranvr pattern)
3. **Async-first** - Pipeline operations run in background threads
4. **Event-driven** - Module events flow to JS via thread-safe callbacks
5. **TypeScript types** - Full type definitions for IntelliSense

---

## 2. JSON Schema Design

### 2.1 Pipeline Description Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://aprapipes.io/schemas/pipeline.json",
  "title": "ApraPipes Pipeline",
  "type": "object",
  "required": ["modules", "connections"],
  "properties": {
    "pipeline": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "version": { "type": "string", "default": "1.0" },
        "#desc": { "type": "string", "description": "Pipeline description (comment)" }
      }
    },
    "settings": {
      "type": "object",
      "properties": {
        "queueSize": { "type": "integer", "default": 20 },
        "onError": { "enum": ["stop", "skip", "retry"], "default": "stop" },
        "autoStart": { "type": "boolean", "default": true }
      }
    },
    "modules": {
      "type": "object",
      "additionalProperties": { "$ref": "#/definitions/module" }
    },
    "connections": {
      "type": "array",
      "items": { "$ref": "#/definitions/connection" }
    }
  },
  "definitions": {
    "module": {
      "type": "object",
      "required": ["type"],
      "properties": {
        "type": { "type": "string" },
        "#desc": { "type": "string" },
        "props": { "type": "object" }
      }
    },
    "connection": {
      "type": "object",
      "required": ["from", "to"],
      "properties": {
        "from": { "type": "string", "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*(\\.[a-zA-Z_][a-zA-Z0-9_]*)?$" },
        "to": { "type": "string", "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*(\\.[a-zA-Z_][a-zA-Z0-9_]*)?$" },
        "#desc": { "type": "string" }
      }
    }
  }
}
```

### 2.2 Example Pipeline JSON

```json
{
  "pipeline": {
    "name": "Affine Transform Demo",
    "#desc": "Demonstrates rotation, scaling, and translation of images"
  },
  "settings": {
    "queueSize": 20,
    "onError": "stop"
  },
  "modules": {
    "generator": {
      "type": "TestSignalGenerator",
      "#desc": "Generate test frames (YUV420 planar format)",
      "props": {
        "width": 640,
        "height": 480
      }
    },
    "colorConvert": {
      "type": "ColorConversion",
      "#desc": "Convert YUV420 planar to RGB",
      "props": {
        "conversionType": "YUV420PLANAR_TO_RGB"
      }
    },
    "transform": {
      "type": "AffineTransform",
      "#desc": "Apply 30 degree rotation with 0.9x scale",
      "props": {
        "angle": 30.0,
        "scale": 0.9
      }
    },
    "encoder": {
      "type": "ImageEncoderCV",
      "#desc": "Encode to JPEG format"
    },
    "writer": {
      "type": "FileWriterModule",
      "props": {
        "strFullFileNameWithPattern": "./output/affine_????.jpg"
      }
    }
  },
  "connections": [
    { "from": "generator", "to": "colorConvert" },
    { "from": "colorConvert", "to": "transform" },
    { "from": "transform", "to": "encoder" },
    { "from": "encoder", "to": "writer" }
  ]
}
```

### 2.3 Comment Convention

Since JSON doesn't support comments, we use `#desc` properties:

```json
{
  "#desc": "This is a comment about the parent object",
  "type": "ModuleName",
  "props": {
    "#desc": "Comments can appear in props too",
    "width": 640
  }
}
```

The `#desc` prefix is:
- Ignored by the parser (filtered out)
- Preserved in JSON for documentation
- Can be extracted for auto-generated documentation

---

## 3. JavaScript API Design

### 3.1 Core API

```typescript
// aprapipes.d.ts - TypeScript definitions

declare module 'aprapipes' {

  // ============================================================
  // Pipeline Lifecycle
  // ============================================================

  /**
   * Create a pipeline from JSON configuration
   * @param config Pipeline configuration object or JSON string
   * @returns Pipeline handle for further operations
   */
  export function createPipeline(config: PipelineConfig | string): Pipeline;

  /**
   * Validate a pipeline configuration without creating it
   * @param config Pipeline configuration to validate
   * @returns Validation result with errors/warnings
   */
  export function validatePipeline(config: PipelineConfig | string): ValidationResult;

  /**
   * Parse a JSON file into pipeline configuration
   * @param filePath Path to JSON file
   * @returns Parsed configuration object
   */
  export function parseFile(filePath: string): PipelineConfig;

  // ============================================================
  // Module Registry
  // ============================================================

  /**
   * List all registered modules
   * @param filter Optional filter by category or tag
   */
  export function listModules(filter?: ModuleFilter): ModuleInfo[];

  /**
   * Get detailed information about a module
   * @param moduleType Module type name
   */
  export function describeModule(moduleType: string): ModuleInfo | null;

  /**
   * Get JSON schema for module properties
   * @param moduleType Module type name
   */
  export function getModuleSchema(moduleType: string): object;

  // ============================================================
  // Types
  // ============================================================

  export interface PipelineConfig {
    pipeline?: {
      name?: string;
      version?: string;
      '#desc'?: string;
    };
    settings?: PipelineSettings;
    modules: Record<string, ModuleConfig>;
    connections: ConnectionConfig[];
  }

  export interface PipelineSettings {
    queueSize?: number;
    onError?: 'stop' | 'skip' | 'retry';
    autoStart?: boolean;
  }

  export interface ModuleConfig {
    type: string;
    '#desc'?: string;
    props?: Record<string, any>;
  }

  export interface ConnectionConfig {
    from: string;
    to: string;
    '#desc'?: string;
  }

  export interface ValidationResult {
    valid: boolean;
    errors: ValidationIssue[];
    warnings: ValidationIssue[];
    suggestions: ValidationSuggestion[];
  }

  export interface ValidationIssue {
    code: string;
    message: string;
    location?: string;
    line?: number;
  }

  export interface ValidationSuggestion {
    issue: string;
    suggestion: string;
    fix?: string;  // JSON snippet to fix the issue
  }

  // ============================================================
  // Pipeline Handle
  // ============================================================

  export interface Pipeline {
    /** Pipeline unique identifier */
    readonly id: string;

    /** Current pipeline status */
    readonly status: PipelineStatus;

    /** Initialize the pipeline (required before run) */
    init(): Promise<boolean>;

    /** Start pipeline execution (async, returns immediately) */
    run(): Promise<void>;

    /** Stop pipeline execution */
    stop(): Promise<void>;

    /** Pause pipeline (frames stop flowing) */
    pause(): void;

    /** Resume paused pipeline */
    play(): void;

    /** Process single frame (when paused) */
    step(): Promise<boolean>;

    /** Terminate and cleanup */
    terminate(): Promise<void>;

    /** Get module by instance ID */
    getModule(instanceId: string): ModuleHandle | null;

    /** Get all module handles */
    getModules(): ModuleHandle[];

    /** Get pipeline statistics */
    getStats(): PipelineStats;

    /** Register event listener */
    on(event: PipelineEventType, callback: PipelineEventCallback): void;

    /** Remove event listener */
    off(event: PipelineEventType, callback: PipelineEventCallback): void;
  }

  export type PipelineStatus =
    | 'created'
    | 'initialized'
    | 'running'
    | 'paused'
    | 'stopped'
    | 'terminated'
    | 'error';

  // ============================================================
  // Module Handle (Runtime Access)
  // ============================================================

  export interface ModuleHandle {
    /** Module instance ID (from config) */
    readonly instanceId: string;

    /** Module type name */
    readonly type: string;

    /** Module status */
    readonly status: ModuleStatus;

    /** Get current property value */
    getProperty(name: string): any;

    /** Set property value (for dynamic properties) */
    setProperty(name: string, value: any): boolean;

    /** Get all current property values */
    getProperties(): Record<string, any>;

    /** Get module statistics */
    getStats(): ModuleStats;
  }

  export type ModuleStatus = 'initialized' | 'running' | 'paused' | 'stopped' | 'error';

  export interface ModuleStats {
    framesProcessed: number;
    avgProcessingTimeMs: number;
    avgPipelineTimeMs: number;
    currentFps: number;
    queueDepth: number;
  }

  export interface PipelineStats {
    status: PipelineStatus;
    uptimeMs: number;
    modules: Record<string, ModuleStats>;
  }

  // ============================================================
  // Events
  // ============================================================

  export type PipelineEventType =
    | 'started'
    | 'stopped'
    | 'paused'
    | 'resumed'
    | 'error'
    | 'moduleError'
    | 'stats'
    | 'endOfStream';

  export type PipelineEventCallback = (event: PipelineEvent) => void;

  export interface PipelineEvent {
    type: PipelineEventType;
    timestamp: Date;
    pipelineId: string;
    moduleId?: string;
    data?: any;
  }

  export interface ErrorEvent extends PipelineEvent {
    type: 'error' | 'moduleError';
    errorCode: number;
    errorMessage: string;
  }

  export interface StatsEvent extends PipelineEvent {
    type: 'stats';
    data: PipelineStats;
  }

  // ============================================================
  // Module Registry Types
  // ============================================================

  export interface ModuleInfo {
    type: string;
    category: ModuleCategory;
    description: string;
    tags: string[];
    inputs: PinInfo[];
    outputs: PinInfo[];
    properties: PropertyInfo[];
  }

  export type ModuleCategory =
    | 'source'
    | 'sink'
    | 'transform'
    | 'analytics'
    | 'control'
    | 'utility';

  export interface PinInfo {
    name: string;
    frameTypes: string[];
    required: boolean;
  }

  export interface PropertyInfo {
    name: string;
    type: 'int' | 'float' | 'bool' | 'string' | 'enum';
    required: boolean;
    default?: any;
    description?: string;
    enumValues?: string[];
    min?: number;
    max?: number;
    dynamic: boolean;  // Can be changed at runtime
  }

  export interface ModuleFilter {
    category?: ModuleCategory;
    tag?: string;
    search?: string;
  }
}
```

### 3.2 Usage Examples

```typescript
import * as aprapipes from 'aprapipes';

// Example 1: Create and run pipeline from JSON
const config = {
  modules: {
    source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
    sink: { type: 'StatSink' }
  },
  connections: [
    { from: 'source', to: 'sink' }
  ]
};

const pipeline = aprapipes.createPipeline(config);

pipeline.on('error', (event) => {
  console.error(`Pipeline error: ${event.errorMessage}`);
});

pipeline.on('stats', (event) => {
  console.log(`FPS: ${event.data.modules.source.currentFps}`);
});

await pipeline.init();
await pipeline.run();

// Later...
await pipeline.stop();
await pipeline.terminate();


// Example 2: Validate without running
const result = aprapipes.validatePipeline(config);
if (!result.valid) {
  result.errors.forEach(e => console.error(e.message));
  result.suggestions.forEach(s => console.log(`Suggestion: ${s.suggestion}`));
}


// Example 3: Dynamic property changes
const ptz = pipeline.getModule('ptz');
if (ptz) {
  ptz.setProperty('roiX', 0.25);
  ptz.setProperty('roiY', 0.25);
  ptz.setProperty('roiWidth', 0.5);
  ptz.setProperty('roiHeight', 0.5);
}


// Example 4: Module discovery
const transforms = aprapipes.listModules({ category: 'transform' });
transforms.forEach(m => {
  console.log(`${m.type}: ${m.description}`);
});

const schema = aprapipes.getModuleSchema('AffineTransform');
console.log(JSON.stringify(schema, null, 2));
```

---

## 4. Node Addon Implementation

### 4.1 Addon Structure

Based on apranvr patterns, the addon will use:
- **node-addon-api** - C++ wrapper for N-API
- **napi-thread-safe-callback** - Thread-safe JS callback invocation
- **cmake-js** - Build system integration

### 4.2 C++ Binding Layer

```cpp
// base/bindings/node/addon.cpp

#include <napi.h>
#include "napi-thread-safe-callback.hpp"
#include "declarative/JsonParser.h"
#include "declarative/ModuleFactory.h"
#include "declarative/PipelineValidator.h"
#include "declarative/ModuleRegistry.h"

// Global state
static bool g_initialized = false;

// ============================================================
// Pipeline Management
// ============================================================

Napi::Value CreatePipeline(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    try {
        // Parse JSON config (string or object)
        std::string jsonStr;
        if (info[0].IsString()) {
            jsonStr = info[0].As<Napi::String>().Utf8Value();
        } else if (info[0].IsObject()) {
            // Stringify object to JSON
            Napi::Object JSON = env.Global().Get("JSON").As<Napi::Object>();
            Napi::Function stringify = JSON.Get("stringify").As<Napi::Function>();
            jsonStr = stringify.Call(JSON, {info[0]}).As<Napi::String>().Utf8Value();
        }

        // Parse and build pipeline
        auto desc = JsonParser::parseString(jsonStr);
        ModuleFactory::Options opts;
        auto result = ModuleFactory::build(desc, opts);

        if (result.hasErrors()) {
            Napi::Error::New(env, result.formatIssues()).ThrowAsJavaScriptException();
            return env.Null();
        }

        // Create Pipeline wrapper object
        // ... (return handle with methods)

    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value ValidatePipeline(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    try {
        std::string jsonStr = /* extract from info[0] */;
        auto desc = JsonParser::parseString(jsonStr);

        PipelineValidator::Options opts;
        auto result = PipelineValidator::validate(desc, opts);

        // Build result object
        Napi::Object resultObj = Napi::Object::New(env);
        resultObj.Set("valid", !result.hasErrors());

        // Convert errors to JS array
        Napi::Array errors = Napi::Array::New(env);
        for (size_t i = 0; i < result.issues.size(); i++) {
            if (result.issues[i].severity == Issue::Severity::Error) {
                Napi::Object err = Napi::Object::New(env);
                err.Set("code", result.issues[i].code);
                err.Set("message", result.issues[i].message);
                errors.Set(i, err);
            }
        }
        resultObj.Set("errors", errors);

        // ... warnings, suggestions

        return resultObj;
    } catch (...) { /* error handling */ }
}

// ============================================================
// Module Registry
// ============================================================

Napi::Value ListModules(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    auto& registry = ModuleRegistry::instance();
    auto modules = registry.listAll();

    Napi::Array result = Napi::Array::New(env, modules.size());
    for (size_t i = 0; i < modules.size(); i++) {
        Napi::Object mod = Napi::Object::New(env);
        mod.Set("type", modules[i].name);
        mod.Set("category", categoryToString(modules[i].category));
        mod.Set("description", modules[i].description);
        // ... inputs, outputs, properties
        result.Set(i, mod);
    }

    return result;
}

// ============================================================
// Pipeline Handle Class
// ============================================================

class PipelineWrapper : public Napi::ObjectWrap<PipelineWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    PipelineWrapper(const Napi::CallbackInfo& info);

    // Methods bound to JS
    Napi::Value Init(const Napi::CallbackInfo& info);
    Napi::Value Run(const Napi::CallbackInfo& info);
    Napi::Value Stop(const Napi::CallbackInfo& info);
    Napi::Value GetStatus(const Napi::CallbackInfo& info);
    Napi::Value GetModule(const Napi::CallbackInfo& info);
    Napi::Value On(const Napi::CallbackInfo& info);  // Event registration

private:
    boost::shared_ptr<PipeLine> m_pipeline;
    std::map<std::string, std::shared_ptr<ThreadSafeCallback>> m_callbacks;

    void emitEvent(const std::string& type, const std::string& data);
};

// ============================================================
// Module Initialization
// ============================================================

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Initialize registries
    ensureBuiltinModulesRegistered();
    ensureBuiltinFrameTypesRegistered();
    g_initialized = true;

    // Export functions
    exports.Set("createPipeline", Napi::Function::New(env, CreatePipeline));
    exports.Set("validatePipeline", Napi::Function::New(env, ValidatePipeline));
    exports.Set("parseFile", Napi::Function::New(env, ParseFile));
    exports.Set("listModules", Napi::Function::New(env, ListModules));
    exports.Set("describeModule", Napi::Function::New(env, DescribeModule));
    exports.Set("getModuleSchema", Napi::Function::New(env, GetModuleSchema));

    // Export Pipeline class
    PipelineWrapper::Init(env, exports);

    return exports;
}

NODE_API_MODULE(aprapipes, Init)
```

### 4.3 Event Bridge

```cpp
// Event callback registration and emission

void PipelineWrapper::registerHealthCallback() {
    // Register with all modules
    for (auto& module : m_pipeline->getModules()) {
        module->registerHealthCallback([this](const APHealthObject& health) {
            // Thread-safe callback to JS
            if (m_callbacks.count("stats")) {
                m_callbacks["stats"]->call([health](Napi::Env env, std::vector<napi_value>& args) {
                    Napi::Object event = Napi::Object::New(env);
                    event.Set("type", "stats");
                    event.Set("moduleId", health.getModuleId());
                    event.Set("timestamp", health.getTimestamp());
                    args.push_back(event);
                });
            }
        });

        module->registerErrorCallback([this](const APErrorObject& error) {
            if (m_callbacks.count("moduleError")) {
                m_callbacks["moduleError"]->call([error](Napi::Env env, std::vector<napi_value>& args) {
                    Napi::Object event = Napi::Object::New(env);
                    event.Set("type", "moduleError");
                    event.Set("moduleId", error.getModuleId());
                    event.Set("errorCode", error.getErrorCode());
                    event.Set("errorMessage", error.getErrorMessage());
                    args.push_back(event);
                });
            }
        });
    }
}
```

---

## 5. Event System

### 5.1 Events Emitted

| Event | When | Data |
|-------|------|------|
| `started` | Pipeline starts running | `{ }` |
| `stopped` | Pipeline stops | `{ reason: 'user' \| 'error' \| 'eos' }` |
| `paused` | Pipeline pauses | `{ }` |
| `resumed` | Pipeline resumes | `{ }` |
| `error` | Pipeline-level error | `{ errorCode, errorMessage }` |
| `moduleError` | Module-level error | `{ moduleId, errorCode, errorMessage }` |
| `stats` | Periodic stats update | `{ modules: {...} }` |
| `endOfStream` | Source reaches end | `{ moduleId }` |

### 5.2 Event Flow

```
C++ Module Thread          N-API Layer              Node.js Event Loop
       │                        │                          │
       │  APHealthCallback      │                          │
       ├───────────────────────▶│                          │
       │                        │  ThreadSafeCallback      │
       │                        ├─────────────────────────▶│
       │                        │                          │  emit('stats', data)
       │                        │                          ├─────────▶ User Callback
       │                        │                          │
```

---

## 6. TOML Removal & JSON Migration

### 6.1 Strategy

1. **Remove TOML completely** - Delete TomlParser and all TOML dependencies
2. **JSON only** - JsonParser becomes the sole parser
3. **Clean IR** - PipelineDescription unchanged (format-agnostic)
4. **CLI uses JSON** - All pipeline files are `.json`

### 6.2 Files to DELETE

```
base/include/declarative/TomlParser.h      # DELETE
base/src/declarative/TomlParser.cpp        # DELETE
base/test/declarative/toml_parser_tests.cpp # DELETE
docs/declarative-pipeline/examples/working/*.toml  # CONVERT to JSON, then DELETE
docs/declarative-pipeline/examples/templates/*.toml # CONVERT to JSON, then DELETE
```

### 6.3 Dependencies to REMOVE

From `base/vcpkg.json`:
```json
// REMOVE this dependency:
"tomlplusplus"
```

### 6.4 JsonParser Implementation

```cpp
// base/include/declarative/JsonParser.h

class JsonParser {
public:
    static PipelineDescription parseFile(const std::string& filePath);
    static PipelineDescription parseString(const std::string& jsonStr);

private:
    static PropertyValue toPropertyValue(const nlohmann::json& node);
    static void parsePipelineSection(const nlohmann::json& j, PipelineDescription& desc);
    static void parseModulesSection(const nlohmann::json& j, PipelineDescription& desc);
    static void parseConnectionsSection(const nlohmann::json& j, PipelineDescription& desc);
};
```

### 6.3 Conversion Tools

```bash
# Convert TOML to JSON
aprapipes_cli convert pipeline.toml --output pipeline.json

# Convert JSON to TOML
aprapipes_cli convert pipeline.json --output pipeline.toml
```

---

## 7. Implementation Tasks

### Phase 0: TOML Removal (First!)

#### Files to DELETE

| Task | Description | Files |
|------|-------------|-------|
| **R1** | Delete TomlParser files | `base/include/declarative/TomlParser.h`, `base/src/declarative/TomlParser.cpp` |
| **R2** | Delete TOML tests | `base/test/declarative/toml_parser_tests.cpp` |
| **R3** | Remove tomlplusplus from vcpkg | `base/vcpkg.json` |
| **R6** | Delete original .toml files | `docs/declarative-pipeline/examples/**/*.toml` |

#### Test Files to UPDATE (not delete)

| Task | Description | Files |
|------|-------------|-------|
| **R8** | Update integration tests | `base/test/declarative/pipeline_integration_tests.cpp` - change to use JSON pipelines |
| **R9** | Update pipeline description tests | `base/test/declarative/pipeline_description_tests.cpp` - update TOML refs in tests |
| **R10** | Update module registry tests | `base/test/declarative/module_registry_tests.cpp` - update TOML refs in comments |

#### Source Files to UPDATE

| Task | Description | Files |
|------|-------------|-------|
| **R11** | Update ModuleRegistrations | `base/src/declarative/ModuleRegistrations.cpp` - remove TomlParser include |
| **R12** | Update ModuleFactory | `base/src/declarative/ModuleFactory.cpp` - update error messages, remove TOML refs |
| **R13** | Update PipelineValidator | `base/src/declarative/PipelineValidator.cpp` - update error messages, remove TOML refs |
| **R14** | Update ModuleRegistry | `base/src/declarative/ModuleRegistry.cpp` - update comments |
| **R15** | Update CLI tool | `base/tools/aprapipes_cli.cpp` - change from TOML to JSON handling |

#### Headers to UPDATE

| Task | Description | Files |
|------|-------------|-------|
| **R16** | Update ModuleFactory header | `base/include/declarative/ModuleFactory.h` - update comments |
| **R17** | Update ModuleRegistry header | `base/include/declarative/ModuleRegistry.h` - update comments |
| **R18** | Update PipelineDescription header | `base/include/declarative/PipelineDescription.h` - update comments |

#### Other Updates

| Task | Description | Files |
|------|-------------|-------|
| **R4** | Update CMakeLists.txt | `base/CMakeLists.txt` (remove TomlParser references) |
| **R5** | Convert all .toml examples to .json | `docs/declarative-pipeline/examples/` |
| **R7** | Update integration test script | `scripts/test_declarative_pipelines.sh` - use .json files |
| **R19** | Update documentation | All `docs/declarative-pipeline/*.md` files referencing TOML |

### Phase 1: JSON Parser

| Task | Description | Files |
|------|-------------|-------|
| **J1** | Create JsonParser class | `base/include/declarative/JsonParser.h`, `base/src/declarative/JsonParser.cpp` |
| **J2** | Unit tests for JsonParser | `base/test/declarative/json_parser_tests.cpp` |
| **J3** | Update CLI to use JSON only | `base/tools/aprapipes_cli.cpp` |
| **J4** | Create JSON schema file | `docs/declarative-pipeline/schemas/pipeline.schema.json` |
| **J5** | Update all documentation | `docs/declarative-pipeline/*.md` |

### Phase 2: Node Addon Foundation (Week 2)

| Task | Description | Files |
|------|-------------|-------|
| **N1** | Add node-addon-api to vcpkg | `base/vcpkg.json` |
| **N2** | Create addon directory structure | `base/bindings/node/` |
| **N3** | Implement basic addon with Init | `base/bindings/node/addon.cpp` |
| **N4** | CMake configuration for .node build | `base/bindings/node/CMakeLists.txt` |
| **N5** | Create package.json | `package.json` |
| **N6** | Basic smoke test | `test/node/smoke.test.js` |

### Phase 3: Core JS API (Week 3)

| Task | Description | Files |
|------|-------------|-------|
| **A1** | Implement createPipeline | `base/bindings/node/addon.cpp` |
| **A2** | Implement validatePipeline | `base/bindings/node/addon.cpp` |
| **A3** | Implement listModules, describeModule | `base/bindings/node/addon.cpp` |
| **A4** | PipelineWrapper class (init, run, stop) | `base/bindings/node/pipeline_wrapper.cpp` |
| **A5** | ModuleWrapper class (getProperty, setProperty) | `base/bindings/node/module_wrapper.cpp` |
| **A6** | TypeScript definitions | `types/aprapipes.d.ts` |

### Phase 4: Event System (Week 4)

| Task | Description | Files |
|------|-------------|-------|
| **E1** | ThreadSafeCallback integration | `base/bindings/node/event_bridge.cpp` |
| **E2** | Health callback bridge | `base/bindings/node/event_bridge.cpp` |
| **E3** | Error callback bridge | `base/bindings/node/event_bridge.cpp` |
| **E4** | Pipeline lifecycle events | `base/bindings/node/pipeline_wrapper.cpp` |
| **E5** | Event listener API (on/off) | `base/bindings/node/pipeline_wrapper.cpp` |

### Phase 5: Testing & Documentation (Week 5)

| Task | Description | Files |
|------|-------------|-------|
| **T1** | Node.js unit tests | `test/node/*.test.js` |
| **T2** | Integration tests | `test/node/integration/*.test.js` |
| **T3** | Example applications | `examples/node/` |
| **T4** | API documentation | `docs/declarative-pipeline/JS_API_GUIDE.md` |
| **T5** | npm publish setup | `.npmrc`, `scripts/publish.sh` |

---

## 8. File Structure

### 8.1 New Files

```
ApraPipes/
├── package.json                           # npm package definition
├── types/
│   └── aprapipes.d.ts                     # TypeScript definitions
├── base/
│   ├── vcpkg.json                         # + node-addon-api
│   ├── include/declarative/
│   │   └── JsonParser.h                   # NEW: JSON parser
│   ├── src/declarative/
│   │   └── JsonParser.cpp                 # NEW: JSON parser impl
│   ├── bindings/
│   │   └── node/
│   │       ├── CMakeLists.txt             # Node addon build
│   │       ├── addon.cpp                  # Main addon entry
│   │       ├── pipeline_wrapper.cpp       # Pipeline class binding
│   │       ├── pipeline_wrapper.h
│   │       ├── module_wrapper.cpp         # Module class binding
│   │       ├── module_wrapper.h
│   │       └── event_bridge.cpp           # Event system
│   └── test/
│       └── declarative/
│           └── json_parser_tests.cpp      # JSON parser tests
├── test/
│   └── node/
│       ├── smoke.test.js                  # Basic load test
│       ├── pipeline.test.js               # Pipeline API tests
│       ├── validation.test.js             # Validation tests
│       └── events.test.js                 # Event system tests
├── examples/
│   └── node/
│       ├── simple-pipeline.js             # Basic example
│       ├── dynamic-properties.js          # Runtime property changes
│       └── event-handling.js              # Event listener example
└── docs/declarative-pipeline/
    ├── schemas/
    │   └── pipeline.schema.json           # JSON Schema
    ├── examples/
    │   └── json/                          # JSON pipeline examples
    │       ├── 01_simple_source_sink.json
    │       ├── 02_three_module_chain.json
    │       └── ...
    ├── JS_INTERFACE_PLAN.md               # THIS FILE
    └── JS_API_GUIDE.md                    # API documentation
```

### 8.2 Modified Files

| File | Changes |
|------|---------|
| `base/vcpkg.json` | Add `node-addon-api`, `napi-thread-safe-callback` |
| `base/CMakeLists.txt` | Add bindings/node subdirectory |
| `base/tools/aprapipes_cli.cpp` | Add JSON support, convert command |

---

## 9. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **npm package name** | `@apralabs/aprapipes` | Scoped under ApraPipes organization |
| **TOML support** | **REMOVE completely** | Clean slate, JSON only, no legacy baggage |
| **Frame data access** | No direct access (Phase 1) | Simpler, safer; can add later if needed |
| **Implementation order** | JSON parser → Node addon | Logical progression, test JSON independently |

### Future Considerations

1. **Prebuilt binaries**: Consider publishing prebuilt .node files for common platforms
2. **WebAssembly**: Future consideration for browser support
3. **Frame data access**: May add read-only Buffer access in Phase 2 if needed for ML

---

## 10. Success Criteria

1. ✅ JSON pipelines parse and run identically to TOML
2. ✅ All existing TOML examples have JSON equivalents
3. ✅ Node.js can create, validate, and run pipelines
4. ✅ Events flow from C++ modules to JS callbacks
5. ✅ Dynamic properties can be changed at runtime from JS
6. ✅ TypeScript definitions provide full IntelliSense
7. ✅ Integration tests pass on Linux, macOS, Windows

---

## Appendix A: Dependencies

### npm dependencies (package.json)

```json
{
  "name": "@apralabs/aprapipes",
  "version": "1.0.0",
  "description": "ApraPipes - Video processing pipeline framework",
  "main": "build/Release/aprapipes.node",
  "types": "types/aprapipes.d.ts",
  "scripts": {
    "install": "cmake-js compile",
    "build": "cmake-js build",
    "rebuild": "cmake-js rebuild",
    "test": "jest"
  },
  "dependencies": {
    "bindings": "^1.5.0",
    "node-addon-api": "^7.1.0",
    "napi-thread-safe-callback": "^0.0.6"
  },
  "devDependencies": {
    "cmake-js": "^7.3.0",
    "jest": "^29.0.0",
    "@types/node": "^20.0.0",
    "typescript": "^5.0.0"
  },
  "binary": {
    "napi_versions": [7]
  }
}
```

### vcpkg dependencies (addition)

```json
{
  "dependencies": [
    "node-addon-api"
  ]
}
```
