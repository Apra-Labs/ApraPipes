// ============================================================
// File: bindings/node/addon.cpp
// Node.js native addon for ApraPipes
// Package: @apralabs/aprapipes
// ============================================================

#include <napi.h>
#include <string>
#include <memory>

// ApraPipes headers
#include "declarative/JsonParser.h"
#include "declarative/PipelineValidator.h"
#include "declarative/ModuleFactory.h"
#include "declarative/ModuleRegistry.h"
#include "declarative/ModuleRegistrations.h"

// Phase 3: Pipeline and Module wrappers
#include "pipeline_wrapper.h"
#include "module_wrapper.h"

namespace aprapipes_node {

// ============================================================
// listModules() - Get list of registered modules
// Returns: Array of module names
// ============================================================
Napi::Value ListModules(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // Ensure modules are registered
    apra::ensureBuiltinModulesRegistered();

    auto& registry = apra::ModuleRegistry::instance();
    auto modules = registry.getAllModules();

    Napi::Array result = Napi::Array::New(env, modules.size());
    for (size_t i = 0; i < modules.size(); i++) {
        result.Set(i, Napi::String::New(env, modules[i]));
    }

    return result;
}

// ============================================================
// describeModule(moduleName) - Get detailed module information
// Returns: Object with module metadata
// ============================================================
Napi::Value DescribeModule(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Module name (string) required").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string moduleName = info[0].As<Napi::String>().Utf8Value();

    // Ensure modules are registered
    apra::ensureBuiltinModulesRegistered();

    auto& registry = apra::ModuleRegistry::instance();
    const apra::ModuleInfo* moduleInfo = registry.getModule(moduleName);

    if (!moduleInfo) {
        Napi::Error::New(env, "Module not found: " + moduleName).ThrowAsJavaScriptException();
        return env.Null();
    }

    // Build result object
    Napi::Object result = Napi::Object::New(env);
    result.Set("name", Napi::String::New(env, moduleInfo->name));
    result.Set("description", Napi::String::New(env, moduleInfo->description));
    result.Set("version", Napi::String::New(env, moduleInfo->version));

    // Category
    std::string categoryStr;
    switch (moduleInfo->category) {
        case apra::ModuleCategory::Source: categoryStr = "source"; break;
        case apra::ModuleCategory::Sink: categoryStr = "sink"; break;
        case apra::ModuleCategory::Transform: categoryStr = "transform"; break;
        case apra::ModuleCategory::Analytics: categoryStr = "analytics"; break;
        case apra::ModuleCategory::Utility: categoryStr = "utility"; break;
        default: categoryStr = "unknown"; break;
    }
    result.Set("category", Napi::String::New(env, categoryStr));

    // Tags
    Napi::Array tags = Napi::Array::New(env, moduleInfo->tags.size());
    for (size_t i = 0; i < moduleInfo->tags.size(); i++) {
        tags.Set(i, Napi::String::New(env, moduleInfo->tags[i]));
    }
    result.Set("tags", tags);

    // Properties
    Napi::Array props = Napi::Array::New(env, moduleInfo->properties.size());
    for (size_t i = 0; i < moduleInfo->properties.size(); i++) {
        const auto& prop = moduleInfo->properties[i];
        Napi::Object propObj = Napi::Object::New(env);
        propObj.Set("name", Napi::String::New(env, prop.name));
        propObj.Set("type", Napi::String::New(env, prop.type));
        propObj.Set("required", Napi::Boolean::New(env, prop.required));
        propObj.Set("description", Napi::String::New(env, prop.description));
        props.Set(i, propObj);
    }
    result.Set("properties", props);

    // Inputs
    Napi::Array inputs = Napi::Array::New(env, moduleInfo->inputs.size());
    for (size_t i = 0; i < moduleInfo->inputs.size(); i++) {
        const auto& pin = moduleInfo->inputs[i];
        Napi::Object pinObj = Napi::Object::New(env);
        pinObj.Set("name", Napi::String::New(env, pin.name));
        pinObj.Set("required", Napi::Boolean::New(env, pin.required));

        Napi::Array frameTypes = Napi::Array::New(env, pin.frame_types.size());
        for (size_t j = 0; j < pin.frame_types.size(); j++) {
            frameTypes.Set(j, Napi::String::New(env, pin.frame_types[j]));
        }
        pinObj.Set("frameTypes", frameTypes);
        inputs.Set(i, pinObj);
    }
    result.Set("inputs", inputs);

    // Outputs
    Napi::Array outputs = Napi::Array::New(env, moduleInfo->outputs.size());
    for (size_t i = 0; i < moduleInfo->outputs.size(); i++) {
        const auto& pin = moduleInfo->outputs[i];
        Napi::Object pinObj = Napi::Object::New(env);
        pinObj.Set("name", Napi::String::New(env, pin.name));

        Napi::Array frameTypes = Napi::Array::New(env, pin.frame_types.size());
        for (size_t j = 0; j < pin.frame_types.size(); j++) {
            frameTypes.Set(j, Napi::String::New(env, pin.frame_types[j]));
        }
        pinObj.Set("frameTypes", frameTypes);
        outputs.Set(i, pinObj);
    }
    result.Set("outputs", outputs);

    return result;
}

// ============================================================
// validatePipeline(jsonConfig) - Validate pipeline configuration
// Returns: Object with valid flag and issues array
// ============================================================
Napi::Value ValidatePipeline(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Pipeline configuration required").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string jsonConfig;

    // Accept either string or object
    if (info[0].IsString()) {
        jsonConfig = info[0].As<Napi::String>().Utf8Value();
    } else if (info[0].IsObject()) {
        // Stringify the object
        Napi::Object JSON = env.Global().Get("JSON").As<Napi::Object>();
        Napi::Function stringify = JSON.Get("stringify").As<Napi::Function>();
        jsonConfig = stringify.Call(JSON, {info[0]}).As<Napi::String>().Utf8Value();
    } else {
        Napi::TypeError::New(env, "Pipeline configuration must be a string or object").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Ensure modules are registered
    apra::ensureBuiltinModulesRegistered();

    // Parse JSON
    auto parseResult = apra::JsonParser::parseString(jsonConfig);

    Napi::Object result = Napi::Object::New(env);

    if (!parseResult.error.empty()) {
        result.Set("valid", Napi::Boolean::New(env, false));

        Napi::Array issues = Napi::Array::New(env, 1);
        Napi::Object issue = Napi::Object::New(env);
        issue.Set("level", Napi::String::New(env, "error"));
        issue.Set("code", Napi::String::New(env, "E000"));
        issue.Set("message", Napi::String::New(env, parseResult.error));
        issue.Set("location", Napi::String::New(env, "json"));
        issues.Set(uint32_t(0), issue);
        result.Set("issues", issues);

        return result;
    }

    // Validate the pipeline
    apra::PipelineValidator validator;
    apra::PipelineValidator::Result validationResult = validator.validate(parseResult.description);

    result.Set("valid", Napi::Boolean::New(env, !validationResult.hasErrors()));

    // Convert issues to JS array
    Napi::Array issues = Napi::Array::New(env, validationResult.issues.size());
    for (size_t i = 0; i < validationResult.issues.size(); i++) {
        const auto& issue = validationResult.issues[i];
        Napi::Object issueObj = Napi::Object::New(env);

        std::string levelStr;
        switch (issue.level) {
            case apra::Issue::Level::Error: levelStr = "error"; break;
            case apra::Issue::Level::Warning: levelStr = "warning"; break;
            case apra::Issue::Level::Info: levelStr = "info"; break;
        }

        issueObj.Set("level", Napi::String::New(env, levelStr));
        issueObj.Set("code", Napi::String::New(env, issue.code));
        issueObj.Set("message", Napi::String::New(env, issue.message));
        issueObj.Set("location", Napi::String::New(env, issue.location));

        if (!issue.suggestion.empty()) {
            issueObj.Set("suggestion", Napi::String::New(env, issue.suggestion));
        }

        issues.Set(i, issueObj);
    }
    result.Set("issues", issues);

    return result;
}

// ============================================================
// getVersion() - Get addon version
// Returns: String version number
// ============================================================
Napi::Value GetVersion(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return Napi::String::New(env, "0.1.0");
}

// ============================================================
// Module initialization
// ============================================================
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Basic functions
    exports.Set("getVersion", Napi::Function::New(env, GetVersion));
    exports.Set("listModules", Napi::Function::New(env, ListModules));
    exports.Set("describeModule", Napi::Function::New(env, DescribeModule));
    exports.Set("validatePipeline", Napi::Function::New(env, ValidatePipeline));

    // Phase 3: Pipeline and Module wrapper classes
    ModuleWrapper::Init(env, exports);
    PipelineWrapper::Init(env, exports);

    return exports;
}

NODE_API_MODULE(aprapipes, Init)

} // namespace aprapipes_node
