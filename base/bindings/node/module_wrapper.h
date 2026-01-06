// ============================================================
// File: bindings/node/module_wrapper.h
// ModuleWrapper - N-API wrapper for individual modules
// Phase 3: Core JS API
// ============================================================

#pragma once

#include <napi.h>
#include <string>
#include <boost/shared_ptr.hpp>
#include "Module.h"

namespace aprapipes_node {

// ============================================================
// ModuleWrapper - Wraps a C++ Module for JavaScript access
// ============================================================
class ModuleWrapper : public Napi::ObjectWrap<ModuleWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Function GetClass(Napi::Env env);

    // Create a wrapper for an existing module
    static Napi::Object Create(Napi::Env env,
                               boost::shared_ptr<Module> module,
                               const std::string& instanceId,
                               const std::string& moduleType);

    ModuleWrapper(const Napi::CallbackInfo& info);
    ~ModuleWrapper() = default;

    // Set the wrapped module (called after construction)
    void SetModule(boost::shared_ptr<Module> module,
                   const std::string& instanceId,
                   const std::string& moduleType);

private:
    // Property access
    Napi::Value GetId(const Napi::CallbackInfo& info);
    Napi::Value GetType(const Napi::CallbackInfo& info);
    Napi::Value GetName(const Napi::CallbackInfo& info);

    // Props access (returns ModuleProps values)
    Napi::Value GetProps(const Napi::CallbackInfo& info);

    // State
    Napi::Value IsRunning(const Napi::CallbackInfo& info);
    Napi::Value IsFull(const Napi::CallbackInfo& info);

    // Internal state
    boost::shared_ptr<Module> module_;
    std::string instanceId_;
    std::string moduleType_;
};

} // namespace aprapipes_node
