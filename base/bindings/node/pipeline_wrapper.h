// ============================================================
// File: bindings/node/pipeline_wrapper.h
// PipelineWrapper - N-API wrapper for PipeLine
// Phase 3: Core JS API
// ============================================================

#pragma once

#include <napi.h>
#include <memory>
#include <string>
#include <map>
#include "PipeLine.h"
#include "declarative/ModuleFactory.h"
#include "module_wrapper.h"

namespace aprapipes_node {

// ============================================================
// PipelineWrapper - Wraps C++ PipeLine for JavaScript
// ============================================================
class PipelineWrapper : public Napi::ObjectWrap<PipelineWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Function GetClass(Napi::Env env);

    // Constructor - takes BuildResult from factory
    PipelineWrapper(const Napi::CallbackInfo& info);
    ~PipelineWrapper();

    // Static factory method - creates pipeline from JSON config
    static Napi::Value CreatePipeline(const Napi::CallbackInfo& info);

private:
    // Lifecycle methods
    Napi::Value Init(const Napi::CallbackInfo& info);      // async
    Napi::Value Run(const Napi::CallbackInfo& info);       // async
    Napi::Value Stop(const Napi::CallbackInfo& info);      // async
    Napi::Value Terminate(const Napi::CallbackInfo& info); // async

    // Control methods (sync)
    Napi::Value Pause(const Napi::CallbackInfo& info);
    Napi::Value Play(const Napi::CallbackInfo& info);
    Napi::Value Step(const Napi::CallbackInfo& info);

    // Status methods
    Napi::Value GetStatus(const Napi::CallbackInfo& info);
    Napi::Value GetName(const Napi::CallbackInfo& info);

    // Module access
    Napi::Value GetModule(const Napi::CallbackInfo& info);
    Napi::Value GetModuleIds(const Napi::CallbackInfo& info);

    // Internal state
    std::unique_ptr<PipeLine> pipeline_;
    std::vector<apra::BuildIssue> buildIssues_;
    bool initialized_ = false;
    bool running_ = false;

    // Store module info for getModule() / getModuleIds()
    struct ModuleInfo {
        std::string instanceId;
        std::string moduleType;
        boost::shared_ptr<Module> module;
    };
    std::map<std::string, ModuleInfo> moduleInfoMap_;

    // Store reference to prevent GC
    Napi::ObjectReference selfRef_;
};

} // namespace aprapipes_node
