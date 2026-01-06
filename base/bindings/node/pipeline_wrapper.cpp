// ============================================================
// File: bindings/node/pipeline_wrapper.cpp
// PipelineWrapper implementation
// Phase 3: Core JS API
// ============================================================

#include "pipeline_wrapper.h"
#include "declarative/JsonParser.h"
#include "declarative/PipelineValidator.h"
#include "declarative/ModuleFactory.h"
#include "declarative/ModuleRegistrations.h"

namespace aprapipes_node {

// ============================================================
// AsyncWorker for Init operation
// ============================================================
class InitWorker : public Napi::AsyncWorker {
public:
    InitWorker(Napi::Env env, PipeLine* pipeline, Napi::Promise::Deferred deferred)
        : Napi::AsyncWorker(env), pipeline_(pipeline), deferred_(deferred) {}

    void Execute() override {
        try {
            success_ = pipeline_->init();
            if (!success_) {
                SetError("Pipeline initialization failed");
            }
        } catch (const std::exception& e) {
            SetError(std::string("Init error: ") + e.what());
        }
    }

    void OnOK() override {
        deferred_.Resolve(Napi::Boolean::New(Env(), true));
    }

    void OnError(const Napi::Error& error) override {
        deferred_.Reject(error.Value());
    }

private:
    PipeLine* pipeline_;
    Napi::Promise::Deferred deferred_;
    bool success_ = false;
};

// ============================================================
// AsyncWorker for Run operation
// ============================================================
class RunWorker : public Napi::AsyncWorker {
public:
    RunWorker(Napi::Env env, PipeLine* pipeline, Napi::Promise::Deferred deferred, bool withPause)
        : Napi::AsyncWorker(env), pipeline_(pipeline), deferred_(deferred), withPause_(withPause) {}

    void Execute() override {
        try {
            if (withPause_) {
                pipeline_->run_all_threaded_withpause();
            } else {
                pipeline_->run_all_threaded();
            }
        } catch (const std::exception& e) {
            SetError(std::string("Run error: ") + e.what());
        }
    }

    void OnOK() override {
        deferred_.Resolve(Napi::Boolean::New(Env(), true));
    }

    void OnError(const Napi::Error& error) override {
        deferred_.Reject(error.Value());
    }

private:
    PipeLine* pipeline_;
    Napi::Promise::Deferred deferred_;
    bool withPause_;
};

// ============================================================
// AsyncWorker for Stop operation
// ============================================================
class StopWorker : public Napi::AsyncWorker {
public:
    StopWorker(Napi::Env env, PipeLine* pipeline, Napi::Promise::Deferred deferred)
        : Napi::AsyncWorker(env), pipeline_(pipeline), deferred_(deferred) {}

    void Execute() override {
        try {
            pipeline_->stop();
            pipeline_->wait_for_all(true);
        } catch (const std::exception& e) {
            SetError(std::string("Stop error: ") + e.what());
        }
    }

    void OnOK() override {
        deferred_.Resolve(Napi::Boolean::New(Env(), true));
    }

    void OnError(const Napi::Error& error) override {
        deferred_.Reject(error.Value());
    }

private:
    PipeLine* pipeline_;
    Napi::Promise::Deferred deferred_;
};

// ============================================================
// AsyncWorker for Terminate operation
// ============================================================
class TerminateWorker : public Napi::AsyncWorker {
public:
    TerminateWorker(Napi::Env env, PipeLine* pipeline, Napi::Promise::Deferred deferred)
        : Napi::AsyncWorker(env), pipeline_(pipeline), deferred_(deferred) {}

    void Execute() override {
        try {
            pipeline_->term();
        } catch (const std::exception& e) {
            SetError(std::string("Terminate error: ") + e.what());
        }
    }

    void OnOK() override {
        deferred_.Resolve(Napi::Boolean::New(Env(), true));
    }

    void OnError(const Napi::Error& error) override {
        deferred_.Reject(error.Value());
    }

private:
    PipeLine* pipeline_;
    Napi::Promise::Deferred deferred_;
};

// ============================================================
// PipelineWrapper Implementation
// ============================================================

Napi::Object PipelineWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Pipeline", {
        InstanceMethod("init", &PipelineWrapper::Init),
        InstanceMethod("run", &PipelineWrapper::Run),
        InstanceMethod("stop", &PipelineWrapper::Stop),
        InstanceMethod("terminate", &PipelineWrapper::Terminate),
        InstanceMethod("pause", &PipelineWrapper::Pause),
        InstanceMethod("play", &PipelineWrapper::Play),
        InstanceMethod("step", &PipelineWrapper::Step),
        InstanceMethod("getStatus", &PipelineWrapper::GetStatus),
        InstanceMethod("getName", &PipelineWrapper::GetName),
        InstanceMethod("getModule", &PipelineWrapper::GetModule),
        InstanceMethod("getModuleIds", &PipelineWrapper::GetModuleIds),
    });

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Pipeline", func);
    exports.Set("createPipeline", Napi::Function::New(env, CreatePipeline));

    return exports;
}

Napi::Function PipelineWrapper::GetClass(Napi::Env env) {
    return env.GetInstanceData<Napi::FunctionReference>()->Value();
}

PipelineWrapper::PipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<PipelineWrapper>(info) {
    // Constructor is called internally by CreatePipeline
    // Pipeline is set up via CreatePipeline static method
}

PipelineWrapper::~PipelineWrapper() {
    if (pipeline_) {
        try {
            if (running_) {
                pipeline_->stop();
                pipeline_->wait_for_all(true);
            }
            pipeline_->term();
        } catch (...) {
            // Ignore cleanup errors
        }
    }
}

// ============================================================
// CreatePipeline - Static factory method
// ============================================================
Napi::Value PipelineWrapper::CreatePipeline(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Pipeline configuration required").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Get JSON config
    std::string jsonConfig;
    if (info[0].IsString()) {
        jsonConfig = info[0].As<Napi::String>().Utf8Value();
    } else if (info[0].IsObject()) {
        Napi::Object JSON = env.Global().Get("JSON").As<Napi::Object>();
        Napi::Function stringify = JSON.Get("stringify").As<Napi::Function>();
        jsonConfig = stringify.Call(JSON, {info[0]}).As<Napi::String>().Utf8Value();
    } else {
        Napi::TypeError::New(env, "Config must be string or object").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Ensure modules are registered
    apra::ensureBuiltinModulesRegistered();

    // Parse JSON
    auto parseResult = apra::JsonParser::parseString(jsonConfig);
    if (!parseResult.error.empty()) {
        Napi::Error::New(env, "JSON parse error: " + parseResult.error).ThrowAsJavaScriptException();
        return env.Null();
    }

    // Validate
    apra::PipelineValidator validator;
    auto validationResult = validator.validate(parseResult.description);
    if (validationResult.hasErrors()) {
        std::string errorMsg = "Validation failed:\n";
        for (const auto& issue : validationResult.issues) {
            if (issue.level == apra::Issue::Level::Error) {
                errorMsg += "  [" + issue.code + "] " + issue.message + "\n";
            }
        }
        Napi::Error::New(env, errorMsg).ThrowAsJavaScriptException();
        return env.Null();
    }

    // Build pipeline
    apra::ModuleFactory factory;
    auto buildResult = factory.build(parseResult.description);
    if (!buildResult.success()) {
        std::string errorMsg = "Build failed:\n";
        for (const auto& issue : buildResult.issues) {
            if (issue.level == apra::Issue::Level::Error) {
                errorMsg += "  [" + issue.code + "] " + issue.message + "\n";
            }
        }
        Napi::Error::New(env, errorMsg).ThrowAsJavaScriptException();
        return env.Null();
    }

    // Create wrapper instance
    Napi::Function constructor = GetClass(env);
    Napi::Object wrapper = constructor.New({});
    PipelineWrapper* instance = Napi::ObjectWrap<PipelineWrapper>::Unwrap(wrapper);

    // Transfer ownership
    instance->pipeline_ = std::move(buildResult.pipeline);
    instance->buildIssues_ = std::move(buildResult.issues);

    return wrapper;
}

// ============================================================
// Lifecycle Methods (async)
// ============================================================

Napi::Value PipelineWrapper::Init(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!pipeline_) {
        Napi::Error::New(env, "Pipeline not created").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (initialized_) {
        // Already initialized, resolve immediately
        Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
        deferred.Resolve(Napi::Boolean::New(env, true));
        return deferred.Promise();
    }

    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
    InitWorker* worker = new InitWorker(env, pipeline_.get(), deferred);
    worker->Queue();

    initialized_ = true;
    return deferred.Promise();
}

Napi::Value PipelineWrapper::Run(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!pipeline_) {
        Napi::Error::New(env, "Pipeline not created").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Check for pauseSupport option
    bool withPause = false;
    if (info.Length() > 0 && info[0].IsObject()) {
        Napi::Object options = info[0].As<Napi::Object>();
        if (options.Has("pauseSupport") && options.Get("pauseSupport").IsBoolean()) {
            withPause = options.Get("pauseSupport").As<Napi::Boolean>().Value();
        }
    }

    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
    RunWorker* worker = new RunWorker(env, pipeline_.get(), deferred, withPause);
    worker->Queue();

    running_ = true;
    return deferred.Promise();
}

Napi::Value PipelineWrapper::Stop(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!pipeline_) {
        Napi::Error::New(env, "Pipeline not created").ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
    StopWorker* worker = new StopWorker(env, pipeline_.get(), deferred);
    worker->Queue();

    running_ = false;
    return deferred.Promise();
}

Napi::Value PipelineWrapper::Terminate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!pipeline_) {
        Napi::Error::New(env, "Pipeline not created").ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
    TerminateWorker* worker = new TerminateWorker(env, pipeline_.get(), deferred);
    worker->Queue();

    return deferred.Promise();
}

// ============================================================
// Control Methods (sync)
// ============================================================

Napi::Value PipelineWrapper::Pause(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!pipeline_) {
        Napi::Error::New(env, "Pipeline not created").ThrowAsJavaScriptException();
        return env.Null();
    }

    pipeline_->pause();
    return env.Undefined();
}

Napi::Value PipelineWrapper::Play(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!pipeline_) {
        Napi::Error::New(env, "Pipeline not created").ThrowAsJavaScriptException();
        return env.Null();
    }

    pipeline_->play();
    return env.Undefined();
}

Napi::Value PipelineWrapper::Step(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!pipeline_) {
        Napi::Error::New(env, "Pipeline not created").ThrowAsJavaScriptException();
        return env.Null();
    }

    pipeline_->step();
    return env.Undefined();
}

// ============================================================
// Status Methods
// ============================================================

Napi::Value PipelineWrapper::GetStatus(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!pipeline_) {
        return Napi::String::New(env, "not_created");
    }

    return Napi::String::New(env, pipeline_->getStatus());
}

Napi::Value PipelineWrapper::GetName(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!pipeline_) {
        return Napi::String::New(env, "");
    }

    return Napi::String::New(env, pipeline_->getName());
}

// ============================================================
// Module Access
// ============================================================

Napi::Value PipelineWrapper::GetModule(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // TODO: Implement ModuleWrapper and return it here
    // For now, return null
    return env.Null();
}

Napi::Value PipelineWrapper::GetModuleIds(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Array result = Napi::Array::New(env, moduleContexts_.size());
    size_t i = 0;
    for (const auto& [id, context] : moduleContexts_) {
        result.Set(i++, Napi::String::New(env, id));
    }

    return result;
}

} // namespace aprapipes_node
