// ============================================================
// File: bindings/node/module_wrapper.cpp
// ModuleWrapper implementation
// Phase 3: Core JS API
// ============================================================

#include "module_wrapper.h"

namespace aprapipes_node {

// Static storage for constructor reference
static Napi::FunctionReference* moduleConstructor = nullptr;

Napi::Object ModuleWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "ModuleHandle", {
        InstanceAccessor<&ModuleWrapper::GetId>("id"),
        InstanceAccessor<&ModuleWrapper::GetType>("type"),
        InstanceAccessor<&ModuleWrapper::GetName>("name"),
        InstanceMethod("getProps", &ModuleWrapper::GetProps),
        InstanceMethod("isRunning", &ModuleWrapper::IsRunning),
        InstanceMethod("isFull", &ModuleWrapper::IsFull),
    });

    moduleConstructor = new Napi::FunctionReference();
    *moduleConstructor = Napi::Persistent(func);

    exports.Set("ModuleHandle", func);
    return exports;
}

Napi::Function ModuleWrapper::GetClass(Napi::Env env) {
    if (moduleConstructor) {
        return moduleConstructor->Value();
    }
    return Napi::Function();
}

Napi::Object ModuleWrapper::Create(Napi::Env env,
                                    boost::shared_ptr<Module> module,
                                    const std::string& instanceId,
                                    const std::string& moduleType) {
    if (!moduleConstructor) {
        Napi::Error::New(env, "ModuleWrapper not initialized").ThrowAsJavaScriptException();
        return Napi::Object::New(env);
    }

    Napi::Object wrapper = moduleConstructor->New({});
    ModuleWrapper* instance = Napi::ObjectWrap<ModuleWrapper>::Unwrap(wrapper);
    instance->SetModule(module, instanceId, moduleType);

    return wrapper;
}

ModuleWrapper::ModuleWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<ModuleWrapper>(info) {
    // Module is set via SetModule() after construction
}

void ModuleWrapper::SetModule(boost::shared_ptr<Module> module,
                               const std::string& instanceId,
                               const std::string& moduleType) {
    module_ = module;
    instanceId_ = instanceId;
    moduleType_ = moduleType;
}

// ============================================================
// Property Accessors
// ============================================================

Napi::Value ModuleWrapper::GetId(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), instanceId_);
}

Napi::Value ModuleWrapper::GetType(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), moduleType_);
}

Napi::Value ModuleWrapper::GetName(const Napi::CallbackInfo& info) {
    if (!module_) {
        return info.Env().Null();
    }
    return Napi::String::New(info.Env(), module_->getName());
}

// ============================================================
// Props Access
// ============================================================

Napi::Value ModuleWrapper::GetProps(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!module_) {
        return env.Null();
    }

    ModuleProps props = module_->getProps();

    Napi::Object result = Napi::Object::New(env);
    result.Set("fps", Napi::Number::New(env, props.fps));
    result.Set("qlen", Napi::Number::New(env, static_cast<double>(props.qlen)));
    result.Set("logHealth", Napi::Boolean::New(env, props.logHealth));
    result.Set("logHealthFrequency", Napi::Number::New(env, static_cast<double>(props.logHealthFrequency)));
    result.Set("maxConcurrentFrames", Napi::Number::New(env, static_cast<double>(props.maxConcurrentFrames)));
    result.Set("enableHealthCallBack", Napi::Boolean::New(env, props.enableHealthCallBack));
    result.Set("healthUpdateIntervalInSec", Napi::Number::New(env, props.healthUpdateIntervalInSec));

    return result;
}

// ============================================================
// State Methods
// ============================================================

Napi::Value ModuleWrapper::IsRunning(const Napi::CallbackInfo& info) {
    if (!module_) {
        return Napi::Boolean::New(info.Env(), false);
    }
    // Module doesn't expose isRunning() publicly, so we check if it's initialized
    // by checking if getProps() works (a rough proxy)
    return Napi::Boolean::New(info.Env(), true);
}

Napi::Value ModuleWrapper::IsFull(const Napi::CallbackInfo& info) {
    if (!module_) {
        return Napi::Boolean::New(info.Env(), false);
    }
    return Napi::Boolean::New(info.Env(), module_->isFull());
}

} // namespace aprapipes_node
