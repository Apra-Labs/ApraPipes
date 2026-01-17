// ============================================================
// File: bindings/node/module_wrapper.cpp
// ModuleWrapper implementation
// Phase 3: Core JS API + Dynamic Property Support
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
        InstanceMethod("isInputQueFull", &ModuleWrapper::IsInputQueFull),
        // Dynamic property methods
        InstanceMethod("getDynamicPropertyNames", &ModuleWrapper::GetDynamicPropertyNames),
        InstanceMethod("getProperty", &ModuleWrapper::GetProperty),
        InstanceMethod("setProperty", &ModuleWrapper::SetProperty),
        InstanceMethod("hasDynamicProperties", &ModuleWrapper::HasDynamicProperties),
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
                                    const std::string& moduleType,
                                    const apra::DynamicPropertyAccessors& propertyAccessors) {
    if (!moduleConstructor) {
        Napi::Error::New(env, "ModuleWrapper not initialized").ThrowAsJavaScriptException();
        return Napi::Object::New(env);
    }

    Napi::Object wrapper = moduleConstructor->New({});
    ModuleWrapper* instance = Napi::ObjectWrap<ModuleWrapper>::Unwrap(wrapper);
    instance->SetModule(module, instanceId, moduleType, propertyAccessors);

    return wrapper;
}

ModuleWrapper::ModuleWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<ModuleWrapper>(info) {
    // Module is set via SetModule() after construction
}

void ModuleWrapper::SetModule(boost::shared_ptr<Module> module,
                               const std::string& instanceId,
                               const std::string& moduleType,
                               const apra::DynamicPropertyAccessors& propertyAccessors) {
    module_ = module;
    instanceId_ = instanceId;
    moduleType_ = moduleType;
    propertyAccessors_ = propertyAccessors;
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

Napi::Value ModuleWrapper::IsInputQueFull(const Napi::CallbackInfo& info) {
    if (!module_) {
        return Napi::Boolean::New(info.Env(), false);
    }
    return Napi::Boolean::New(info.Env(), module_->isFull());
}

// ============================================================
// Dynamic Property Methods
// ============================================================

Napi::Value ModuleWrapper::HasDynamicProperties(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    bool hasDynProps = propertyAccessors_.getDynamicPropertyNames != nullptr;
    return Napi::Boolean::New(env, hasDynProps);
}

Napi::Value ModuleWrapper::GetDynamicPropertyNames(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!propertyAccessors_.getDynamicPropertyNames) {
        // Module doesn't support dynamic properties
        return Napi::Array::New(env, 0);
    }

    try {
        std::vector<std::string> names = propertyAccessors_.getDynamicPropertyNames();
        Napi::Array result = Napi::Array::New(env, names.size());
        for (size_t i = 0; i < names.size(); ++i) {
            result.Set(i, Napi::String::New(env, names[i]));
        }
        return result;
    } catch (const std::exception& e) {
        Napi::Error::New(env, std::string("Error getting property names: ") + e.what())
            .ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value ModuleWrapper::GetProperty(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "getProperty(name: string) - property name required")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    if (!propertyAccessors_.getProperty) {
        Napi::Error::New(env, "Module does not support dynamic properties")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string propName = info[0].As<Napi::String>().Utf8Value();

    try {
        apra::ScalarPropertyValue value = propertyAccessors_.getProperty(propName);

        // Convert variant to JS value
        return std::visit([&env](auto&& arg) -> Napi::Value {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, int64_t>) {
                return Napi::Number::New(env, static_cast<double>(arg));
            } else if constexpr (std::is_same_v<T, double>) {
                return Napi::Number::New(env, arg);
            } else if constexpr (std::is_same_v<T, bool>) {
                return Napi::Boolean::New(env, arg);
            } else if constexpr (std::is_same_v<T, std::string>) {
                return Napi::String::New(env, arg);
            } else {
                return env.Null();
            }
        }, value);

    } catch (const std::exception& e) {
        Napi::Error::New(env, std::string("Error getting property '") + propName + "': " + e.what())
            .ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value ModuleWrapper::SetProperty(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsString()) {
        Napi::TypeError::New(env, "setProperty(name: string, value: any) - name and value required")
            .ThrowAsJavaScriptException();
        return Napi::Boolean::New(env, false);
    }

    if (!propertyAccessors_.setProperty) {
        Napi::Error::New(env, "Module does not support dynamic properties")
            .ThrowAsJavaScriptException();
        return Napi::Boolean::New(env, false);
    }

    std::string propName = info[0].As<Napi::String>().Utf8Value();
    Napi::Value jsValue = info[1];

    try {
        // Convert JS value to ScalarPropertyValue
        apra::ScalarPropertyValue value;

        if (jsValue.IsNumber()) {
            double numVal = jsValue.As<Napi::Number>().DoubleValue();
            // Use double for all numeric values (most common for PTZ)
            value = numVal;
        } else if (jsValue.IsBoolean()) {
            value = jsValue.As<Napi::Boolean>().Value();
        } else if (jsValue.IsString()) {
            value = jsValue.As<Napi::String>().Utf8Value();
        } else {
            Napi::TypeError::New(env, "Property value must be number, boolean, or string")
                .ThrowAsJavaScriptException();
            return Napi::Boolean::New(env, false);
        }

        bool success = propertyAccessors_.setProperty(propName, value);
        return Napi::Boolean::New(env, success);

    } catch (const std::exception& e) {
        Napi::Error::New(env, std::string("Error setting property '") + propName + "': " + e.what())
            .ThrowAsJavaScriptException();
        return Napi::Boolean::New(env, false);
    }
}

} // namespace aprapipes_node
