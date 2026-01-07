// ============================================================
// File: bindings/node/event_emitter.cpp
// EventEmitter implementation
// Phase 4: Event System
// ============================================================

#include "event_emitter.h"

namespace aprapipes_node {

EventEmitter::EventEmitter(Napi::Env env)
    : listenerData_(std::make_shared<ListenerData>()) {
    ensureTSFN(env);
}

EventEmitter::~EventEmitter() {
    release();
}

void EventEmitter::ensureTSFN(Napi::Env env) {
    if (tsfnCreated_) return;

    // Store reference to listenerData for use in CallJS
    ListenerData* listenerPtr = listenerData_.get();

    // Create thread-safe function with simpler signature
    // Using a lambda that captures listenerPtr
    tsfn_ = Napi::ThreadSafeFunction::New(
        env,
        Napi::Function::New(env, [](const Napi::CallbackInfo&) {}),  // Dummy function
        "AprapipesEventEmitter",
        0,  // Unlimited queue
        1   // Initial thread count
    );

    tsfnCreated_ = true;
}

void EventEmitter::release() {
    if (tsfnCreated_) {
        tsfn_.Release();
        tsfnCreated_ = false;
    }
}

void EventEmitter::on(const std::string& event, Napi::Function callback) {
    std::lock_guard<std::mutex> lock(listenerData_->mutex);

    // Create persistent reference to the callback
    Napi::FunctionReference ref = Napi::Persistent(callback);
    listenerData_->listeners[event].push_back(std::move(ref));
}

void EventEmitter::off(const std::string& event, Napi::Function callback) {
    std::lock_guard<std::mutex> lock(listenerData_->mutex);

    auto it = listenerData_->listeners.find(event);
    if (it == listenerData_->listeners.end()) return;

    auto& callbacks = it->second;
    for (auto cbIt = callbacks.begin(); cbIt != callbacks.end(); ++cbIt) {
        if (cbIt->Value() == callback) {
            callbacks.erase(cbIt);
            break;
        }
    }

    // Remove empty event entry
    if (callbacks.empty()) {
        listenerData_->listeners.erase(it);
    }
}

void EventEmitter::removeAllListeners(const std::string& event) {
    std::lock_guard<std::mutex> lock(listenerData_->mutex);

    if (event.empty()) {
        listenerData_->listeners.clear();
    } else {
        listenerData_->listeners.erase(event);
    }
}

bool EventEmitter::hasListeners(const std::string& event) const {
    std::lock_guard<std::mutex> lock(listenerData_->mutex);

    auto it = listenerData_->listeners.find(event);
    return it != listenerData_->listeners.end() && !it->second.empty();
}

void EventEmitter::emitError(const APErrorObject& error) {
    if (!tsfnCreated_) return;

    auto* data = new EventData();
    data->type = EventData::Type::Error;
    data->errorCode = error.getErrorCode();
    data->errorMessage = error.getErrorMessage();
    data->moduleName = error.getModuleName();
    data->moduleId = error.getModuleId();
    data->timestamp = error.getTimestamp();
    data->listenerData = listenerData_;

    tsfn_.NonBlockingCall(data, [](Napi::Env env, Napi::Function, EventData* data) {
        if (!data || !data->listenerData) {
            delete data;
            return;
        }

        std::string eventName = "error";
        Napi::Object eventObj = Napi::Object::New(env);
        eventObj.Set("errorCode", Napi::Number::New(env, data->errorCode));
        eventObj.Set("errorMessage", Napi::String::New(env, data->errorMessage));
        eventObj.Set("moduleName", Napi::String::New(env, data->moduleName));
        eventObj.Set("moduleId", Napi::String::New(env, data->moduleId));
        eventObj.Set("timestamp", Napi::String::New(env, data->timestamp));

        // Call all registered listeners for this event
        std::lock_guard<std::mutex> lock(data->listenerData->mutex);
        auto it = data->listenerData->listeners.find(eventName);
        if (it != data->listenerData->listeners.end()) {
            for (auto& callback : it->second) {
                if (!callback.IsEmpty()) {
                    try {
                        callback.Call({eventObj});
                    } catch (...) {
                        // Ignore callback errors
                    }
                }
            }
        }

        delete data;
    });
}

void EventEmitter::emitHealth(const APHealthObject& health) {
    if (!tsfnCreated_) return;

    auto* data = new EventData();
    data->type = EventData::Type::Health;
    data->moduleId = health.getModuleId();
    data->timestamp = health.getTimestamp();
    data->listenerData = listenerData_;

    tsfn_.NonBlockingCall(data, [](Napi::Env env, Napi::Function, EventData* data) {
        if (!data || !data->listenerData) {
            delete data;
            return;
        }

        std::string eventName = "health";
        Napi::Object eventObj = Napi::Object::New(env);
        eventObj.Set("moduleId", Napi::String::New(env, data->moduleId));
        eventObj.Set("timestamp", Napi::String::New(env, data->timestamp));

        // Call all registered listeners for this event
        std::lock_guard<std::mutex> lock(data->listenerData->mutex);
        auto it = data->listenerData->listeners.find(eventName);
        if (it != data->listenerData->listeners.end()) {
            for (auto& callback : it->second) {
                if (!callback.IsEmpty()) {
                    try {
                        callback.Call({eventObj});
                    } catch (...) {
                        // Ignore callback errors
                    }
                }
            }
        }

        delete data;
    });
}

void EventEmitter::emitLifecycle(const std::string& event) {
    if (!tsfnCreated_) return;

    auto* data = new EventData();
    data->type = EventData::Type::Lifecycle;
    data->lifecycleEvent = event;
    data->listenerData = listenerData_;

    tsfn_.NonBlockingCall(data, [](Napi::Env env, Napi::Function, EventData* data) {
        if (!data || !data->listenerData) {
            delete data;
            return;
        }

        std::string eventName = data->lifecycleEvent;
        Napi::Object eventObj = Napi::Object::New(env);
        eventObj.Set("event", Napi::String::New(env, data->lifecycleEvent));
        eventObj.Set("timestamp", Napi::String::New(env, data->timestamp));

        // Call all registered listeners for this event
        std::lock_guard<std::mutex> lock(data->listenerData->mutex);
        auto it = data->listenerData->listeners.find(eventName);
        if (it != data->listenerData->listeners.end()) {
            for (auto& callback : it->second) {
                if (!callback.IsEmpty()) {
                    try {
                        callback.Call({eventObj});
                    } catch (...) {
                        // Ignore callback errors
                    }
                }
            }
        }

        delete data;
    });
}

} // namespace aprapipes_node
