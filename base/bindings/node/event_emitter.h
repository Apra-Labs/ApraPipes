// ============================================================
// File: bindings/node/event_emitter.h
// EventEmitter - Thread-safe event delivery to JavaScript
// Phase 4: Event System
// ============================================================

#pragma once

#include <napi.h>
#include <string>
#include <map>
#include <vector>
#include <mutex>
#include "APErrorObject.h"
#include "APHealthObject.h"

namespace aprapipes_node {

// ============================================================
// EventEmitter - Manages JS event listeners and thread-safe delivery
// ============================================================
class EventEmitter {
public:
    EventEmitter(Napi::Env env);
    ~EventEmitter();

    // Register/unregister event listeners
    void on(const std::string& event, Napi::Function callback);
    void off(const std::string& event, Napi::Function callback);
    void removeAllListeners(const std::string& event = "");

    // Emit events (thread-safe - can be called from any thread)
    void emitError(const APErrorObject& error);
    void emitHealth(const APHealthObject& health);
    void emitLifecycle(const std::string& event);  // started, stopped, paused, resumed, endOfStream

    // Check if there are listeners for an event
    bool hasListeners(const std::string& event) const;

    // Cleanup (must be called before destruction if TSFN was created)
    void release();

private:
    // Thread-safe function for cross-thread calls
    Napi::ThreadSafeFunction tsfn_;
    bool tsfnCreated_ = false;

    // Event listeners (accessed only on main thread via TSFN)
    struct ListenerData {
        std::map<std::string, std::vector<Napi::FunctionReference>> listeners;
        std::mutex mutex;  // Protects listeners map
    };
    std::shared_ptr<ListenerData> listenerData_;

    // Event data structures for thread-safe passing
    struct EventData {
        enum class Type { Error, Health, Lifecycle };
        Type type;

        // Error data
        int errorCode = 0;
        std::string errorMessage;
        std::string moduleName;
        std::string moduleId;
        std::string timestamp;

        // Lifecycle event name
        std::string lifecycleEvent;

        // Reference to listener data for callback invocation
        std::shared_ptr<ListenerData> listenerData;
    };

    void ensureTSFN(Napi::Env env);
};

} // namespace aprapipes_node
