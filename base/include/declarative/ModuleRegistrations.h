// ============================================================
// File: declarative/ModuleRegistrations.h
// Task D2: Property Binding System
//
// Declares ensureBuiltinModulesRegistered() which triggers
// registration of all built-in modules on first call.
// ============================================================

#pragma once

namespace apra {

// ============================================================
// Ensures all built-in modules are registered with ModuleRegistry.
//
// This function is thread-safe (uses std::call_once internally).
// It should be called before any module lookup/creation.
//
// Automatically called by:
//   - TomlParser::parse()
//   - ModuleFactory::build()
//
// Can also be called explicitly if needed before those.
// ============================================================
void ensureBuiltinModulesRegistered();

} // namespace apra
