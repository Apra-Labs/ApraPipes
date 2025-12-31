// ============================================================
// File: declarative/FrameTypeRegistrations.h
// Task F1-F4: Frame Type Metadata
//
// Header for frame type registration functions.
// ============================================================

#pragma once

namespace apra {

// Ensure all built-in frame types are registered.
// Called automatically at static init, but can be called explicitly
// to ensure registration in dynamic loading scenarios.
void ensureBuiltinFrameTypesRegistered();

} // namespace apra
