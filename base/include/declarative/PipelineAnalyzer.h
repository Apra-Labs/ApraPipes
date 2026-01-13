// ============================================================
// File: declarative/PipelineAnalyzer.h
// Pipeline Compatibility Analyzer for auto-bridging
// Sprint 7, Phase 3
// ============================================================

#pragma once

#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
#include "declarative/PipelineDescription.h"
#include "declarative/ModuleRegistry.h"
#include "FrameMetadata.h"
#include "ImageMetadata.h"

namespace apra {

// ============================================================
// Bridge Types
// ============================================================
enum class BridgeType {
    Memory,   // HOST <-> CUDA_DEVICE (CudaMemCopy)
    Format    // Pixel format conversion (ColorConversion/CCNPPI)
};

// ============================================================
// Memory Transfer Direction
// ============================================================
enum class MemoryDirection {
    HostToDevice,   // HOST -> CUDA_DEVICE
    DeviceToHost    // CUDA_DEVICE -> HOST
};

// ============================================================
// BridgeSpec - Describes a bridge module to be inserted
// ============================================================
struct BridgeSpec {
    std::string fromModule;     // Source module ID
    std::string toModule;       // Target module ID
    std::string fromPin;        // Source output pin name
    std::string toPin;          // Target input pin name
    BridgeType type;            // MEMORY or FORMAT
    std::string bridgeModule;   // Module type (e.g., "CudaMemCopy", "ColorConversion")
    nlohmann::json props;       // Properties for bridge module

    // For MEMORY bridges
    MemoryDirection memoryDirection = MemoryDirection::HostToDevice;

    // For FORMAT bridges
    ImageMetadata::ImageType fromFormat = ImageMetadata::UNSET;
    ImageMetadata::ImageType toFormat = ImageMetadata::UNSET;
};

// ============================================================
// Warning - Suboptimal pattern detected
// ============================================================
struct Warning {
    std::string code;           // Warning code (e.g., "W001")
    std::string message;        // Human-readable message
    std::string moduleId;       // Module causing the warning
    std::string suggestion;     // Suggested fix
};

// ============================================================
// Suggestion - Alternative module recommendation
// ============================================================
struct Suggestion {
    std::string currentModule;  // Currently used module type
    std::string suggestedModule; // Recommended replacement
    std::string reason;         // Why it's better
    std::string moduleId;       // Instance ID in pipeline
};

// ============================================================
// Error - Incompatibility that cannot be bridged
// ============================================================
struct AnalysisError {
    std::string code;           // Error code (e.g., "E001")
    std::string message;        // Human-readable message
    std::string fromModule;     // Source module
    std::string toModule;       // Target module
    std::string details;        // Additional details
};

// ============================================================
// AnalysisResult - Complete analysis output
// ============================================================
struct AnalysisResult {
    bool hasErrors = false;                     // True if there are incompatibilities
    std::vector<AnalysisError> errors;          // Frame type incompatibilities
    std::vector<BridgeSpec> bridges;            // Required bridges in insertion order
    std::vector<Suggestion> suggestions;        // Module replacement recommendations
    std::vector<Warning> warnings;              // Suboptimal pattern warnings

    // Summary statistics
    size_t memoryBridgeCount = 0;
    size_t formatBridgeCount = 0;
};

// ============================================================
// ConnectionInfo - Information about a connection for analysis
// ============================================================
struct ConnectionInfo {
    std::string fromModuleId;
    std::string toModuleId;
    std::string fromPin;
    std::string toPin;

    // Source module info
    std::string fromModuleType;
    FrameMetadata::MemType fromMemType = FrameMetadata::HOST;
    std::vector<std::string> fromFrameTypes;
    std::vector<ImageMetadata::ImageType> fromImageTypes;

    // Target module info
    std::string toModuleType;
    FrameMetadata::MemType toMemType = FrameMetadata::HOST;
    std::vector<std::string> toFrameTypes;
    std::vector<ImageMetadata::ImageType> toImageTypes;
};

// ============================================================
// PipelineAnalyzer - Analyzes pipeline for compatibility issues
// ============================================================
class PipelineAnalyzer {
public:
    PipelineAnalyzer() = default;

    // Analyze a pipeline description and return results
    AnalysisResult analyze(const PipelineDescription& pipeline) const;

    // Analyze with explicit module registry (for testing)
    AnalysisResult analyze(
        const PipelineDescription& pipeline,
        const ModuleRegistry& registry
    ) const;

private:
    // Build connection info list from pipeline description
    std::vector<ConnectionInfo> buildConnectionInfoList(
        const PipelineDescription& pipeline,
        const ModuleRegistry& registry
    ) const;

    // Check frame type compatibility
    bool checkFrameTypeCompatibility(
        const ConnectionInfo& conn,
        AnalysisResult& result
    ) const;

    // Check memory type compatibility and add bridge if needed
    void checkMemoryTypeCompatibility(
        const ConnectionInfo& conn,
        AnalysisResult& result
    ) const;

    // Check pixel format compatibility and add bridge if needed
    void checkFormatCompatibility(
        const ConnectionInfo& conn,
        FrameMetadata::MemType effectiveMemType,  // After any memory bridge
        AnalysisResult& result
    ) const;

    // Check for suboptimal patterns (CPU module in CUDA chain)
    void checkForSuboptimalPatterns(
        const PipelineDescription& pipeline,
        const ModuleRegistry& registry,
        AnalysisResult& result
    ) const;

    // Find output pin info for a module
    const ModuleInfo::PinInfo* findOutputPin(
        const ModuleInfo* info,
        const std::string& pinName
    ) const;

    // Find input pin info for a module
    const ModuleInfo::PinInfo* findInputPin(
        const ModuleInfo* info,
        const std::string& pinName
    ) const;

    // Determine best bridge module for memory transfer
    std::string selectMemoryBridgeModule(
        FrameMetadata::MemType from,
        FrameMetadata::MemType to
    ) const;

    // Determine best bridge module for format conversion
    std::string selectFormatBridgeModule(
        FrameMetadata::MemType memType,
        ImageMetadata::ImageType fromFormat,
        ImageMetadata::ImageType toFormat
    ) const;

    // Generate unique bridge ID
    std::string generateBridgeId(
        const std::string& fromModule,
        const std::string& toModule,
        BridgeType type
    ) const;
};

} // namespace apra
