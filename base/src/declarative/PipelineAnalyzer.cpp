// ============================================================
// File: declarative/PipelineAnalyzer.cpp
// Pipeline Compatibility Analyzer implementation
// Sprint 7, Phase 3
// ============================================================

#include "declarative/PipelineAnalyzer.h"
#include <algorithm>
#include <sstream>

namespace apra {

// ============================================================
// Main Analysis Entry Point
// ============================================================

AnalysisResult PipelineAnalyzer::analyze(const PipelineDescription& pipeline) const {
    return analyze(pipeline, ModuleRegistry::instance());
}

AnalysisResult PipelineAnalyzer::analyze(
    const PipelineDescription& pipeline,
    const ModuleRegistry& registry
) const {
    AnalysisResult result;

    // Build connection info from pipeline
    auto connections = buildConnectionInfoList(pipeline, registry);

    // Analyze each connection
    for (const auto& conn : connections) {
        // 1. Check frame type compatibility (errors if incompatible)
        if (!checkFrameTypeCompatibility(conn, result)) {
            continue;  // Skip further checks for this connection
        }

        // Track the effective memory type after any bridges
        FrameMetadata::MemType effectiveMemType = conn.fromMemType;

        // 2. Check memory type compatibility (may add bridge)
        if (conn.fromMemType != conn.toMemType) {
            checkMemoryTypeCompatibility(conn, result);
            // After memory bridge, effective memory type is target
            effectiveMemType = conn.toMemType;
        }

        // 3. Check pixel format compatibility (may add bridge)
        // Note: Format conversion must happen in the target memory space
        checkFormatCompatibility(conn, effectiveMemType, result);
    }

    // 4. Check for suboptimal patterns
    checkForSuboptimalPatterns(pipeline, registry, result);

    // Update summary
    result.memoryBridgeCount = std::count_if(
        result.bridges.begin(), result.bridges.end(),
        [](const BridgeSpec& b) { return b.type == BridgeType::Memory; }
    );
    result.formatBridgeCount = std::count_if(
        result.bridges.begin(), result.bridges.end(),
        [](const BridgeSpec& b) { return b.type == BridgeType::Format; }
    );

    return result;
}

// ============================================================
// Build Connection Info List
// ============================================================

std::vector<ConnectionInfo> PipelineAnalyzer::buildConnectionInfoList(
    const PipelineDescription& pipeline,
    const ModuleRegistry& registry
) const {
    std::vector<ConnectionInfo> connections;

    for (const auto& conn : pipeline.connections) {
        ConnectionInfo info;
        info.fromModuleId = conn.from_module;
        info.toModuleId = conn.to_module;
        info.fromPin = conn.from_pin.empty() ? "output" : conn.from_pin;
        info.toPin = conn.to_pin.empty() ? "input" : conn.to_pin;

        // Get source module info
        const auto* fromModule = pipeline.findModule(conn.from_module);
        if (fromModule) {
            info.fromModuleType = fromModule->module_type;
            const auto* fromModuleInfo = registry.getModule(fromModule->module_type);
            if (fromModuleInfo) {
                const auto* outPin = findOutputPin(fromModuleInfo, info.fromPin);
                if (outPin) {
                    info.fromMemType = outPin->memType;
                    info.fromFrameTypes = outPin->frame_types;
                    info.fromImageTypes = outPin->image_types;
                }
            }
        }

        // Get target module info
        const auto* toModule = pipeline.findModule(conn.to_module);
        if (toModule) {
            info.toModuleType = toModule->module_type;
            const auto* toModuleInfo = registry.getModule(toModule->module_type);
            if (toModuleInfo) {
                const auto* inPin = findInputPin(toModuleInfo, info.toPin);
                if (inPin) {
                    info.toMemType = inPin->memType;
                    info.toFrameTypes = inPin->frame_types;
                    info.toImageTypes = inPin->image_types;
                }
            }
        }

        connections.push_back(std::move(info));
    }

    return connections;
}

// ============================================================
// Frame Type Compatibility Check
// ============================================================

bool PipelineAnalyzer::checkFrameTypeCompatibility(
    const ConnectionInfo& conn,
    AnalysisResult& result
) const {
    // If either side has no frame types specified, assume compatible
    if (conn.fromFrameTypes.empty() || conn.toFrameTypes.empty()) {
        return true;
    }

    // Check if any frame type from source matches any acceptable type at target
    for (const auto& fromType : conn.fromFrameTypes) {
        for (const auto& toType : conn.toFrameTypes) {
            if (fromType == toType) {
                return true;
            }
            // Also check inheritance: e.g., "RawImage" is compatible with "Frame"
            if (toType == "Frame") {
                return true;  // Frame accepts anything
            }
        }
    }

    // No compatible frame types found - this is an error
    result.hasErrors = true;
    AnalysisError error;
    error.code = "E001";
    error.message = "Incompatible frame types between modules";
    error.fromModule = conn.fromModuleId;
    error.toModule = conn.toModuleId;

    std::ostringstream details;
    details << "Output produces: ";
    for (size_t i = 0; i < conn.fromFrameTypes.size(); ++i) {
        if (i > 0) details << ", ";
        details << conn.fromFrameTypes[i];
    }
    details << "; Input accepts: ";
    for (size_t i = 0; i < conn.toFrameTypes.size(); ++i) {
        if (i > 0) details << ", ";
        details << conn.toFrameTypes[i];
    }
    error.details = details.str();

    result.errors.push_back(std::move(error));
    return false;
}

// ============================================================
// Memory Type Compatibility Check
// ============================================================

void PipelineAnalyzer::checkMemoryTypeCompatibility(
    const ConnectionInfo& conn,
    AnalysisResult& result
) const {
    if (conn.fromMemType == conn.toMemType) {
        return;  // Already compatible
    }

    BridgeSpec bridge;
    bridge.fromModule = conn.fromModuleId;
    bridge.toModule = conn.toModuleId;
    bridge.fromPin = conn.fromPin;
    bridge.toPin = conn.toPin;
    bridge.type = BridgeType::Memory;
    bridge.bridgeModule = selectMemoryBridgeModule(conn.fromMemType, conn.toMemType);

    // Set direction
    if (conn.fromMemType == FrameMetadata::HOST &&
        conn.toMemType == FrameMetadata::CUDA_DEVICE) {
        bridge.memoryDirection = MemoryDirection::HostToDevice;
        bridge.props["direction"] = "hostToDevice";
    } else if (conn.fromMemType == FrameMetadata::CUDA_DEVICE &&
               conn.toMemType == FrameMetadata::HOST) {
        bridge.memoryDirection = MemoryDirection::DeviceToHost;
        bridge.props["direction"] = "deviceToHost";
    }

    result.bridges.push_back(std::move(bridge));
}

// ============================================================
// Pixel Format Compatibility Check
// ============================================================

void PipelineAnalyzer::checkFormatCompatibility(
    const ConnectionInfo& conn,
    FrameMetadata::MemType effectiveMemType,
    AnalysisResult& result
) const {
    // If target has no image type restrictions, assume compatible
    if (conn.toImageTypes.empty()) {
        return;
    }

    // If source has no specific format, we can't determine compatibility
    // This would be a runtime check
    if (conn.fromImageTypes.empty()) {
        // Add warning about potential runtime incompatibility
        Warning warning;
        warning.code = "W002";
        warning.message = "Cannot verify pixel format compatibility at build time";
        warning.moduleId = conn.fromModuleId;
        warning.suggestion = "Ensure source produces a format accepted by " + conn.toModuleId;
        result.warnings.push_back(std::move(warning));
        return;
    }

    // Check if any source format is accepted by target
    for (const auto& fromFormat : conn.fromImageTypes) {
        for (const auto& toFormat : conn.toImageTypes) {
            if (fromFormat == toFormat) {
                return;  // Compatible
            }
        }
    }

    // Need format conversion bridge
    // Use first format from each side (in practice, would need more intelligence)
    ImageMetadata::ImageType fromFormat = conn.fromImageTypes[0];
    ImageMetadata::ImageType toFormat = conn.toImageTypes[0];

    BridgeSpec bridge;
    bridge.fromModule = conn.fromModuleId;
    bridge.toModule = conn.toModuleId;
    bridge.fromPin = conn.fromPin;
    bridge.toPin = conn.toPin;
    bridge.type = BridgeType::Format;
    bridge.bridgeModule = selectFormatBridgeModule(effectiveMemType, fromFormat, toFormat);
    bridge.fromFormat = fromFormat;
    bridge.toFormat = toFormat;

    // Set format props
    bridge.props["inputFormat"] = static_cast<int>(fromFormat);
    bridge.props["outputFormat"] = static_cast<int>(toFormat);

    result.bridges.push_back(std::move(bridge));
}

// ============================================================
// Suboptimal Pattern Detection
// ============================================================

void PipelineAnalyzer::checkForSuboptimalPatterns(
    const PipelineDescription& pipeline,
    const ModuleRegistry& registry,
    AnalysisResult& result
) const {
    // Detect CPU modules in a predominantly CUDA pipeline
    // This suggests using GPU alternatives

    // Count CUDA vs CPU modules
    size_t cudaModules = 0;
    size_t cpuModules = 0;
    std::vector<std::string> cpuModuleIds;

    for (const auto& moduleInst : pipeline.modules) {
        const auto* info = registry.getModule(moduleInst.module_type);
        if (!info) continue;

        // Check if module has CUDA tag
        bool isCuda = std::find(info->tags.begin(), info->tags.end(), "cuda") != info->tags.end();

        if (isCuda) {
            ++cudaModules;
        } else {
            ++cpuModules;
            cpuModuleIds.push_back(moduleInst.instance_id);
        }
    }

    // If pipeline is mostly CUDA (>50%), suggest GPU alternatives for CPU modules
    if (cudaModules > 0 && cpuModules > 0 && cudaModules >= cpuModules) {
        for (const auto& cpuId : cpuModuleIds) {
            const auto* moduleInst = pipeline.findModule(cpuId);
            if (!moduleInst) continue;

            const auto* cpuInfo = registry.getModule(moduleInst->module_type);
            if (!cpuInfo) continue;

            // Find function tag (resize, blur, encode, etc.)
            std::string functionTag;
            for (const auto& tag : cpuInfo->tags) {
                if (tag == "resize" || tag == "blur" || tag == "rotate" ||
                    tag == "encoder" || tag == "decoder" || tag == "color") {
                    functionTag = tag;
                    break;
                }
            }

            if (!functionTag.empty()) {
                // Look for CUDA alternative
                auto cudaAlternatives = registry.getModulesWithAllTags({"cuda", functionTag});
                if (!cudaAlternatives.empty()) {
                    Suggestion suggestion;
                    suggestion.currentModule = moduleInst->module_type;
                    suggestion.suggestedModule = cudaAlternatives[0];
                    suggestion.moduleId = cpuId;
                    suggestion.reason = "Using CUDA module would avoid CPU<->GPU memory transfers";
                    result.suggestions.push_back(std::move(suggestion));
                }
            }
        }
    }
}

// ============================================================
// Helper Methods
// ============================================================

const ModuleInfo::PinInfo* PipelineAnalyzer::findOutputPin(
    const ModuleInfo* info,
    const std::string& pinName
) const {
    if (!info) return nullptr;

    for (const auto& pin : info->outputs) {
        if (pin.name == pinName) {
            return &pin;
        }
    }

    // If no match and there's exactly one output, use that
    if (info->outputs.size() == 1) {
        return &info->outputs[0];
    }

    return nullptr;
}

const ModuleInfo::PinInfo* PipelineAnalyzer::findInputPin(
    const ModuleInfo* info,
    const std::string& pinName
) const {
    if (!info) return nullptr;

    for (const auto& pin : info->inputs) {
        if (pin.name == pinName) {
            return &pin;
        }
    }

    // If no match and there's exactly one input, use that
    if (info->inputs.size() == 1) {
        return &info->inputs[0];
    }

    return nullptr;
}

std::string PipelineAnalyzer::selectMemoryBridgeModule(
    FrameMetadata::MemType from,
    FrameMetadata::MemType to
) const {
    // For HOST <-> CUDA_DEVICE, use CudaMemCopy
    if ((from == FrameMetadata::HOST && to == FrameMetadata::CUDA_DEVICE) ||
        (from == FrameMetadata::CUDA_DEVICE && to == FrameMetadata::HOST)) {
        return "CudaMemCopy";
    }

    // For DMABUF conversions, use MemTypeConversion
    if (from == FrameMetadata::DMABUF || to == FrameMetadata::DMABUF) {
        return "MemTypeConversion";
    }

    // For HOST <-> HOST_PINNED, use MemTypeConversion
    if ((from == FrameMetadata::HOST && to == FrameMetadata::HOST_PINNED) ||
        (from == FrameMetadata::HOST_PINNED && to == FrameMetadata::HOST)) {
        return "MemTypeConversion";
    }

    // Default
    return "MemTypeConversion";
}

std::string PipelineAnalyzer::selectFormatBridgeModule(
    FrameMetadata::MemType memType,
    ImageMetadata::ImageType fromFormat,
    ImageMetadata::ImageType toFormat
) const {
    // Use CUDA-accelerated conversion if data is on GPU
    if (memType == FrameMetadata::CUDA_DEVICE) {
        return "CCNPPI";  // CUDA/NPP color conversion
    }

    // Use CPU conversion for host memory
    return "ColorConversion";
}

std::string PipelineAnalyzer::generateBridgeId(
    const std::string& fromModule,
    const std::string& toModule,
    BridgeType type
) const {
    std::ostringstream ss;
    ss << "_bridge_";
    if (type == BridgeType::Memory) {
        ss << "mem_";
    } else {
        ss << "fmt_";
    }
    ss << fromModule << "_" << toModule;
    return ss.str();
}

} // namespace apra
