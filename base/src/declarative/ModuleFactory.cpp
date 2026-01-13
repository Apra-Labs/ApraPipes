// ============================================================
// File: declarative/ModuleFactory.cpp
// Module Factory implementation
// Task D1: Module Factory
// ============================================================

#include "declarative/ModuleFactory.h"
#include "declarative/ModuleRegistrations.h"
#include "declarative/PipelineAnalyzer.h"
#include "Module.h"
#include "FrameMetadata.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include <sstream>
#include <iostream>
#include <cctype>
#include <set>

#ifdef ENABLE_CUDA
#include "CudaCommon.h"
#endif

namespace apra {

// ============================================================
// CUDA Stream Management (conditional compilation)
// ============================================================

#ifdef ENABLE_CUDA
// Holder for CUDA stream to ensure proper lifetime management
static thread_local cudastream_sp g_cudaStream;

static void* createCudaStream() {
    try {
        g_cudaStream = cudastream_sp(new ApraCudaStream());
        return &g_cudaStream;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create CUDA stream: " << e.what() << std::endl;
        return nullptr;
    }
}
#else
static void* createCudaStream() {
    return nullptr;  // CUDA not enabled
}
#endif

// ============================================================
// Type Conversion Registry - Known conversions between frame types
// ============================================================

struct TypeConversion {
    std::string fromType;
    std::string toType;
    std::string moduleName;
    std::map<std::string, std::string> props;
};

// Static registry of known type conversions
static const std::vector<TypeConversion> KNOWN_CONVERSIONS = {
    // Planar to Packed conversions
    {"RawImagePlanar", "RawImage", "ColorConversion", {{"conversionType", "YUV420PLANAR_TO_RGB"}}},

    // Packed to Planar conversions
    {"RawImage", "RawImagePlanar", "ColorConversion", {{"conversionType", "RGB_TO_YUV420PLANAR"}}},

    // Decode conversions
    {"EncodedImage", "RawImage", "ImageDecoderCV", {}},
    {"H264Data", "RawImagePlanar", "H264Decoder", {}},
    {"H264Frame", "RawImagePlanar", "H264Decoder", {}},

    // Encode conversions
    {"RawImage", "EncodedImage", "ImageEncoderCV", {}},
};

// Find a bridge module that converts from srcType to dstType
static const TypeConversion* findTypeConversion(const std::string& srcType, const std::string& dstType) {
    for (const auto& conv : KNOWN_CONVERSIONS) {
        if (conv.fromType == srcType && conv.toType == dstType) {
            return &conv;
        }
    }
    return nullptr;
}

// Generate a suggested module instance name based on conversion
static std::string generateBridgeModuleName(const std::string& fromModule, const std::string& toModule) {
    std::string name = "convert_" + fromModule + "_to_" + toModule;
    for (char& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            c = '_';
        }
    }
    return name;
}

// Generate TOML snippet for inserting a bridge module
static std::string generateTomlSnippet(
    const std::string& fromModule,
    const std::string& toModule,
    const TypeConversion& conv
) {
    std::string bridgeName = generateBridgeModuleName(fromModule, toModule);
    std::ostringstream oss;

    oss << "\n  Add this module:\n\n";
    oss << "  [modules." << bridgeName << "]\n";
    oss << "  type = \"" << conv.moduleName << "\"\n";

    if (!conv.props.empty()) {
        oss << "    [modules." << bridgeName << ".props]\n";
        for (const auto& [key, value] : conv.props) {
            oss << "    " << key << " = \"" << value << "\"\n";
        }
    }

    oss << "\n  And update connections:\n\n";
    oss << "  [[connections]]\n";
    oss << "  from = \"" << fromModule << "\"\n";
    oss << "  to = \"" << bridgeName << "\"\n";
    oss << "\n";
    oss << "  [[connections]]\n";
    oss << "  from = \"" << bridgeName << "\"\n";
    oss << "  to = \"" << toModule << "\"\n";

    return oss.str();
}

// ============================================================
// Helper: Convert frame type string to FrameMetadata::FrameType
// ============================================================
static FrameMetadata::FrameType stringToFrameType(const std::string& typeStr) {
    // Map string names to FrameType enum
    static const std::map<std::string, FrameMetadata::FrameType> typeMap = {
        {"Frame", FrameMetadata::GENERAL},
        {"General", FrameMetadata::GENERAL},
        {"EncodedImage", FrameMetadata::ENCODED_IMAGE},
        {"RawImage", FrameMetadata::RAW_IMAGE},
        {"RawImagePlanar", FrameMetadata::RAW_IMAGE_PLANAR},
        {"Audio", FrameMetadata::AUDIO},
        {"Array", FrameMetadata::ARRAY},
        {"H264Frame", FrameMetadata::H264_DATA},
        {"H265Frame", FrameMetadata::HEVC_DATA},
        {"BMPImage", FrameMetadata::BMP_IMAGE},
        {"Command", FrameMetadata::COMMAND},
        {"PropsChange", FrameMetadata::PROPS_CHANGE},
    };

    auto it = typeMap.find(typeStr);
    if (it != typeMap.end()) {
        return it->second;
    }
    return FrameMetadata::GENERAL;  // Default to GENERAL for unknown types
}

// ============================================================
// Parse connection endpoint "instance.pin" into (instance, pin)
// ============================================================
std::pair<std::string, std::string> ModuleFactory::parseConnectionEndpoint(
    const std::string& endpoint
) {
    auto dotPos = endpoint.find('.');
    if (dotPos == std::string::npos) {
        // No pin specified, use empty string for default handling
        return {endpoint, ""};
    }
    return {endpoint.substr(0, dotPos), endpoint.substr(dotPos + 1)};
}

// ============================================================
// Set up output pins for a module based on registry info
// Returns map of TOML pin name → internal pin ID
// If selfManagedOutputPins is true and no existing pins, returns empty map
// (module will create its pins later, e.g., in addInputPin)
// ============================================================
std::map<std::string, std::string> ModuleFactory::setupOutputPins(
    Module* module,
    const ModuleInfo& info,
    const ModuleInstance& instance,
    std::vector<BuildIssue>& issues,
    bool selfManagedOutputPins
) {
    std::map<std::string, std::string> pinMap;

    // Check if module already has output pins (set up in constructor)
    // Some modules like TestSignalGenerator create their own metadata with proper dimensions
    // Collect ALL existing pins across all frame types (TEXT=24 is the last enum value as of Dec 2025)
    std::vector<std::string> existingPins;
    for (int ft = FrameMetadata::GENERAL; ft <= FrameMetadata::TEXT; ++ft) {
        auto pins = module->getAllOutputPinsByType(ft);
        for (const auto& pin : pins) {
            existingPins.push_back(pin);
        }
    }

    if (!existingPins.empty()) {
        // Map registry output names to existing pins (in order)
        // This allows multi-output modules with existing pins to work correctly
        size_t pinIndex = 0;
        for (const auto& outputPin : info.outputs) {
            if (pinIndex < existingPins.size()) {
                pinMap[outputPin.name] = existingPins[pinIndex++];
            }
        }
        return pinMap;
    }

    // For self-managed modules with no existing pins, return empty map.
    // The module will create its pins later (e.g., Split creates pins in addInputPin).
    if (selfManagedOutputPins) {
        return pinMap;
    }

    // Check for outputFrameType property override (used by FileReaderModule etc.)
    std::string outputFrameTypeOverride;
    auto propIt = instance.properties.find("outputFrameType");
    if (propIt != instance.properties.end()) {
        if (auto* strVal = std::get_if<std::string>(&propIt->second)) {
            outputFrameTypeOverride = *strVal;
        }
    }

    // For each declared output pin, create appropriate FrameMetadata and add it
    for (const auto& outputPin : info.outputs) {
        // Use outputFrameType property if set, otherwise use first frame type from registry
        std::string frameTypeStr;
        if (!outputFrameTypeOverride.empty()) {
            frameTypeStr = outputFrameTypeOverride;
        } else {
            frameTypeStr = outputPin.frame_types.empty() ? "Frame" : outputPin.frame_types[0];
        }
        FrameMetadata::FrameType frameType = stringToFrameType(frameTypeStr);

        // Create the appropriate metadata subclass based on frame type
        // Some modules require specific metadata types (e.g., ImageDecoderCV requires RawImageMetadata)
        framemetadata_sp metadata;
        switch (frameType) {
            case FrameMetadata::RAW_IMAGE:
                metadata = framemetadata_sp(new RawImageMetadata());
                break;
            case FrameMetadata::RAW_IMAGE_PLANAR:
                metadata = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::HOST));
                break;
            default:
                // For all other types, use generic FrameMetadata
                metadata = framemetadata_sp(new FrameMetadata(frameType));
                break;
        }

        // Add the output pin and capture the generated ID
        std::string generatedPinId = module->addOutputPin(metadata);

        // Store mapping: TOML pin name → internal pin ID
        pinMap[outputPin.name] = generatedPinId;
    }

    return pinMap;
}

// ============================================================
// BuildResult methods
// ============================================================

std::string ModuleFactory::BuildResult::formatIssues() const {
    std::ostringstream oss;
    for (const auto& issue : issues) {
        switch (issue.level) {
            case Issue::Level::Error:
                oss << "[ERROR] ";
                break;
            case Issue::Level::Warning:
                oss << "[WARN]  ";
                break;
            case Issue::Level::Info:
                oss << "[INFO]  ";
                break;
        }
        oss << issue.code << " @ " << issue.location << ": " << issue.message << "\n";
    }
    return oss.str();
}

std::string ModuleFactory::BuildResult::formatPipelineGraph(
    const std::vector<Connection>& connections
) const {
    std::ostringstream oss;
    oss << "\n=== Pipeline Graph ===\n";

    if (modules.empty()) {
        oss << "(empty pipeline)\n";
        return oss.str();
    }

    // List modules with their types
    oss << "\nModules:\n";
    for (const auto& [instanceId, entry] : modules) {
        bool isBridge = instanceId.find("_bridge_") == 0;
        oss << "  " << (isBridge ? "[*] " : "[ ] ") << instanceId
            << " (" << entry.moduleType << ")\n";
    }

    // Show connections
    if (!connections.empty()) {
        oss << "\nConnections:\n";
        for (const auto& conn : connections) {
            oss << "  " << conn.from_module;
            if (!conn.from_pin.empty() && conn.from_pin != "output") {
                oss << "." << conn.from_pin;
            }
            oss << " --> " << conn.to_module;
            if (!conn.to_pin.empty() && conn.to_pin != "input") {
                oss << "." << conn.to_pin;
            }
            if (conn.sieve.has_value() && !conn.sieve.value()) {
                oss << " [sieve=false]";
            }
            oss << "\n";
        }
    }

    // Show auto-inserted bridges (from issues)
    bool hasBridgeInfo = false;
    for (const auto& issue : issues) {
        if (issue.code == "I_BRIDGE_INSERTED") {
            if (!hasBridgeInfo) {
                oss << "\nAuto-inserted bridges:\n";
                hasBridgeInfo = true;
            }
            oss << "  [*] " << issue.location << ": " << issue.message << "\n";
        }
    }

    oss << "\n======================\n";
    return oss.str();
}

// ============================================================
// ModuleFactory constructor
// ============================================================

ModuleFactory::ModuleFactory(Options opts) : options_(std::move(opts)) {}

// ============================================================
// Main build method
// ============================================================

ModuleFactory::BuildResult ModuleFactory::build(const PipelineDescription& desc) {
    // Ensure all modules are registered before building
    ensureBuiltinModulesRegistered();

    BuildResult result;

    // Validate we have something to build
    if (desc.modules.empty()) {
        result.issues.push_back(Issue::error(
            Issue::EMPTY_PIPELINE,
            "pipeline",
            "Pipeline has no modules"
        ));
        return result;
    }

    // Generate pipeline name from settings or use default
    std::string pipelineName = desc.settings.name;
    if (pipelineName.empty()) {
        pipelineName = "declarative_pipeline";
    }

    // Create pipeline
    result.pipeline = std::make_unique<PipeLine>(pipelineName);

    // Map of instance_id -> ModuleContext for connection phase (with pin mappings)
    std::map<std::string, ModuleContext> contextMap;

    auto& registry = ModuleRegistry::instance();

    // Phase 0: Check if any modules require CUDA and create stream if needed
    bool needsCudaStream = false;
    for (const auto& instance : desc.modules) {
        if (registry.moduleRequiresCudaStream(instance.module_type)) {
            needsCudaStream = true;
            break;
        }
    }

    if (needsCudaStream) {
        cudaStreamPtr_ = createCudaStream();
        if (!cudaStreamPtr_) {
#ifdef ENABLE_CUDA
            result.issues.push_back(Issue::error(
                "E_CUDA_STREAM",
                "pipeline",
                "Failed to create CUDA stream for CUDA-enabled modules"
            ));
            return result;
#else
            result.issues.push_back(Issue::error(
                "E_CUDA_NOT_ENABLED",
                "pipeline",
                "Pipeline contains CUDA modules but CUDA is not enabled in this build"
            ));
            return result;
#endif
        }
    }

    // Phase 1: Create all modules and set up their output pins
    for (const auto& instance : desc.modules) {
        auto module = createModule(instance, result.issues);
        if (module) {
            // Create context for this module instance
            ModuleContext ctx;
            ctx.module = module;
            ctx.moduleType = instance.module_type;
            ctx.instanceId = instance.instance_id;

            // Set up output pins based on registry info (required before setNext)
            // This also populates the outputPinMap with TOML name → internal ID
            const ModuleInfo* info = registry.getModule(instance.module_type);
            if (info) {
                // Call setupOutputPins with the selfManagedOutputPins flag.
                // For modules that already created pins (selfManaged=true, has existing pins), it maps registry names.
                // For modules without pins (selfManaged=false), it creates pins from registry metadata.
                // For modules that create pins later (selfManaged=true, no existing pins), outputPinMap stays empty.
                if (!info->outputs.empty()) {
                    ctx.outputPinMap = setupOutputPins(module.get(), *info, instance, result.issues,
                                                        info->selfManagedOutputPins);
                }
                // Populate inputPinMap from registry info (for validation)
                for (const auto& inputPin : info->inputs) {
                    // Store pin name with required flag (using "required:" prefix for tracking)
                    ctx.inputPinMap[inputPin.name] = inputPin.required ? "required" : "optional";
                }
            }

            contextMap[instance.instance_id] = std::move(ctx);
            // Note: appendModule is called AFTER connections are made (see Phase 4)

            if (options_.collect_info_messages) {
                result.issues.push_back(Issue::info(
                    Issue::MODULE_CREATED,
                    "modules." + instance.instance_id,
                    "Created module: " + instance.module_type
                ));
            }
        }
    }

    // If any critical errors during module creation, stop
    if (result.hasErrors()) {
        result.pipeline.reset();
        return result;
    }

    // Phase 1b: Auto-Bridge Analysis and Insertion
    // Analyze the pipeline for compatibility issues and auto-insert bridge modules
    std::vector<Connection> modifiedConnections = desc.connections;
    if (options_.auto_bridge_enabled && !desc.connections.empty()) {
        PipelineAnalyzer analyzer;
        auto analysis = analyzer.analyze(desc, registry);

        // Report any errors from analysis (frame type mismatches)
        for (const auto& error : analysis.errors) {
            result.issues.push_back(Issue::error(
                error.code,
                error.fromModule + " -> " + error.toModule,
                error.message + ": " + error.details
            ));
        }

        // Report warnings
        for (const auto& warning : analysis.warnings) {
            result.issues.push_back(Issue::warning(
                warning.code,
                warning.moduleId,
                warning.message + ". " + warning.suggestion
            ));
        }

        // Report suggestions (as info)
        if (options_.collect_info_messages) {
            for (const auto& suggestion : analysis.suggestions) {
                result.issues.push_back(Issue::info(
                    "S001",
                    suggestion.moduleId,
                    "Consider using '" + suggestion.suggestedModule +
                    "' instead of '" + suggestion.currentModule + "': " + suggestion.reason
                ));
            }
        }

        // Insert bridge modules
        if (!analysis.bridges.empty()) {
            modifiedConnections = insertBridgeModules(
                desc.connections, analysis.bridges, contextMap, result.issues
            );
        }
    }

    // Phase 2: Connect modules using pin name resolution
    if (!modifiedConnections.empty()) {
        connectModules(modifiedConnections, contextMap, result.issues);
    }

    // Phase 3: Validate required inputs are connected
    validateInputConnections(contextMap, result.issues);

    // Phase 4: Add source modules to pipeline (appendModule is recursive and follows connections)
    // Identify source modules: modules that are not destination of any connection
    std::set<std::string> destinationModules;
    for (const auto& conn : modifiedConnections) {
        destinationModules.insert(conn.to_module);
    }
    for (const auto& [instanceId, ctx] : contextMap) {
        if (destinationModules.find(instanceId) == destinationModules.end()) {
            // This is a source module (not a destination of any connection)
            result.pipeline->appendModule(ctx.module);
        }
    }

    // If errors during connection or validation, pipeline is invalid
    if (result.hasErrors()) {
        result.pipeline.reset();
        return result;
    }

    // Phase 5: Populate result.modules for external access (e.g., event callbacks, dynamic props)
    for (const auto& [instanceId, ctx] : contextMap) {
        ModuleEntry entry;
        entry.module = ctx.module;
        entry.moduleType = ctx.moduleType;
        entry.instanceId = ctx.instanceId;

        // Create property accessors if module type supports them
        const ModuleInfo* info = registry.getModule(ctx.moduleType);
        if (info && info->propertyAccessorFactory && ctx.module) {
            entry.propertyAccessors = info->propertyAccessorFactory(ctx.module.get());
        }

        result.modules[instanceId] = entry;
    }

    // In strict mode, treat warnings as errors
    if (options_.strict_mode && result.hasWarnings()) {
        result.pipeline.reset();
        result.issues.push_back(Issue::error(
            "E099",
            "pipeline",
            "Build failed due to warnings in strict mode"
        ));
    }

    // Remove info messages if not requested
    if (!options_.collect_info_messages) {
        auto& issues = result.issues;
        issues.erase(
            std::remove_if(issues.begin(), issues.end(),
                [](const BuildIssue& i) { return i.level == Issue::Level::Info; }),
            issues.end()
        );
    }

    return result;
}

// ============================================================
// Create a single module
// ============================================================

boost::shared_ptr<Module> ModuleFactory::createModule(
    const ModuleInstance& instance,
    std::vector<BuildIssue>& issues
) {
    auto& registry = ModuleRegistry::instance();

    // Check module exists in registry
    if (!registry.hasModule(instance.module_type)) {
        issues.push_back(Issue::error(
            Issue::UNKNOWN_MODULE,
            "modules." + instance.instance_id,
            "Unknown module type: " + instance.module_type
        ));
        return nullptr;
    }

    // Get module info for property validation
    const ModuleInfo* info = registry.getModule(instance.module_type);

    // Convert PipelineDescription properties to ModuleRegistry format
    // PipelineDescription::PropertyValue includes array types, ScalarPropertyValue doesn't
    std::map<std::string, ScalarPropertyValue> convertedProps;

    for (const auto& [propName, propValue] : instance.properties) {
        // Find property info for type conversion
        const ModuleInfo::PropInfo* propInfo = nullptr;
        if (info) {
            for (const auto& pi : info->properties) {
                if (pi.name == propName) {
                    propInfo = &pi;
                    break;
                }
            }
        }

        std::string location = "modules." + instance.instance_id + ".props." + propName;

        // Convert based on variant type
        std::visit([&](auto&& val) {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, int64_t>) {
                convertedProps[propName] = val;
            }
            else if constexpr (std::is_same_v<T, double>) {
                convertedProps[propName] = val;
            }
            else if constexpr (std::is_same_v<T, bool>) {
                convertedProps[propName] = val;
            }
            else if constexpr (std::is_same_v<T, std::string>) {
                convertedProps[propName] = val;
            }
            else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                if (!val.empty()) {
                    convertedProps[propName] = val[0];
                    issues.push_back(Issue::warning(
                        Issue::PROP_TYPE_CONVERSION,
                        location,
                        "Array property converted to scalar (using first element)"
                    ));
                }
            }
            else if constexpr (std::is_same_v<T, std::vector<double>>) {
                if (!val.empty()) {
                    convertedProps[propName] = val[0];
                    issues.push_back(Issue::warning(
                        Issue::PROP_TYPE_CONVERSION,
                        location,
                        "Array property converted to scalar (using first element)"
                    ));
                }
            }
            else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                if (!val.empty()) {
                    convertedProps[propName] = val[0];
                    issues.push_back(Issue::warning(
                        Issue::PROP_TYPE_CONVERSION,
                        location,
                        "Array property converted to scalar (using first element)"
                    ));
                }
            }
        }, propValue);
    }

    // Validate required properties are provided
    if (info) {
        for (const auto& propInfo : info->properties) {
            if (propInfo.required) {
                auto it = convertedProps.find(propInfo.name);
                if (it == convertedProps.end()) {
                    issues.push_back(Issue::error(
                        Issue::MISSING_REQUIRED_PROP,
                        "modules." + instance.instance_id + ".props." + propInfo.name,
                        "Required property '" + propInfo.name + "' not provided for module '" +
                        instance.module_type + "'"
                    ));
                }
            }
        }
    }

    // If we have errors from missing required properties, don't try to create the module
    bool hasRequiredPropErrors = std::any_of(issues.begin(), issues.end(),
        [&instance](const BuildIssue& i) {
            return i.level == Issue::Level::Error &&
                   i.location.find("modules." + instance.instance_id) != std::string::npos;
        });
    if (hasRequiredPropErrors) {
        return nullptr;
    }

    // Create via registry factory (use CUDA factory if module requires CUDA stream)
    try {
        std::unique_ptr<Module> modulePtr;

        if (registry.moduleRequiresCudaStream(instance.module_type)) {
            // Use CUDA factory
            if (!cudaStreamPtr_) {
                issues.push_back(Issue::error(
                    Issue::MODULE_CREATION_FAILED,
                    "modules." + instance.instance_id,
                    "Module " + instance.module_type + " requires CUDA stream but none available"
                ));
                return nullptr;
            }
            modulePtr = registry.createCudaModule(instance.module_type, convertedProps, cudaStreamPtr_);
        } else {
            // Use regular factory
            modulePtr = registry.createModule(instance.module_type, convertedProps);
        }

        if (!modulePtr) {
            issues.push_back(Issue::error(
                Issue::MODULE_CREATION_FAILED,
                "modules." + instance.instance_id,
                "Factory returned null for module: " + instance.module_type
            ));
            return nullptr;
        }

        // Convert unique_ptr to shared_ptr for ApraPipes compatibility
        return boost::shared_ptr<Module>(modulePtr.release());
    }
    catch (const std::exception& e) {
        issues.push_back(Issue::error(
            Issue::MODULE_CREATION_FAILED,
            "modules." + instance.instance_id,
            "Failed to create module: " + std::string(e.what())
        ));
        return nullptr;
    }
    catch (...) {
        issues.push_back(Issue::error(
            Issue::MODULE_CREATION_FAILED,
            "modules." + instance.instance_id,
            "Unknown exception while creating module"
        ));
        return nullptr;
    }
}

// ============================================================
// Apply properties to module (future use - properties are
// currently passed to factory function)
// ============================================================

void ModuleFactory::applyProperties(
    Module* module,
    const ModuleInstance& instance,
    const ModuleInfo* info,
    std::vector<BuildIssue>& issues
) {
    // Properties are currently passed to the factory function during module creation.
    // This method is here for future use if we need to apply dynamic properties
    // after module creation.
}

// ============================================================
// Insert bridge modules for memory/format conversion
// Returns modified connections list with bridges inserted
// ============================================================

std::vector<Connection> ModuleFactory::insertBridgeModules(
    const std::vector<Connection>& originalConnections,
    const std::vector<BridgeSpec>& bridges,
    std::map<std::string, ModuleContext>& contextMap,
    std::vector<BuildIssue>& issues
) {
    // Build a map of original connections to their indices
    // Key: "fromModule.fromPin -> toModule.toPin"
    auto makeConnKey = [](const Connection& conn) {
        return conn.from_module + "." + conn.from_pin + "->" +
               conn.to_module + "." + conn.to_pin;
    };

    // Build map of bridge specs by connection key
    std::map<std::string, std::vector<const BridgeSpec*>> bridgesByConnection;
    for (const auto& bridge : bridges) {
        std::string key = bridge.fromModule + "." + bridge.fromPin + "->" +
                          bridge.toModule + "." + bridge.toPin;
        bridgesByConnection[key].push_back(&bridge);
    }

    auto& registry = ModuleRegistry::instance();
    std::vector<Connection> modifiedConnections;
    int bridgeCounter = 0;

    for (const auto& conn : originalConnections) {
        std::string connKey = makeConnKey(conn);
        auto bridgeIt = bridgesByConnection.find(connKey);

        if (bridgeIt == bridgesByConnection.end() || bridgeIt->second.empty()) {
            // No bridge needed for this connection, keep as-is
            modifiedConnections.push_back(conn);
            continue;
        }

        // Insert bridges for this connection
        const auto& bridgesForConn = bridgeIt->second;
        std::string currentFromModule = conn.from_module;
        std::string currentFromPin = conn.from_pin;

        for (const auto* bridgePtr : bridgesForConn) {
            const auto& bridge = *bridgePtr;
            ++bridgeCounter;

            // Generate unique bridge instance ID
            std::string bridgeInstanceId = "_bridge_" + std::to_string(bridgeCounter) +
                                           "_" + bridge.bridgeModule;

            // Create module instance for bridge
            ModuleInstance bridgeInstance;
            bridgeInstance.instance_id = bridgeInstanceId;
            bridgeInstance.module_type = bridge.bridgeModule;

            // Set properties from bridge spec
            if (bridge.type == BridgeType::Memory) {
                // For CudaMemCopy, set direction
                if (bridge.memoryDirection == MemoryDirection::HostToDevice) {
                    bridgeInstance.properties["direction"] = std::string("hostToDevice");
                } else if (bridge.memoryDirection == MemoryDirection::DeviceToHost) {
                    bridgeInstance.properties["direction"] = std::string("deviceToHost");
                }
            } else if (bridge.type == BridgeType::Format) {
                // For ColorConversion/CCNPPI, set input/output formats
                bridgeInstance.properties["inputFormat"] = static_cast<int64_t>(bridge.fromFormat);
                bridgeInstance.properties["outputFormat"] = static_cast<int64_t>(bridge.toFormat);
            }

            // Create the bridge module
            auto bridgeModule = createModule(bridgeInstance, issues);
            if (!bridgeModule) {
                issues.push_back(Issue::error(
                    "E_BRIDGE_CREATION",
                    bridgeInstanceId,
                    "Failed to create bridge module: " + bridge.bridgeModule
                ));
                continue;
            }

            // Create context for bridge module
            ModuleContext bridgeCtx;
            bridgeCtx.module = bridgeModule;
            bridgeCtx.moduleType = bridge.bridgeModule;
            bridgeCtx.instanceId = bridgeInstanceId;

            // Set up output pins for bridge module
            const ModuleInfo* bridgeInfo = registry.getModule(bridge.bridgeModule);
            if (bridgeInfo && !bridgeInfo->outputs.empty()) {
                bridgeCtx.outputPinMap = setupOutputPins(
                    bridgeModule.get(), *bridgeInfo, bridgeInstance, issues,
                    bridgeInfo->selfManagedOutputPins
                );
            }

            contextMap[bridgeInstanceId] = std::move(bridgeCtx);

            // Report bridge insertion
            if (options_.collect_info_messages) {
                std::string direction;
                if (bridge.type == BridgeType::Memory) {
                    direction = (bridge.memoryDirection == MemoryDirection::HostToDevice)
                                ? "HOST->CUDA" : "CUDA->HOST";
                } else {
                    direction = "Format conversion";
                }
                issues.push_back(Issue::info(
                    "I_BRIDGE_INSERTED",
                    bridgeInstanceId,
                    "Auto-inserted " + bridge.bridgeModule + " (" + direction +
                    ") between " + conn.from_module + " and " + conn.to_module
                ));
            }

            // Create connection from current source to bridge
            Connection connToBridge;
            connToBridge.from_module = currentFromModule;
            connToBridge.from_pin = currentFromPin;
            connToBridge.to_module = bridgeInstanceId;
            connToBridge.to_pin = "input";
            connToBridge.sieve = conn.sieve;
            modifiedConnections.push_back(connToBridge);

            // Update current source for next bridge (or final connection)
            currentFromModule = bridgeInstanceId;
            currentFromPin = "output";
        }

        // Create final connection from last bridge to original destination
        Connection connFromBridge;
        connFromBridge.from_module = currentFromModule;
        connFromBridge.from_pin = currentFromPin;
        connFromBridge.to_module = conn.to_module;
        connFromBridge.to_pin = conn.to_pin;
        connFromBridge.sieve = conn.sieve;
        modifiedConnections.push_back(connFromBridge);
    }

    return modifiedConnections;
}

// ============================================================
// Connect modules according to connections list
// Uses ModuleContext for pin name resolution
// ============================================================

bool ModuleFactory::connectModules(
    const std::vector<Connection>& connections,
    std::map<std::string, ModuleContext>& contextMap,
    std::vector<BuildIssue>& issues
) {
    bool allSuccess = true;

    for (const auto& conn : connections) {
        // Find source module context
        auto srcIt = contextMap.find(conn.from_module);
        if (srcIt == contextMap.end()) {
            issues.push_back(Issue::error(
                Issue::UNKNOWN_SOURCE_MODULE,
                "connections",
                "Unknown source module: " + conn.from_module
            ));
            allSuccess = false;
            continue;
        }

        // Find destination module context
        auto dstIt = contextMap.find(conn.to_module);
        if (dstIt == contextMap.end()) {
            issues.push_back(Issue::error(
                Issue::UNKNOWN_DEST_MODULE,
                "connections",
                "Unknown destination module: " + conn.to_module
            ));
            allSuccess = false;
            continue;
        }

        ModuleContext& srcCtx = srcIt->second;
        ModuleContext& dstCtx = dstIt->second;

        // Resolve source pin name to internal ID
        std::string srcPinId;
        if (!conn.from_pin.empty()) {
            auto pinIt = srcCtx.outputPinMap.find(conn.from_pin);
            if (pinIt != srcCtx.outputPinMap.end()) {
                srcPinId = pinIt->second;
            } else {
                // Pin name not found in map - might be a single-output module
                // or an unknown pin name
                if (srcCtx.outputPinMap.size() == 1) {
                    // Single output module, use that pin
                    srcPinId = srcCtx.outputPinMap.begin()->second;
                } else if (!srcCtx.outputPinMap.empty()) {
                    // Multi-output module with unknown pin name
                    issues.push_back(Issue::error(
                        Issue::UNKNOWN_SOURCE_PIN,
                        "connections",
                        "Unknown output pin '" + conn.from_pin + "' on module '" +
                        conn.from_module + "'. Available pins: " +
                        [&]() {
                            std::string pins;
                            for (const auto& p : srcCtx.outputPinMap) {
                                if (!pins.empty()) pins += ", ";
                                pins += p.first;
                            }
                            return pins;
                        }()
                    ));
                    allSuccess = false;
                    continue;
                }
            }
        } else if (srcCtx.outputPinMap.size() == 1) {
            // No pin specified, single output - use it
            srcPinId = srcCtx.outputPinMap.begin()->second;
        }
        // If srcPinId is empty, we'll use default behavior (all pins)

        // Track which input is connected on destination
        dstCtx.connectedInputs.push_back(conn.to_pin);

        // Static type checking BEFORE attempting connection
        // Get source and destination module info from registry
        auto& registry = ModuleRegistry::instance();
        const ModuleInfo* srcInfo = registry.getModule(srcCtx.moduleType);
        const ModuleInfo* dstInfo = registry.getModule(dstCtx.moduleType);

        if (srcInfo && dstInfo) {
            // Get source output type
            std::string srcOutputType;
            if (!srcInfo->outputs.empty()) {
                // Use specified pin or first output
                for (const auto& outputPin : srcInfo->outputs) {
                    if (conn.from_pin.empty() || outputPin.name == conn.from_pin) {
                        if (!outputPin.frame_types.empty()) {
                            srcOutputType = outputPin.frame_types[0];
                        }
                        break;
                    }
                }
            }

            // Get destination input types
            std::vector<std::string> dstInputTypes;
            if (!dstInfo->inputs.empty()) {
                for (const auto& inputPin : dstInfo->inputs) {
                    if (conn.to_pin.empty() || inputPin.name == conn.to_pin) {
                        dstInputTypes = inputPin.frame_types;
                        break;
                    }
                }
            }

            // Check type compatibility
            if (!srcOutputType.empty() && !dstInputTypes.empty()) {
                bool compatible = false;
                for (const auto& dstType : dstInputTypes) {
                    if (srcOutputType == dstType ||
                        dstType == "*" || srcOutputType == "*" ||
                        dstType == "Frame" || srcOutputType == "Frame") {
                        compatible = true;
                        break;
                    }
                }

                if (!compatible) {
                    // Look for a known type conversion
                    std::string suggestion;
                    for (const auto& dstType : dstInputTypes) {
                        const TypeConversion* conv = findTypeConversion(srcOutputType, dstType);
                        if (conv) {
                            suggestion = "Insert " + conv->moduleName + " module to convert " +
                                         srcOutputType + " → " + dstType + ":" +
                                         generateTomlSnippet(conn.from_module, conn.to_module, *conv);
                            break;
                        }
                    }

                    std::string dstTypesStr;
                    for (size_t i = 0; i < dstInputTypes.size(); ++i) {
                        if (i > 0) dstTypesStr += ", ";
                        dstTypesStr += dstInputTypes[i];
                    }

                    issues.push_back(Issue::error(
                        Issue::FRAME_TYPE_INCOMPATIBLE,
                        "connections",
                        "Frame type mismatch: " + conn.from_module + " outputs '" + srcOutputType +
                            "', but " + conn.to_module + " expects [" + dstTypesStr + "]",
                        suggestion
                    ));
                    allSuccess = false;
                    continue;  // Skip the actual connection attempt
                }
            }
        }

        // Connect using ApraPipes API
        // If sieve is explicitly set in JSON, use that value.
        // Otherwise, use C++ API default (sieve=true in Module::setNext).
        // When sieve=false, input frames pass through along with explicit outputs
        // When sieve=true, only explicit output pins are connected
        try {
            bool connected;
            if (!srcPinId.empty()) {
                // Pin-specific connection
                // Note: Pin-specific connections currently don't support sieve=false passthrough
                // The internal setNext(pinIdArr) always uses sieve=true
                std::vector<std::string> pinIds = {srcPinId};
                connected = srcCtx.module->setNext(dstCtx.module, pinIds);
            } else if (conn.sieve.has_value()) {
                // Sieve explicitly set in JSON - use specified value
                connected = srcCtx.module->setNext(dstCtx.module, true, conn.sieve.value());
            } else {
                // Sieve not specified - use C++ API default (sieve=true)
                connected = srcCtx.module->setNext(dstCtx.module, true);
            }

            if (!connected) {
                issues.push_back(Issue::error(
                    Issue::CONNECTION_FAILED,
                    "connections",
                    "setNext() returned false for: " + conn.from_module +
                    (conn.from_pin.empty() ? "" : "." + conn.from_pin) +
                    " -> " + conn.to_module +
                    (conn.to_pin.empty() ? "" : "." + conn.to_pin)
                ));
                allSuccess = false;
            } else if (options_.collect_info_messages) {
                issues.push_back(Issue::info(
                    Issue::CONNECTION_ESTABLISHED,
                    "connections",
                    "Connected: " + conn.from_module + "." + conn.from_pin +
                    " -> " + conn.to_module + "." + conn.to_pin
                ));
            }
        }
        catch (const std::exception& e) {
            issues.push_back(Issue::error(
                Issue::CONNECTION_FAILED,
                "connections",
                "Connection failed: " + std::string(e.what())
            ));
            allSuccess = false;
        }
        catch (...) {
            issues.push_back(Issue::error(
                Issue::CONNECTION_FAILED,
                "connections",
                "Unknown exception during connection"
            ));
            allSuccess = false;
        }
    }

    return allSuccess;
}

// ============================================================
// Convert PropertyValue from PipelineDescription format
// ============================================================

std::optional<ScalarPropertyValue> ModuleFactory::convertPropertyValue(
    const PropertyValue& value,
    const ModuleInfo::PropInfo& propInfo,
    std::vector<BuildIssue>& issues,
    const std::string& location
) {
    // This is handled inline in createModule for now
    // Keeping this method for potential future use with more complex conversions
    return std::nullopt;
}

// ============================================================
// Validate that all required inputs are connected
// ============================================================

void ModuleFactory::validateInputConnections(
    const std::map<std::string, ModuleContext>& contextMap,
    std::vector<BuildIssue>& issues
) {
    for (const auto& [instanceId, ctx] : contextMap) {
        // Check each declared input pin
        for (const auto& [inputName, requiredFlag] : ctx.inputPinMap) {
            bool isRequired = (requiredFlag == "required");

            if (isRequired) {
                // Check if this input is in the connectedInputs list
                bool isConnected = std::find(
                    ctx.connectedInputs.begin(),
                    ctx.connectedInputs.end(),
                    inputName
                ) != ctx.connectedInputs.end();

                // Also check for empty pin name (default input connection)
                if (!isConnected && ctx.inputPinMap.size() == 1) {
                    // Single-input module - check if any connection targets this module
                    isConnected = !ctx.connectedInputs.empty();
                }

                if (!isConnected) {
                    issues.push_back(Issue::error(
                        Issue::MISSING_REQUIRED_INPUT,
                        "modules." + instanceId,
                        "Required input '" + inputName + "' is not connected on module '" +
                        ctx.moduleType + "'"
                    ));
                }
            }
            // Optional inputs don't generate errors if not connected
        }
    }
}

} // namespace apra
