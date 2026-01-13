// ============================================================
// Pipeline Analyzer Tests
// Sprint 7, Phase 3
// ============================================================

#include <boost/test/unit_test.hpp>
#include "declarative/PipelineAnalyzer.h"
#include "declarative/ModuleRegistry.h"
#include "declarative/PipelineDescription.h"
#include "declarative/Metadata.h"
#include "FrameMetadata.h"
#include "ImageMetadata.h"

using namespace apra;

// ============================================================
// Test Fixtures
// ============================================================

struct AnalyzerFixture {
    AnalyzerFixture() {
        // Clear registry before each test
        ModuleRegistry::instance().clear();
        ModuleRegistry::instance().rerunRegistrations();
    }

    ~AnalyzerFixture() {
        ModuleRegistry::instance().clear();
        ModuleRegistry::instance().rerunRegistrations();
    }

    // Helper to register a mock module with specific pin specs
    void registerMockModule(
        const std::string& name,
        FrameMetadata::MemType inputMemType,
        FrameMetadata::MemType outputMemType,
        const std::string& inputFrameType = "RawImage",
        const std::string& outputFrameType = "RawImage",
        const std::vector<std::string>& tags = {}
    ) {
        ModuleInfo info;
        info.name = name;
        info.category = ModuleCategory::Transform;
        info.version = "1.0.0";
        info.description = "Mock module for testing";
        info.tags = tags;

        // Input pin
        ModuleInfo::PinInfo inputPin;
        inputPin.name = "input";
        inputPin.frame_types.push_back(inputFrameType);
        inputPin.memType = inputMemType;
        inputPin.required = true;
        info.inputs.push_back(inputPin);

        // Output pin
        ModuleInfo::PinInfo outputPin;
        outputPin.name = "output";
        outputPin.frame_types.push_back(outputFrameType);
        outputPin.memType = outputMemType;
        outputPin.required = true;
        info.outputs.push_back(outputPin);

        ModuleRegistry::instance().registerModule(std::move(info));
    }

    // Helper to register module with image types
    void registerMockModuleWithFormats(
        const std::string& name,
        FrameMetadata::MemType memType,
        const std::vector<ImageMetadata::ImageType>& inputFormats,
        const std::vector<ImageMetadata::ImageType>& outputFormats,
        const std::vector<std::string>& tags = {}
    ) {
        ModuleInfo info;
        info.name = name;
        info.category = ModuleCategory::Transform;
        info.version = "1.0.0";
        info.description = "Mock module with format specs";
        info.tags = tags;

        // Input pin with image types
        ModuleInfo::PinInfo inputPin;
        inputPin.name = "input";
        inputPin.frame_types.push_back("RawImage");
        inputPin.memType = memType;
        inputPin.required = true;
        inputPin.image_types = inputFormats;
        info.inputs.push_back(inputPin);

        // Output pin with image types
        ModuleInfo::PinInfo outputPin;
        outputPin.name = "output";
        outputPin.frame_types.push_back("RawImage");
        outputPin.memType = memType;
        outputPin.required = true;
        outputPin.image_types = outputFormats;
        info.outputs.push_back(outputPin);

        ModuleRegistry::instance().registerModule(std::move(info));
    }

    // Helper to add a module instance to a pipeline
    void addModuleToPipeline(
        PipelineDescription& pipeline,
        const std::string& instanceId,
        const std::string& moduleType
    ) {
        ModuleInstance inst;
        inst.instance_id = instanceId;
        inst.module_type = moduleType;
        pipeline.addModule(std::move(inst));
    }

    // Helper to add a connection to a pipeline
    void addConnectionToPipeline(
        PipelineDescription& pipeline,
        const std::string& fromModule,
        const std::string& toModule,
        const std::string& fromPin = "output",
        const std::string& toPin = "input"
    ) {
        Connection conn;
        conn.from_module = fromModule;
        conn.to_module = toModule;
        conn.from_pin = fromPin;
        conn.to_pin = toPin;
        pipeline.addConnection(std::move(conn));
    }

    // Helper to create a simple pipeline
    PipelineDescription createSimplePipeline(
        const std::string& sourceType,
        const std::string& sinkType
    ) {
        PipelineDescription pipeline;
        pipeline.settings.name = "TestPipeline";

        addModuleToPipeline(pipeline, "source", sourceType);
        addModuleToPipeline(pipeline, "sink", sinkType);
        addConnectionToPipeline(pipeline, "source", "sink");

        return pipeline;
    }

    PipelineAnalyzer analyzer;
};

// ============================================================
// Basic Analysis Tests
// ============================================================

BOOST_FIXTURE_TEST_SUITE(PipelineAnalyzerTests, AnalyzerFixture)

BOOST_AUTO_TEST_CASE(AnalyzeEmptyPipeline_ReturnsNoErrors)
{
    PipelineDescription pipeline;
    pipeline.settings.name = "EmptyPipeline";

    auto result = analyzer.analyze(pipeline);

    BOOST_CHECK(!result.hasErrors);
    BOOST_CHECK(result.errors.empty());
    BOOST_CHECK(result.bridges.empty());
}

BOOST_AUTO_TEST_CASE(AnalyzeCompatiblePipeline_ReturnsNoErrors)
{
    // Register two HOST modules
    registerMockModule("HostSource", FrameMetadata::HOST, FrameMetadata::HOST);
    registerMockModule("HostSink", FrameMetadata::HOST, FrameMetadata::HOST);

    auto pipeline = createSimplePipeline("HostSource", "HostSink");
    auto result = analyzer.analyze(pipeline);

    BOOST_CHECK(!result.hasErrors);
    BOOST_CHECK(result.errors.empty());
    BOOST_CHECK(result.bridges.empty());
}

// ============================================================
// Memory Type Mismatch Tests
// ============================================================

BOOST_AUTO_TEST_CASE(AnalyzeHostToCuda_RequiresMemoryBridge)
{
    // Source outputs HOST, sink expects CUDA_DEVICE
    registerMockModule("HostSource", FrameMetadata::HOST, FrameMetadata::HOST);
    registerMockModule("CudaSink", FrameMetadata::CUDA_DEVICE, FrameMetadata::CUDA_DEVICE);

    auto pipeline = createSimplePipeline("HostSource", "CudaSink");
    auto result = analyzer.analyze(pipeline);

    BOOST_CHECK(!result.hasErrors);
    BOOST_REQUIRE_EQUAL(result.bridges.size(), 1);
    BOOST_CHECK(result.bridges[0].type == BridgeType::Memory);
    BOOST_CHECK_EQUAL(result.bridges[0].bridgeModule, "CudaMemCopy");
    BOOST_CHECK(result.bridges[0].memoryDirection == MemoryDirection::HostToDevice);
    BOOST_CHECK_EQUAL(result.memoryBridgeCount, 1);
}

BOOST_AUTO_TEST_CASE(AnalyzeCudaToHost_RequiresMemoryBridge)
{
    // Source outputs CUDA_DEVICE, sink expects HOST
    registerMockModule("CudaSource", FrameMetadata::CUDA_DEVICE, FrameMetadata::CUDA_DEVICE);
    registerMockModule("HostSink", FrameMetadata::HOST, FrameMetadata::HOST);

    auto pipeline = createSimplePipeline("CudaSource", "HostSink");
    auto result = analyzer.analyze(pipeline);

    BOOST_CHECK(!result.hasErrors);
    BOOST_REQUIRE_EQUAL(result.bridges.size(), 1);
    BOOST_CHECK(result.bridges[0].type == BridgeType::Memory);
    BOOST_CHECK_EQUAL(result.bridges[0].bridgeModule, "CudaMemCopy");
    BOOST_CHECK(result.bridges[0].memoryDirection == MemoryDirection::DeviceToHost);
}

BOOST_AUTO_TEST_CASE(AnalyzeCudaToCuda_NoBridge)
{
    // Both modules use CUDA_DEVICE
    registerMockModule("CudaSource", FrameMetadata::CUDA_DEVICE, FrameMetadata::CUDA_DEVICE);
    registerMockModule("CudaSink", FrameMetadata::CUDA_DEVICE, FrameMetadata::CUDA_DEVICE);

    auto pipeline = createSimplePipeline("CudaSource", "CudaSink");
    auto result = analyzer.analyze(pipeline);

    BOOST_CHECK(!result.hasErrors);
    BOOST_CHECK(result.bridges.empty());
}

// ============================================================
// Frame Type Mismatch Tests
// ============================================================

BOOST_AUTO_TEST_CASE(AnalyzeFrameTypeMismatch_ReturnsError)
{
    // Source outputs H264Frame, sink expects RawImage
    registerMockModule("H264Source", FrameMetadata::HOST, FrameMetadata::HOST,
                       "RawImage", "H264Frame");
    registerMockModule("ImageSink", FrameMetadata::HOST, FrameMetadata::HOST,
                       "RawImage", "RawImage");

    auto pipeline = createSimplePipeline("H264Source", "ImageSink");
    auto result = analyzer.analyze(pipeline);

    BOOST_CHECK(result.hasErrors);
    BOOST_REQUIRE_EQUAL(result.errors.size(), 1);
    BOOST_CHECK_EQUAL(result.errors[0].code, "E001");
}

BOOST_AUTO_TEST_CASE(AnalyzeFrameAcceptsAny_NoError)
{
    // Sink accepts "Frame" which should match any frame type
    registerMockModule("SpecificSource", FrameMetadata::HOST, FrameMetadata::HOST,
                       "Frame", "RawImage");
    registerMockModule("GenericSink", FrameMetadata::HOST, FrameMetadata::HOST,
                       "Frame", "Frame");

    auto pipeline = createSimplePipeline("SpecificSource", "GenericSink");
    auto result = analyzer.analyze(pipeline);

    BOOST_CHECK(!result.hasErrors);
}

// ============================================================
// Format (Pixel Type) Mismatch Tests
// ============================================================

BOOST_AUTO_TEST_CASE(AnalyzeFormatMismatch_RequiresFormatBridge)
{
    // Source outputs BGR, sink requires YUV420
    registerMockModuleWithFormats(
        "BGRSource", FrameMetadata::HOST,
        {}, {ImageMetadata::BGR}
    );
    registerMockModuleWithFormats(
        "YUVSink", FrameMetadata::HOST,
        {ImageMetadata::YUV420}, {}
    );

    auto pipeline = createSimplePipeline("BGRSource", "YUVSink");
    auto result = analyzer.analyze(pipeline);

    BOOST_CHECK(!result.hasErrors);
    BOOST_REQUIRE_EQUAL(result.bridges.size(), 1);
    BOOST_CHECK(result.bridges[0].type == BridgeType::Format);
    BOOST_CHECK_EQUAL(result.bridges[0].bridgeModule, "ColorConversion");
    BOOST_CHECK_EQUAL(result.formatBridgeCount, 1);
}

BOOST_AUTO_TEST_CASE(AnalyzeFormatMismatchOnGPU_UsesCCNPPI)
{
    // Source outputs BGR on CUDA, sink requires NV12 on CUDA
    registerMockModuleWithFormats(
        "CudaBGRSource", FrameMetadata::CUDA_DEVICE,
        {}, {ImageMetadata::BGR},
        {"cuda"}
    );
    registerMockModuleWithFormats(
        "CudaNV12Sink", FrameMetadata::CUDA_DEVICE,
        {ImageMetadata::NV12}, {},
        {"cuda"}
    );

    auto pipeline = createSimplePipeline("CudaBGRSource", "CudaNV12Sink");
    auto result = analyzer.analyze(pipeline);

    BOOST_CHECK(!result.hasErrors);
    BOOST_REQUIRE_EQUAL(result.bridges.size(), 1);
    BOOST_CHECK_EQUAL(result.bridges[0].bridgeModule, "CCNPPI");
}

BOOST_AUTO_TEST_CASE(AnalyzeFormatCompatible_NoBridge)
{
    // Both modules accept BGR
    registerMockModuleWithFormats(
        "BGRSource", FrameMetadata::HOST,
        {}, {ImageMetadata::BGR}
    );
    registerMockModuleWithFormats(
        "BGRSink", FrameMetadata::HOST,
        {ImageMetadata::BGR}, {}
    );

    auto pipeline = createSimplePipeline("BGRSource", "BGRSink");
    auto result = analyzer.analyze(pipeline);

    BOOST_CHECK(!result.hasErrors);
    BOOST_CHECK(result.bridges.empty());
}

// ============================================================
// Combined Mismatch Tests
// ============================================================

BOOST_AUTO_TEST_CASE(AnalyzeMemoryAndFormatMismatch_RequiresBothBridges)
{
    // Source: HOST, BGR output
    // Sink: CUDA_DEVICE, NV12 input
    // Need: Memory bridge (HOST->CUDA) then format bridge (BGR->NV12 on CUDA)
    registerMockModuleWithFormats(
        "HostBGRSource", FrameMetadata::HOST,
        {}, {ImageMetadata::BGR}
    );
    registerMockModuleWithFormats(
        "CudaNV12Sink", FrameMetadata::CUDA_DEVICE,
        {ImageMetadata::NV12}, {},
        {"cuda"}
    );

    auto pipeline = createSimplePipeline("HostBGRSource", "CudaNV12Sink");
    auto result = analyzer.analyze(pipeline);

    BOOST_CHECK(!result.hasErrors);
    BOOST_CHECK_EQUAL(result.bridges.size(), 2);

    // First bridge should be memory (HOST -> CUDA)
    BOOST_CHECK(result.bridges[0].type == BridgeType::Memory);
    BOOST_CHECK_EQUAL(result.bridges[0].bridgeModule, "CudaMemCopy");

    // Second bridge should be format (on CUDA, so CCNPPI)
    BOOST_CHECK(result.bridges[1].type == BridgeType::Format);
    BOOST_CHECK_EQUAL(result.bridges[1].bridgeModule, "CCNPPI");
}

// ============================================================
// Suboptimal Pattern Detection Tests
// ============================================================

BOOST_AUTO_TEST_CASE(AnalyzeDetectsCPUInCudaPipeline)
{
    // Register a CPU module and CUDA modules
    registerMockModule("CudaSource", FrameMetadata::CUDA_DEVICE, FrameMetadata::CUDA_DEVICE,
                       "RawImage", "RawImage", {"cuda", "source"});
    registerMockModule("CPUResize", FrameMetadata::HOST, FrameMetadata::HOST,
                       "RawImage", "RawImage", {"resize", "opencv"});
    registerMockModule("CudaSink", FrameMetadata::CUDA_DEVICE, FrameMetadata::CUDA_DEVICE,
                       "RawImage", "RawImage", {"cuda", "sink"});

    // Also register a CUDA resize alternative
    registerMockModule("CudaResize", FrameMetadata::CUDA_DEVICE, FrameMetadata::CUDA_DEVICE,
                       "RawImage", "RawImage", {"cuda", "resize"});

    // Create pipeline with CPU module in middle
    PipelineDescription pipeline;
    pipeline.settings.name = "MixedPipeline";

    addModuleToPipeline(pipeline, "source", "CudaSource");
    addModuleToPipeline(pipeline, "resize", "CPUResize");
    addModuleToPipeline(pipeline, "sink", "CudaSink");

    addConnectionToPipeline(pipeline, "source", "resize");
    addConnectionToPipeline(pipeline, "resize", "sink");

    auto result = analyzer.analyze(pipeline);

    // Should suggest CUDA alternative for CPUResize
    BOOST_CHECK_GE(result.suggestions.size(), 1);
    bool foundResizeSuggestion = false;
    for (const auto& suggestion : result.suggestions) {
        if (suggestion.currentModule == "CPUResize" &&
            suggestion.suggestedModule == "CudaResize") {
            foundResizeSuggestion = true;
            break;
        }
    }
    BOOST_CHECK(foundResizeSuggestion);
}

// ============================================================
// Multi-connection Pipeline Tests
// ============================================================

BOOST_AUTO_TEST_CASE(AnalyzeChainedPipeline_CorrectBridgeOrdering)
{
    // Chain: HOST -> CUDA -> HOST
    registerMockModule("HostSource", FrameMetadata::HOST, FrameMetadata::HOST);
    registerMockModule("CudaProcessor", FrameMetadata::CUDA_DEVICE, FrameMetadata::CUDA_DEVICE,
                       "RawImage", "RawImage", {"cuda"});
    registerMockModule("HostSink", FrameMetadata::HOST, FrameMetadata::HOST);

    PipelineDescription pipeline;
    pipeline.settings.name = "ChainPipeline";

    addModuleToPipeline(pipeline, "source", "HostSource");
    addModuleToPipeline(pipeline, "proc", "CudaProcessor");
    addModuleToPipeline(pipeline, "sink", "HostSink");

    addConnectionToPipeline(pipeline, "source", "proc");
    addConnectionToPipeline(pipeline, "proc", "sink");

    auto result = analyzer.analyze(pipeline);

    BOOST_CHECK(!result.hasErrors);
    BOOST_CHECK_EQUAL(result.bridges.size(), 2);
    BOOST_CHECK_EQUAL(result.memoryBridgeCount, 2);

    // First bridge: HOST -> CUDA
    BOOST_CHECK_EQUAL(result.bridges[0].fromModule, "source");
    BOOST_CHECK(result.bridges[0].memoryDirection == MemoryDirection::HostToDevice);

    // Second bridge: CUDA -> HOST
    BOOST_CHECK_EQUAL(result.bridges[1].fromModule, "proc");
    BOOST_CHECK(result.bridges[1].memoryDirection == MemoryDirection::DeviceToHost);
}

BOOST_AUTO_TEST_SUITE_END()
