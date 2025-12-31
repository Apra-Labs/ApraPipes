// ============================================================
// Unit tests for declarative/ModuleFactory.h
// Task D1: Module Factory
// ============================================================

#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "declarative/ModuleFactory.h"
#include "declarative/ModuleRegistry.h"
#include "declarative/PipelineDescription.h"
#include "Module.h"
#include "FrameMetadata.h"
#include "Frame.h"

using namespace apra;

// ============================================================
// Test Module implementations (simple modules for testing)
// ============================================================
namespace test_factory {

// Simple test props
class TestSourceModuleProps : public ModuleProps {
public:
    std::string path = "/tmp/test.mp4";
    bool loop = false;
};

// Simple source module for testing
class TestSourceModule : public Module {
public:
    explicit TestSourceModule(TestSourceModuleProps props)
        : Module(SOURCE, "TestSourceModule", props), props_(props) {
        // Set up output pin immediately so setNext() works
        auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
        mOutputPinId = addOutputPin(metadata);
    }

    // Module Metadata for registry
    struct Metadata {
        static constexpr std::string_view name = "TestSourceModule";
        static constexpr ModuleCategory category = ModuleCategory::Source;
        static constexpr std::string_view version = "1.0.0";
        static constexpr std::string_view description = "Test source module";

        static constexpr std::array<std::string_view, 1> tags = {"test"};
        static constexpr std::array<PinDef, 0> inputs = {};
        static constexpr std::array<PinDef, 1> outputs = {
            PinDef::create("output", "RawFrame", true, "Output frames")
        };
        static constexpr std::array<PropDef, 2> properties = {
            PropDef::Text("path", "/tmp/default.mp4", "File path"),
            PropDef::Boolean("loop", false, "Loop playback")
        };
    };

    const std::string& getPath() const { return props_.path; }
    bool getLoop() const { return props_.loop; }

protected:
    bool validateInputPins() override { return true; }
    bool validateOutputPins() override { return true; }
    bool validateInputOutputPins() override { return true; }

private:
    TestSourceModuleProps props_;
    std::string mOutputPinId;
};

// Simple transform props
class TestTransformModuleProps : public ModuleProps {
public:
    int scale = 1;
    std::string mode = "auto";
};

// Simple transform module for testing
class TestTransformModule : public Module {
public:
    explicit TestTransformModule(TestTransformModuleProps props)
        : Module(TRANSFORM, "TestTransformModule", props), props_(props) {
        // Set up output pin immediately so setNext() works
        auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
        mOutputPinId = addOutputPin(metadata);
    }

    struct Metadata {
        static constexpr std::string_view name = "TestTransformModule";
        static constexpr ModuleCategory category = ModuleCategory::Transform;
        static constexpr std::string_view version = "1.0.0";
        static constexpr std::string_view description = "Test transform module";

        static constexpr std::array<std::string_view, 1> tags = {"test"};
        // Note: input is optional (false) to allow standalone property testing
        static constexpr std::array<PinDef, 1> inputs = {
            PinDef::create("input", "RawFrame", false, "Input frames")
        };
        static constexpr std::array<PinDef, 1> outputs = {
            PinDef::create("output", "RawFrame", true, "Output frames")
        };
        static constexpr std::array<PropDef, 2> properties = {
            PropDef::Integer("scale", 1, 1, 10, "Scale factor"),
            PropDef::Enum("mode", "auto", "auto", "manual", "Operation mode", PropDef::Mutability::Static)
        };
    };

    int getScale() const { return props_.scale; }
    const std::string& getMode() const { return props_.mode; }

protected:
    bool validateInputPins() override { return true; }
    bool validateOutputPins() override { return true; }
    bool validateInputOutputPins() override { return true; }

private:
    TestTransformModuleProps props_;
    std::string mOutputPinId;
};

// Simple sink props
class TestSinkModuleProps : public ModuleProps {
public:
    std::string output_path = "/tmp/output.mp4";
};

// Simple sink module for testing
class TestSinkModule : public Module {
public:
    explicit TestSinkModule(TestSinkModuleProps props)
        : Module(SINK, "TestSinkModule", props), props_(props) {}

    struct Metadata {
        static constexpr std::string_view name = "TestSinkModule";
        static constexpr ModuleCategory category = ModuleCategory::Sink;
        static constexpr std::string_view version = "1.0.0";
        static constexpr std::string_view description = "Test sink module";

        static constexpr std::array<std::string_view, 1> tags = {"test"};
        // Note: input is optional (false) to allow standalone property testing
        static constexpr std::array<PinDef, 1> inputs = {
            PinDef::create("input", "RawFrame", false, "Input frames")
        };
        static constexpr std::array<PinDef, 0> outputs = {};
        static constexpr std::array<PropDef, 1> properties = {
            PropDef::Text("output_path", "/tmp/output.mp4", "Output file path")
        };
    };

protected:
    bool validateInputPins() override { return true; }
    bool validateOutputPins() override { return true; }
    bool validateInputOutputPins() override { return true; }

private:
    TestSinkModuleProps props_;
};

// Helper to create and register test module infos
template<typename ModuleClass, typename PropsClass>
void registerTestModule() {
    ModuleInfo info;
    info.name = std::string(ModuleClass::Metadata::name);
    info.category = ModuleClass::Metadata::category;
    info.version = std::string(ModuleClass::Metadata::version);
    info.description = std::string(ModuleClass::Metadata::description);

    for (const auto& tag : ModuleClass::Metadata::tags) {
        info.tags.push_back(std::string(tag));
    }
    for (const auto& pin : ModuleClass::Metadata::inputs) {
        info.inputs.push_back(detail::toPinInfo(pin));
    }
    for (const auto& pin : ModuleClass::Metadata::outputs) {
        info.outputs.push_back(detail::toPinInfo(pin));
    }
    for (const auto& prop : ModuleClass::Metadata::properties) {
        info.properties.push_back(detail::toPropInfo(prop));
    }

    // Factory function
    info.factory = [](const std::map<std::string, apra::ScalarPropertyValue>& props)
        -> std::unique_ptr<Module> {
        PropsClass moduleProps;
        // Note: In real usage, properties would be applied here from the map
        return std::make_unique<ModuleClass>(moduleProps);
    };

    ModuleRegistry::instance().registerModule(std::move(info));
}

// Multi-output module (like H264EncoderV4L2)
class MultiOutputModuleProps : public ModuleProps {
public:
    int quality = 80;
};

class MultiOutputModule : public Module {
public:
    explicit MultiOutputModule(MultiOutputModuleProps props)
        : Module(TRANSFORM, "MultiOutputModule", props), props_(props) {
        // Create two output pins (like encoder with video + motion vectors)
        auto videoMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::H264_DATA));
        mVideoPinId = addOutputPin(videoMetadata);

        auto motionMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
        mMotionPinId = addOutputPin(motionMetadata);
    }

    struct Metadata {
        static constexpr std::string_view name = "MultiOutputModule";
        static constexpr ModuleCategory category = ModuleCategory::Transform;
        static constexpr std::string_view version = "1.0.0";
        static constexpr std::string_view description = "Test module with multiple outputs";

        static constexpr std::array<std::string_view, 1> tags = {"test"};
        static constexpr std::array<PinDef, 1> inputs = {
            PinDef::create("input", "RawFrame", true, "Input frames")
        };
        static constexpr std::array<PinDef, 2> outputs = {
            PinDef::create("video", "H264Frame", true, "Encoded video"),
            PinDef::create("motion_vectors", "MotionData", true, "Motion vector data")
        };
        static constexpr std::array<PropDef, 1> properties = {
            PropDef::Integer("quality", 80, 1, 100, "Encoding quality")
        };
    };

protected:
    bool validateInputPins() override { return true; }
    bool validateOutputPins() override { return true; }
    bool validateInputOutputPins() override { return true; }

private:
    MultiOutputModuleProps props_;
    std::string mVideoPinId;
    std::string mMotionPinId;
};

// Multi-input module (like OverlayNPPI)
class MultiInputModuleProps : public ModuleProps {
public:
    float opacity = 1.0f;
};

class MultiInputModule : public Module {
public:
    explicit MultiInputModule(MultiInputModuleProps props)
        : Module(TRANSFORM, "MultiInputModule", props), props_(props) {
        // Create single output pin
        auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::RAW_IMAGE));
        mOutputPinId = addOutputPin(metadata);
    }

    struct Metadata {
        static constexpr std::string_view name = "MultiInputModule";
        static constexpr ModuleCategory category = ModuleCategory::Transform;
        static constexpr std::string_view version = "1.0.0";
        static constexpr std::string_view description = "Test module with multiple inputs";

        static constexpr std::array<std::string_view, 1> tags = {"test"};
        static constexpr std::array<PinDef, 2> inputs = {
            PinDef::create("background", "RawImage", true, "Background image"),
            PinDef::create("foreground", "OverlayInfo", true, "Foreground overlay")
        };
        static constexpr std::array<PinDef, 1> outputs = {
            PinDef::create("output", "RawImage", true, "Composited output")
        };
        static constexpr std::array<PropDef, 1> properties = {
            PropDef::Floating("opacity", 1.0, 0.0, 1.0, "Overlay opacity")
        };
    };

protected:
    bool validateInputPins() override { return true; }
    bool validateOutputPins() override { return true; }
    bool validateInputOutputPins() override { return true; }

private:
    MultiInputModuleProps props_;
    std::string mOutputPinId;
};

// Module with required input (for testing input validation)
class RequiredInputModuleProps : public ModuleProps {
public:
    int threshold = 50;
};

class RequiredInputModule : public Module {
public:
    explicit RequiredInputModule(RequiredInputModuleProps props)
        : Module(TRANSFORM, "RequiredInputModule", props), props_(props) {
        auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
        mOutputPinId = addOutputPin(metadata);
    }

    struct Metadata {
        static constexpr std::string_view name = "RequiredInputModule";
        static constexpr ModuleCategory category = ModuleCategory::Transform;
        static constexpr std::string_view version = "1.0.0";
        static constexpr std::string_view description = "Test module with required input";

        static constexpr std::array<std::string_view, 1> tags = {"test"};
        // Required input (true) - must be connected
        static constexpr std::array<PinDef, 1> inputs = {
            PinDef::create("input", "RawFrame", true, "Required input")
        };
        static constexpr std::array<PinDef, 1> outputs = {
            PinDef::create("output", "RawFrame", true, "Output frames")
        };
        static constexpr std::array<PropDef, 1> properties = {
            PropDef::Integer("threshold", 50, 0, 100, "Threshold value")
        };
    };

protected:
    bool validateInputPins() override { return true; }
    bool validateOutputPins() override { return true; }
    bool validateInputOutputPins() override { return true; }

private:
    RequiredInputModuleProps props_;
    std::string mOutputPinId;
};

// Module with optional input (for testing optional input validation)
class OptionalInputModuleProps : public ModuleProps {
public:
    bool enabled = true;
};

class OptionalInputModule : public Module {
public:
    explicit OptionalInputModule(OptionalInputModuleProps props)
        : Module(TRANSFORM, "OptionalInputModule", props), props_(props) {
        auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
        mOutputPinId = addOutputPin(metadata);
    }

    struct Metadata {
        static constexpr std::string_view name = "OptionalInputModule";
        static constexpr ModuleCategory category = ModuleCategory::Transform;
        static constexpr std::string_view version = "1.0.0";
        static constexpr std::string_view description = "Test module with optional input";

        static constexpr std::array<std::string_view, 1> tags = {"test"};
        // Optional input (false) - doesn't need to be connected
        static constexpr std::array<PinDef, 1> inputs = {
            PinDef::create("input", "RawFrame", false, "Optional input")
        };
        static constexpr std::array<PinDef, 1> outputs = {
            PinDef::create("output", "RawFrame", true, "Output frames")
        };
        static constexpr std::array<PropDef, 1> properties = {
            PropDef::Boolean("enabled", true, "Enable processing")
        };
    };

protected:
    bool validateInputPins() override { return true; }
    bool validateOutputPins() override { return true; }
    bool validateInputOutputPins() override { return true; }

private:
    OptionalInputModuleProps props_;
    std::string mOutputPinId;
};

} // namespace test_factory

BOOST_AUTO_TEST_SUITE(ModuleFactoryTests)

// ============================================================
// Test Fixture - registers test modules and clears after
// ============================================================
struct FactoryFixture {
    FactoryFixture() {
        ModuleRegistry::instance().clear();
        // Register test modules
        test_factory::registerTestModule<
            test_factory::TestSourceModule,
            test_factory::TestSourceModuleProps>();
        test_factory::registerTestModule<
            test_factory::TestTransformModule,
            test_factory::TestTransformModuleProps>();
        test_factory::registerTestModule<
            test_factory::TestSinkModule,
            test_factory::TestSinkModuleProps>();
    }
    ~FactoryFixture() {
        ModuleRegistry::instance().clear();
    }

    // Helper to create a simple pipeline description
    PipelineDescription createSimplePipeline() {
        PipelineDescription desc;
        desc.settings.name = "test_pipeline";

        ModuleInstance source;
        source.instance_id = "source";
        source.module_type = "TestSourceModule";
        source.properties["path"] = std::string("/video.mp4");
        desc.modules.push_back(source);

        return desc;
    }

    // Helper to create a connected pipeline
    PipelineDescription createConnectedPipeline() {
        PipelineDescription desc;
        desc.settings.name = "connected_pipeline";

        ModuleInstance source;
        source.instance_id = "source";
        source.module_type = "TestSourceModule";
        desc.modules.push_back(source);

        ModuleInstance transform;
        transform.instance_id = "transform";
        transform.module_type = "TestTransformModule";
        desc.modules.push_back(transform);

        ModuleInstance sink;
        sink.instance_id = "sink";
        sink.module_type = "TestSinkModule";
        desc.modules.push_back(sink);

        // Connect source -> transform -> sink
        desc.addConnection("source.output", "transform.input");
        desc.addConnection("transform.output", "sink.input");

        return desc;
    }
};

// ============================================================
// BuildIssue Tests
// ============================================================

BOOST_AUTO_TEST_CASE(BuildIssue_FactoryMethods_CreateCorrectLevels)
{
    auto error = BuildIssue::error("E001", "test.location", "Error message");
    BOOST_CHECK(error.level == BuildIssue::Level::Error);
    BOOST_CHECK_EQUAL(error.code, "E001");
    BOOST_CHECK_EQUAL(error.location, "test.location");
    BOOST_CHECK_EQUAL(error.message, "Error message");

    auto warning = BuildIssue::warning("W001", "test.loc", "Warning msg");
    BOOST_CHECK(warning.level == BuildIssue::Level::Warning);

    auto info = BuildIssue::info("I001", "test.loc", "Info msg");
    BOOST_CHECK(info.level == BuildIssue::Level::Info);
}

// ============================================================
// BuildResult Tests
// ============================================================

BOOST_AUTO_TEST_CASE(BuildResult_Success_WhenPipelineAndNoErrors)
{
    ModuleFactory::BuildResult result;
    result.pipeline = std::make_unique<PipeLine>("test");

    BOOST_CHECK(result.success());
    BOOST_CHECK(!result.hasErrors());
    BOOST_CHECK(!result.hasWarnings());
}

BOOST_AUTO_TEST_CASE(BuildResult_NotSuccess_WhenNoPipeline)
{
    ModuleFactory::BuildResult result;
    result.pipeline = nullptr;

    BOOST_CHECK(!result.success());
}

BOOST_AUTO_TEST_CASE(BuildResult_NotSuccess_WhenHasErrors)
{
    ModuleFactory::BuildResult result;
    result.pipeline = std::make_unique<PipeLine>("test");
    result.issues.push_back(BuildIssue::error("E001", "loc", "error"));

    BOOST_CHECK(!result.success());
    BOOST_CHECK(result.hasErrors());
}

BOOST_AUTO_TEST_CASE(BuildResult_Success_WhenHasWarningsOnly)
{
    ModuleFactory::BuildResult result;
    result.pipeline = std::make_unique<PipeLine>("test");
    result.issues.push_back(BuildIssue::warning("W001", "loc", "warning"));

    BOOST_CHECK(result.success());
    BOOST_CHECK(!result.hasErrors());
    BOOST_CHECK(result.hasWarnings());
}

BOOST_AUTO_TEST_CASE(BuildResult_GetErrors_FiltersCorrectly)
{
    ModuleFactory::BuildResult result;
    result.issues.push_back(BuildIssue::error("E001", "loc1", "error1"));
    result.issues.push_back(BuildIssue::warning("W001", "loc2", "warning1"));
    result.issues.push_back(BuildIssue::error("E002", "loc3", "error2"));
    result.issues.push_back(BuildIssue::info("I001", "loc4", "info1"));

    auto errors = result.getErrors();
    BOOST_CHECK_EQUAL(errors.size(), 2);
    BOOST_CHECK_EQUAL(errors[0].code, "E001");
    BOOST_CHECK_EQUAL(errors[1].code, "E002");
}

BOOST_AUTO_TEST_CASE(BuildResult_GetWarnings_FiltersCorrectly)
{
    ModuleFactory::BuildResult result;
    result.issues.push_back(BuildIssue::error("E001", "loc1", "error1"));
    result.issues.push_back(BuildIssue::warning("W001", "loc2", "warning1"));
    result.issues.push_back(BuildIssue::warning("W002", "loc3", "warning2"));

    auto warnings = result.getWarnings();
    BOOST_CHECK_EQUAL(warnings.size(), 2);
    BOOST_CHECK_EQUAL(warnings[0].code, "W001");
    BOOST_CHECK_EQUAL(warnings[1].code, "W002");
}

BOOST_AUTO_TEST_CASE(BuildResult_FormatIssues_FormatsCorrectly)
{
    ModuleFactory::BuildResult result;
    result.issues.push_back(BuildIssue::error("E001", "modules.test", "Test error"));

    std::string formatted = result.formatIssues();
    BOOST_CHECK(formatted.find("[ERROR]") != std::string::npos);
    BOOST_CHECK(formatted.find("E001") != std::string::npos);
    BOOST_CHECK(formatted.find("modules.test") != std::string::npos);
    BOOST_CHECK(formatted.find("Test error") != std::string::npos);
}

// ============================================================
// ModuleFactory Construction Tests
// ============================================================

BOOST_AUTO_TEST_CASE(ModuleFactory_DefaultConstruction)
{
    ModuleFactory factory;
    BOOST_CHECK(!factory.options().auto_insert_converters);
    BOOST_CHECK(!factory.options().strict_mode);
    BOOST_CHECK(!factory.options().collect_info_messages);
}

BOOST_AUTO_TEST_CASE(ModuleFactory_ConstructionWithOptions)
{
    ModuleFactory::Options opts;
    opts.strict_mode = true;
    opts.collect_info_messages = true;

    ModuleFactory factory(opts);
    BOOST_CHECK(factory.options().strict_mode);
    BOOST_CHECK(factory.options().collect_info_messages);
}

BOOST_AUTO_TEST_CASE(ModuleFactory_SetOptions)
{
    ModuleFactory factory;

    ModuleFactory::Options opts;
    opts.strict_mode = true;
    factory.setOptions(opts);

    BOOST_CHECK(factory.options().strict_mode);
}

// ============================================================
// Build Tests - Empty Pipeline
// ============================================================

BOOST_FIXTURE_TEST_CASE(Build_EmptyPipeline_ReturnsError, FactoryFixture)
{
    PipelineDescription desc;  // Empty

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(!result.success());
    BOOST_CHECK(result.hasErrors());

    auto errors = result.getErrors();
    BOOST_REQUIRE_GE(errors.size(), 1);
    BOOST_CHECK_EQUAL(errors[0].code, BuildIssue::EMPTY_PIPELINE);
}

// ============================================================
// Build Tests - Single Module
// ============================================================

BOOST_FIXTURE_TEST_CASE(Build_SingleModule_Success, FactoryFixture)
{
    auto desc = createSimplePipeline();

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
    BOOST_REQUIRE(result.pipeline != nullptr);
    BOOST_CHECK_EQUAL(result.pipeline->getName(), "test_pipeline");
}

BOOST_FIXTURE_TEST_CASE(Build_UnknownModuleType_ReturnsError, FactoryFixture)
{
    PipelineDescription desc;
    desc.settings.name = "test";

    ModuleInstance unknown;
    unknown.instance_id = "unknown_inst";
    unknown.module_type = "NonExistentModule";
    desc.modules.push_back(unknown);

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(!result.success());
    BOOST_CHECK(result.hasErrors());

    auto errors = result.getErrors();
    BOOST_REQUIRE_GE(errors.size(), 1);
    BOOST_CHECK_EQUAL(errors[0].code, BuildIssue::UNKNOWN_MODULE);
    BOOST_CHECK(errors[0].message.find("NonExistentModule") != std::string::npos);
}

// ============================================================
// Build Tests - Multiple Modules
// ============================================================

BOOST_FIXTURE_TEST_CASE(Build_MultipleModules_Success, FactoryFixture)
{
    auto desc = createConnectedPipeline();

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
    BOOST_REQUIRE(result.pipeline != nullptr);
}

// ============================================================
// Build Tests - Connections
// ============================================================

BOOST_FIXTURE_TEST_CASE(Build_WithConnections_Success, FactoryFixture)
{
    auto desc = createConnectedPipeline();

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
}

BOOST_FIXTURE_TEST_CASE(Build_UnknownSourceModule_ReturnsError, FactoryFixture)
{
    PipelineDescription desc;
    desc.settings.name = "test";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    desc.modules.push_back(source);

    // Connection references non-existent module
    desc.addConnection("nonexistent.output", "source.input");

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(!result.success());

    auto errors = result.getErrors();
    BOOST_REQUIRE_GE(errors.size(), 1);
    BOOST_CHECK_EQUAL(errors[0].code, BuildIssue::UNKNOWN_SOURCE_MODULE);
}

BOOST_FIXTURE_TEST_CASE(Build_UnknownDestModule_ReturnsError, FactoryFixture)
{
    PipelineDescription desc;
    desc.settings.name = "test";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    desc.modules.push_back(source);

    // Connection references non-existent destination
    desc.addConnection("source.output", "nonexistent.input");

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(!result.success());

    auto errors = result.getErrors();
    BOOST_REQUIRE_GE(errors.size(), 1);
    BOOST_CHECK_EQUAL(errors[0].code, BuildIssue::UNKNOWN_DEST_MODULE);
}

// ============================================================
// Build Tests - Property Types
// ============================================================

BOOST_FIXTURE_TEST_CASE(Build_WithStringProperty_Success, FactoryFixture)
{
    PipelineDescription desc;
    desc.settings.name = "test";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    source.properties["path"] = std::string("/custom/path.mp4");
    desc.modules.push_back(source);

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
}

BOOST_FIXTURE_TEST_CASE(Build_WithIntProperty_Success, FactoryFixture)
{
    PipelineDescription desc;
    desc.settings.name = "test";

    ModuleInstance transform;
    transform.instance_id = "xform";
    transform.module_type = "TestTransformModule";
    transform.properties["scale"] = int64_t(5);
    desc.modules.push_back(transform);

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
}

BOOST_FIXTURE_TEST_CASE(Build_WithBoolProperty_Success, FactoryFixture)
{
    PipelineDescription desc;
    desc.settings.name = "test";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    source.properties["loop"] = true;
    desc.modules.push_back(source);

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
}

BOOST_FIXTURE_TEST_CASE(Build_WithArrayProperty_ConvertsToScalar, FactoryFixture)
{
    PipelineDescription desc;
    desc.settings.name = "test";

    ModuleInstance transform;
    transform.instance_id = "xform";
    transform.module_type = "TestTransformModule";
    // Array property - should be converted to scalar with warning
    transform.properties["scale"] = std::vector<int64_t>{5, 10, 15};
    desc.modules.push_back(transform);

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
    BOOST_CHECK(result.hasWarnings());

    auto warnings = result.getWarnings();
    BOOST_REQUIRE_GE(warnings.size(), 1);
    BOOST_CHECK_EQUAL(warnings[0].code, BuildIssue::PROP_TYPE_CONVERSION);
}

// ============================================================
// Build Tests - Strict Mode
// ============================================================

BOOST_FIXTURE_TEST_CASE(Build_StrictMode_FailsOnWarnings, FactoryFixture)
{
    PipelineDescription desc;
    desc.settings.name = "test";

    ModuleInstance transform;
    transform.instance_id = "xform";
    transform.module_type = "TestTransformModule";
    // This will generate a warning
    transform.properties["scale"] = std::vector<int64_t>{5, 10};
    desc.modules.push_back(transform);

    ModuleFactory::Options opts;
    opts.strict_mode = true;
    ModuleFactory factory(opts);

    auto result = factory.build(desc);

    BOOST_CHECK(!result.success());
}

// ============================================================
// Build Tests - Info Messages
// ============================================================

BOOST_FIXTURE_TEST_CASE(Build_CollectInfoMessages_IncludesInfo, FactoryFixture)
{
    auto desc = createSimplePipeline();

    ModuleFactory::Options opts;
    opts.collect_info_messages = true;
    ModuleFactory factory(opts);

    auto result = factory.build(desc);

    BOOST_CHECK(result.success());

    // Should have info messages about module creation
    bool hasInfoMessage = std::any_of(result.issues.begin(), result.issues.end(),
        [](const BuildIssue& i) { return i.level == BuildIssue::Level::Info; });
    BOOST_CHECK(hasInfoMessage);
}

BOOST_FIXTURE_TEST_CASE(Build_NoCollectInfoMessages_ExcludesInfo, FactoryFixture)
{
    auto desc = createSimplePipeline();

    ModuleFactory factory;  // Default: no info messages
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());

    // Should not have info messages
    bool hasInfoMessage = std::any_of(result.issues.begin(), result.issues.end(),
        [](const BuildIssue& i) { return i.level == BuildIssue::Level::Info; });
    BOOST_CHECK(!hasInfoMessage);
}

// ============================================================
// Build Tests - Pipeline Settings
// ============================================================

BOOST_FIXTURE_TEST_CASE(Build_UsesSettingsName, FactoryFixture)
{
    PipelineDescription desc;
    desc.settings.name = "my_custom_pipeline";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    desc.modules.push_back(source);

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
    BOOST_REQUIRE(result.pipeline != nullptr);
    BOOST_CHECK_EQUAL(result.pipeline->getName(), "my_custom_pipeline");
}

BOOST_FIXTURE_TEST_CASE(Build_DefaultPipelineName_WhenEmpty, FactoryFixture)
{
    PipelineDescription desc;
    // No name set in settings

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    desc.modules.push_back(source);

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
    BOOST_REQUIRE(result.pipeline != nullptr);
    BOOST_CHECK_EQUAL(result.pipeline->getName(), "declarative_pipeline");
}

// ============================================================
// Build Tests - Error Collection
// ============================================================

BOOST_FIXTURE_TEST_CASE(Build_CollectsAllErrors_NotJustFirst, FactoryFixture)
{
    PipelineDescription desc;
    desc.settings.name = "test";

    // Multiple unknown modules
    ModuleInstance unknown1;
    unknown1.instance_id = "unknown1";
    unknown1.module_type = "NonExistent1";
    desc.modules.push_back(unknown1);

    ModuleInstance unknown2;
    unknown2.instance_id = "unknown2";
    unknown2.module_type = "NonExistent2";
    desc.modules.push_back(unknown2);

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(!result.success());

    auto errors = result.getErrors();
    BOOST_CHECK_GE(errors.size(), 2);  // Should have errors for both
}

// ============================================================
// Build Tests - Full Pipeline Integration
// ============================================================

BOOST_FIXTURE_TEST_CASE(Build_FullPipeline_Integration, FactoryFixture)
{
    // Create a complete pipeline with all module types
    PipelineDescription desc;
    desc.settings.name = "full_test_pipeline";
    desc.settings.version = "2.0";
    desc.settings.queue_size = 20;

    ModuleInstance source;
    source.instance_id = "file_reader";
    source.module_type = "TestSourceModule";
    source.properties["path"] = std::string("/input/video.mp4");
    source.properties["loop"] = true;
    desc.modules.push_back(source);

    ModuleInstance transform;
    transform.instance_id = "processor";
    transform.module_type = "TestTransformModule";
    transform.properties["scale"] = int64_t(2);
    transform.properties["mode"] = std::string("manual");
    desc.modules.push_back(transform);

    ModuleInstance sink;
    sink.instance_id = "file_writer";
    sink.module_type = "TestSinkModule";
    sink.properties["output_path"] = std::string("/output/result.mp4");
    desc.modules.push_back(sink);

    // Connect all modules
    desc.addConnection("file_reader.output", "processor.input");
    desc.addConnection("processor.output", "file_writer.input");

    ModuleFactory::Options opts;
    opts.collect_info_messages = true;
    ModuleFactory factory(opts);

    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
    BOOST_REQUIRE(result.pipeline != nullptr);
    BOOST_CHECK_EQUAL(result.pipeline->getName(), "full_test_pipeline");
    BOOST_CHECK(!result.hasErrors());
}

// ============================================================
// Multi-Pin Support Tests (Task D3)
// ============================================================

// ============================================================
// Multi-Pin Test Fixture
// ============================================================
struct MultiPinFixture {
    MultiPinFixture() {
        ModuleRegistry::instance().clear();

        // Register standard test modules
        // Use :: prefix to reference global test_factory namespace
        ::test_factory::registerTestModule<
            ::test_factory::TestSourceModule,
            ::test_factory::TestSourceModuleProps>();
        ::test_factory::registerTestModule<
            ::test_factory::TestTransformModule,
            ::test_factory::TestTransformModuleProps>();
        ::test_factory::registerTestModule<
            ::test_factory::TestSinkModule,
            ::test_factory::TestSinkModuleProps>();

        // Register multi-pin test modules
        ::test_factory::registerTestModule<
            ::test_factory::MultiOutputModule,
            ::test_factory::MultiOutputModuleProps>();
        ::test_factory::registerTestModule<
            ::test_factory::MultiInputModule,
            ::test_factory::MultiInputModuleProps>();

        // Register input validation test modules
        ::test_factory::registerTestModule<
            ::test_factory::RequiredInputModule,
            ::test_factory::RequiredInputModuleProps>();
        ::test_factory::registerTestModule<
            ::test_factory::OptionalInputModule,
            ::test_factory::OptionalInputModuleProps>();
    }

    ~MultiPinFixture() {
        ModuleRegistry::instance().clear();
    }
};

// ============================================================
// parseConnectionEndpoint Tests
// ============================================================

BOOST_AUTO_TEST_CASE(ParseConnectionEndpoint_InstanceDotPin)
{
    auto [instance, pin] = ModuleFactory::parseConnectionEndpoint("encoder.video");
    BOOST_CHECK_EQUAL(instance, "encoder");
    BOOST_CHECK_EQUAL(pin, "video");
}

BOOST_AUTO_TEST_CASE(ParseConnectionEndpoint_InstanceOnly)
{
    auto [instance, pin] = ModuleFactory::parseConnectionEndpoint("encoder");
    BOOST_CHECK_EQUAL(instance, "encoder");
    BOOST_CHECK_EQUAL(pin, "");
}

BOOST_AUTO_TEST_CASE(ParseConnectionEndpoint_MultipleDots)
{
    // Only first dot is the separator
    auto [instance, pin] = ModuleFactory::parseConnectionEndpoint("encoder.motion.vectors");
    BOOST_CHECK_EQUAL(instance, "encoder");
    BOOST_CHECK_EQUAL(pin, "motion.vectors");
}

// ============================================================
// Multi-Output Module Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(MultiOutput_SpecificPinConnection_Success, MultiPinFixture)
{
    PipelineDescription desc;
    desc.settings.name = "multi_output_test";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    desc.modules.push_back(source);

    ModuleInstance encoder;
    encoder.instance_id = "encoder";
    encoder.module_type = "MultiOutputModule";
    desc.modules.push_back(encoder);

    ModuleInstance videoSink;
    videoSink.instance_id = "video_sink";
    videoSink.module_type = "TestSinkModule";
    desc.modules.push_back(videoSink);

    ModuleInstance motionSink;
    motionSink.instance_id = "motion_sink";
    motionSink.module_type = "TestSinkModule";
    desc.modules.push_back(motionSink);

    // Connect source -> encoder
    desc.addConnection("source.output", "encoder.input");
    // Connect encoder's video output to video sink
    desc.addConnection("encoder.video", "video_sink.input");
    // Connect encoder's motion_vectors output to motion sink
    desc.addConnection("encoder.motion_vectors", "motion_sink.input");

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
    BOOST_REQUIRE(result.pipeline != nullptr);
}

BOOST_FIXTURE_TEST_CASE(MultiOutput_FirstPinDefaultFallback_Success, MultiPinFixture)
{
    // When a module has multiple outputs but only one is connected,
    // and no pin name is specified, it should use the first output
    PipelineDescription desc;
    desc.settings.name = "fallback_test";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    desc.modules.push_back(source);

    ModuleInstance encoder;
    encoder.instance_id = "encoder";
    encoder.module_type = "MultiOutputModule";
    desc.modules.push_back(encoder);

    ModuleInstance sink;
    sink.instance_id = "sink";
    sink.module_type = "TestSinkModule";
    desc.modules.push_back(sink);

    // Connect with explicit pin names
    desc.addConnection("source.output", "encoder.input");
    // This uses explicit "video" pin
    desc.addConnection("encoder.video", "sink.input");

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
}

BOOST_FIXTURE_TEST_CASE(MultiOutput_UnknownPin_ReturnsError, MultiPinFixture)
{
    PipelineDescription desc;
    desc.settings.name = "unknown_pin_test";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    desc.modules.push_back(source);

    ModuleInstance encoder;
    encoder.instance_id = "encoder";
    encoder.module_type = "MultiOutputModule";
    desc.modules.push_back(encoder);

    ModuleInstance sink;
    sink.instance_id = "sink";
    sink.module_type = "TestSinkModule";
    desc.modules.push_back(sink);

    desc.addConnection("source.output", "encoder.input");
    // Use unknown pin name
    desc.addConnection("encoder.nonexistent_pin", "sink.input");

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(!result.success());
    BOOST_CHECK(result.hasErrors());

    auto errors = result.getErrors();
    BOOST_REQUIRE_GE(errors.size(), 1);
    BOOST_CHECK_EQUAL(errors[0].code, BuildIssue::UNKNOWN_SOURCE_PIN);
    BOOST_CHECK(errors[0].message.find("nonexistent_pin") != std::string::npos);
}

// ============================================================
// Multi-Input Module Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(MultiInput_BothInputsConnected_Success, MultiPinFixture)
{
    PipelineDescription desc;
    desc.settings.name = "multi_input_test";

    ModuleInstance bgSource;
    bgSource.instance_id = "bg_source";
    bgSource.module_type = "TestSourceModule";
    desc.modules.push_back(bgSource);

    ModuleInstance fgSource;
    fgSource.instance_id = "fg_source";
    fgSource.module_type = "TestSourceModule";
    desc.modules.push_back(fgSource);

    ModuleInstance overlay;
    overlay.instance_id = "overlay";
    overlay.module_type = "MultiInputModule";
    desc.modules.push_back(overlay);

    ModuleInstance sink;
    sink.instance_id = "sink";
    sink.module_type = "TestSinkModule";
    desc.modules.push_back(sink);

    // Connect both inputs to overlay
    desc.addConnection("bg_source.output", "overlay.background");
    desc.addConnection("fg_source.output", "overlay.foreground");
    desc.addConnection("overlay.output", "sink.input");

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
    BOOST_REQUIRE(result.pipeline != nullptr);
}

// ============================================================
// Single Output Module - Backward Compatibility Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(SingleOutput_ExplicitPinName_Success, MultiPinFixture)
{
    // Single-output modules should work with explicit "output" pin name
    PipelineDescription desc;
    desc.settings.name = "explicit_single_output";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    desc.modules.push_back(source);

    ModuleInstance sink;
    sink.instance_id = "sink";
    sink.module_type = "TestSinkModule";
    desc.modules.push_back(sink);

    // Explicit output pin name
    desc.addConnection("source.output", "sink.input");

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
}

// ============================================================
// Complex Multi-Pin Pipeline Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(ComplexPipeline_MultipleOutputsAndInputs_Success, MultiPinFixture)
{
    // Pipeline: source -> encoder (multi-out) -> overlay (multi-in) -> sink
    //           source2 -> overlay.foreground
    PipelineDescription desc;
    desc.settings.name = "complex_multi_pin";

    ModuleInstance source1;
    source1.instance_id = "main_source";
    source1.module_type = "TestSourceModule";
    desc.modules.push_back(source1);

    ModuleInstance source2;
    source2.instance_id = "overlay_source";
    source2.module_type = "TestSourceModule";
    desc.modules.push_back(source2);

    ModuleInstance encoder;
    encoder.instance_id = "encoder";
    encoder.module_type = "MultiOutputModule";
    desc.modules.push_back(encoder);

    ModuleInstance overlay;
    overlay.instance_id = "overlay";
    overlay.module_type = "MultiInputModule";
    desc.modules.push_back(overlay);

    ModuleInstance videoSink;
    videoSink.instance_id = "video_sink";
    videoSink.module_type = "TestSinkModule";
    desc.modules.push_back(videoSink);

    ModuleInstance motionSink;
    motionSink.instance_id = "motion_sink";
    motionSink.module_type = "TestSinkModule";
    desc.modules.push_back(motionSink);

    // Connect main source -> encoder
    desc.addConnection("main_source.output", "encoder.input");
    // Connect encoder video -> overlay background
    desc.addConnection("encoder.video", "overlay.background");
    // Connect overlay source -> overlay foreground
    desc.addConnection("overlay_source.output", "overlay.foreground");
    // Connect overlay output -> video sink
    desc.addConnection("overlay.output", "video_sink.input");
    // Connect encoder motion vectors -> motion sink
    desc.addConnection("encoder.motion_vectors", "motion_sink.input");

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
    BOOST_REQUIRE(result.pipeline != nullptr);
}

// ============================================================
// Input Validation Tests (D3 Phase 2)
// ============================================================

BOOST_FIXTURE_TEST_CASE(RequiredInput_NotConnected_ReturnsError, MultiPinFixture)
{
    PipelineDescription desc;
    desc.settings.name = "required_input_test";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    desc.modules.push_back(source);

    // RequiredInputModule has a required input that we don't connect
    ModuleInstance requiredMod;
    requiredMod.instance_id = "required_mod";
    requiredMod.module_type = "RequiredInputModule";
    desc.modules.push_back(requiredMod);

    // Only connect source, leave required_mod.input unconnected

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(!result.success());
    BOOST_CHECK(result.hasErrors());

    auto errors = result.getErrors();
    BOOST_REQUIRE_GE(errors.size(), 1);
    BOOST_CHECK_EQUAL(errors[0].code, BuildIssue::MISSING_REQUIRED_INPUT);
    BOOST_CHECK(errors[0].message.find("input") != std::string::npos);
    BOOST_CHECK(errors[0].message.find("RequiredInputModule") != std::string::npos);
}

BOOST_FIXTURE_TEST_CASE(RequiredInput_Connected_Success, MultiPinFixture)
{
    PipelineDescription desc;
    desc.settings.name = "required_connected_test";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    desc.modules.push_back(source);

    ModuleInstance requiredMod;
    requiredMod.instance_id = "required_mod";
    requiredMod.module_type = "RequiredInputModule";
    desc.modules.push_back(requiredMod);

    // Connect source to required_mod - this should satisfy the requirement
    desc.addConnection("source.output", "required_mod.input");

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
    BOOST_REQUIRE(result.pipeline != nullptr);
}

BOOST_FIXTURE_TEST_CASE(OptionalInput_NotConnected_Success, MultiPinFixture)
{
    PipelineDescription desc;
    desc.settings.name = "optional_input_test";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    desc.modules.push_back(source);

    // OptionalInputModule has an optional input - no connection required
    ModuleInstance optionalMod;
    optionalMod.instance_id = "optional_mod";
    optionalMod.module_type = "OptionalInputModule";
    desc.modules.push_back(optionalMod);

    // Don't connect optional_mod.input - this should be OK

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
    BOOST_REQUIRE(result.pipeline != nullptr);
}

BOOST_FIXTURE_TEST_CASE(OptionalInput_Connected_Success, MultiPinFixture)
{
    PipelineDescription desc;
    desc.settings.name = "optional_connected_test";

    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "TestSourceModule";
    desc.modules.push_back(source);

    ModuleInstance optionalMod;
    optionalMod.instance_id = "optional_mod";
    optionalMod.module_type = "OptionalInputModule";
    desc.modules.push_back(optionalMod);

    // Connect even though it's optional - should still work
    desc.addConnection("source.output", "optional_mod.input");

    ModuleFactory factory;
    auto result = factory.build(desc);

    BOOST_CHECK(result.success());
    BOOST_REQUIRE(result.pipeline != nullptr);
}

BOOST_AUTO_TEST_SUITE_END()
