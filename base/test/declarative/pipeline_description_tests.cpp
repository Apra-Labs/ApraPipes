// ============================================================
// File: test/declarative/pipeline_description_tests.cpp
// Tests for Pipeline Description IR
// Task B1: Pipeline Description IR
// ============================================================

#include <boost/test/unit_test.hpp>
#include "declarative/PipelineDescription.h"

using namespace apra;

BOOST_AUTO_TEST_SUITE(PipelineDescriptionTests)

// ============================================================
// ModuleInstance Tests
// ============================================================

BOOST_AUTO_TEST_CASE(ModuleInstance_DefaultConstruction)
{
    ModuleInstance m;
    BOOST_CHECK(m.instance_id.empty());
    BOOST_CHECK(m.module_type.empty());
    BOOST_CHECK(m.properties.empty());
    BOOST_CHECK_EQUAL(m.source_line, -1);
    BOOST_CHECK_EQUAL(m.source_column, -1);
}

BOOST_AUTO_TEST_CASE(ModuleInstance_HoldsInstanceIdAndType)
{
    ModuleInstance m;
    m.instance_id = "my_decoder";
    m.module_type = "H264DecoderNvCodec";

    BOOST_CHECK_EQUAL(m.instance_id, "my_decoder");
    BOOST_CHECK_EQUAL(m.module_type, "H264DecoderNvCodec");
}

BOOST_AUTO_TEST_CASE(ModuleInstance_HoldsProperties)
{
    ModuleInstance m;
    m.instance_id = "source";
    m.module_type = "FileReaderModule";
    m.properties["path"] = std::string("/path/to/video.mp4");
    m.properties["loop"] = true;
    m.properties["fps"] = 30.0;
    m.properties["buffer_count"] = int64_t(4);

    BOOST_CHECK_EQUAL(m.properties.size(), 4u);
    BOOST_CHECK_EQUAL(std::get<std::string>(m.properties["path"]), "/path/to/video.mp4");
    BOOST_CHECK_EQUAL(std::get<bool>(m.properties["loop"]), true);
    BOOST_CHECK_EQUAL(std::get<double>(m.properties["fps"]), 30.0);
    BOOST_CHECK_EQUAL(std::get<int64_t>(m.properties["buffer_count"]), 4);
}

// ============================================================
// PropertyValue Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PropertyValue_HoldsInt64)
{
    PropertyValue v = int64_t(8080);
    BOOST_CHECK_EQUAL(std::get<int64_t>(v), 8080);
}

BOOST_AUTO_TEST_CASE(PropertyValue_HoldsDouble)
{
    PropertyValue v = 3.14159;
    BOOST_CHECK_CLOSE(std::get<double>(v), 3.14159, 0.0001);
}

BOOST_AUTO_TEST_CASE(PropertyValue_HoldsBool)
{
    PropertyValue v = true;
    BOOST_CHECK_EQUAL(std::get<bool>(v), true);
}

BOOST_AUTO_TEST_CASE(PropertyValue_HoldsString)
{
    PropertyValue v = std::string("rtsp://localhost:8554/stream");
    BOOST_CHECK_EQUAL(std::get<std::string>(v), "rtsp://localhost:8554/stream");
}

BOOST_AUTO_TEST_CASE(PropertyValue_HoldsIntVector)
{
    PropertyValue v = std::vector<int64_t>{1, 2, 3, 4, 5};
    auto vec = std::get<std::vector<int64_t>>(v);
    BOOST_CHECK_EQUAL(vec.size(), 5u);
    BOOST_CHECK_EQUAL(vec[0], 1);
    BOOST_CHECK_EQUAL(vec[4], 5);
}

BOOST_AUTO_TEST_CASE(PropertyValue_HoldsDoubleVector)
{
    PropertyValue v = std::vector<double>{1.1, 2.2, 3.3};
    auto vec = std::get<std::vector<double>>(v);
    BOOST_CHECK_EQUAL(vec.size(), 3u);
    BOOST_CHECK_CLOSE(vec[1], 2.2, 0.0001);
}

BOOST_AUTO_TEST_CASE(PropertyValue_HoldsStringVector)
{
    PropertyValue v = std::vector<std::string>{"a", "b", "c"};
    auto vec = std::get<std::vector<std::string>>(v);
    BOOST_CHECK_EQUAL(vec.size(), 3u);
    BOOST_CHECK_EQUAL(vec[0], "a");
    BOOST_CHECK_EQUAL(vec[2], "c");
}

// ============================================================
// Connection Tests
// ============================================================

BOOST_AUTO_TEST_CASE(Connection_DefaultConstruction)
{
    Connection c;
    BOOST_CHECK(c.from_module.empty());
    BOOST_CHECK(c.from_pin.empty());
    BOOST_CHECK(c.to_module.empty());
    BOOST_CHECK(c.to_pin.empty());
    BOOST_CHECK(!c.isValid());
}

BOOST_AUTO_TEST_CASE(Connection_StoresFields)
{
    Connection c;
    c.from_module = "source";
    c.from_pin = "output";
    c.to_module = "decoder";
    c.to_pin = "input";

    BOOST_CHECK_EQUAL(c.from_module, "source");
    BOOST_CHECK_EQUAL(c.from_pin, "output");
    BOOST_CHECK_EQUAL(c.to_module, "decoder");
    BOOST_CHECK_EQUAL(c.to_pin, "input");
    BOOST_CHECK(c.isValid());
}

BOOST_AUTO_TEST_CASE(Connection_ParseModuleDotPin)
{
    auto c = Connection::parse("source.output", "decoder.input");

    BOOST_CHECK_EQUAL(c.from_module, "source");
    BOOST_CHECK_EQUAL(c.from_pin, "output");
    BOOST_CHECK_EQUAL(c.to_module, "decoder");
    BOOST_CHECK_EQUAL(c.to_pin, "input");
    BOOST_CHECK(c.isValid());
}

BOOST_AUTO_TEST_CASE(Connection_ParseHandlesMissingDot)
{
    auto c = Connection::parse("source", "decoder.input");

    // from side has no dot: module name is set, pin is empty (use default)
    BOOST_CHECK_EQUAL(c.from_module, "source");
    BOOST_CHECK(c.from_pin.empty());
    BOOST_CHECK_EQUAL(c.to_module, "decoder");
    BOOST_CHECK_EQUAL(c.to_pin, "input");
    BOOST_CHECK(c.isValid());  // Now valid - modules are set
}

BOOST_AUTO_TEST_CASE(Connection_ParseHandlesBothMissingDot)
{
    auto c = Connection::parse("source", "sink");

    // Neither has dot: module names set, both pins empty (use defaults)
    BOOST_CHECK_EQUAL(c.from_module, "source");
    BOOST_CHECK(c.from_pin.empty());
    BOOST_CHECK_EQUAL(c.to_module, "sink");
    BOOST_CHECK(c.to_pin.empty());
    BOOST_CHECK(c.isValid());  // Valid - modules are set
}

BOOST_AUTO_TEST_CASE(Connection_ParseHandlesComplexPinNames)
{
    auto c = Connection::parse("mux.video_output_0", "sink.video_input");

    BOOST_CHECK_EQUAL(c.from_module, "mux");
    BOOST_CHECK_EQUAL(c.from_pin, "video_output_0");
    BOOST_CHECK_EQUAL(c.to_module, "sink");
    BOOST_CHECK_EQUAL(c.to_pin, "video_input");
}

// ============================================================
// PipelineSettings Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PipelineSettings_DefaultValues)
{
    PipelineSettings s;
    BOOST_CHECK(s.name.empty());
    BOOST_CHECK_EQUAL(s.version, "1.0");
    BOOST_CHECK(s.description.empty());
    BOOST_CHECK_EQUAL(s.queue_size, 10);
    BOOST_CHECK_EQUAL(s.on_error, "restart_module");
    BOOST_CHECK_EQUAL(s.auto_start, false);
}

BOOST_AUTO_TEST_CASE(PipelineSettings_CustomValues)
{
    PipelineSettings s;
    s.name = "my_pipeline";
    s.version = "2.0";
    s.description = "A test pipeline";
    s.queue_size = 20;
    s.on_error = "stop_pipeline";
    s.auto_start = true;

    BOOST_CHECK_EQUAL(s.name, "my_pipeline");
    BOOST_CHECK_EQUAL(s.version, "2.0");
    BOOST_CHECK_EQUAL(s.queue_size, 20);
    BOOST_CHECK_EQUAL(s.auto_start, true);
}

// ============================================================
// PipelineDescription Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PipelineDescription_DefaultConstruction)
{
    PipelineDescription pd;
    BOOST_CHECK(pd.isEmpty());
    BOOST_CHECK(pd.modules.empty());
    BOOST_CHECK(pd.connections.empty());
    BOOST_CHECK(pd.source_format.empty());
    BOOST_CHECK(pd.source_path.empty());
}

BOOST_AUTO_TEST_CASE(PipelineDescription_AddModule)
{
    PipelineDescription pd;

    ModuleInstance m;
    m.instance_id = "source";
    m.module_type = "FileReaderModule";
    pd.addModule(m);

    BOOST_CHECK_EQUAL(pd.modules.size(), 1u);
    BOOST_CHECK(!pd.isEmpty());
}

BOOST_AUTO_TEST_CASE(PipelineDescription_AddConnection)
{
    PipelineDescription pd;

    Connection c;
    c.from_module = "source";
    c.from_pin = "output";
    c.to_module = "decoder";
    c.to_pin = "input";
    pd.addConnection(c);

    BOOST_CHECK_EQUAL(pd.connections.size(), 1u);
}

BOOST_AUTO_TEST_CASE(PipelineDescription_AddConnectionWithStringParsing)
{
    PipelineDescription pd;

    bool success = pd.addConnection("source.output", "decoder.input");
    BOOST_CHECK(success);
    BOOST_CHECK_EQUAL(pd.connections.size(), 1u);
    BOOST_CHECK_EQUAL(pd.connections[0].from_module, "source");
    BOOST_CHECK_EQUAL(pd.connections[0].to_pin, "input");
}

BOOST_AUTO_TEST_CASE(PipelineDescription_AddConnectionSucceedsWithoutPin)
{
    PipelineDescription pd;

    // Missing dot is now valid - uses default pins
    bool success = pd.addConnection("source", "decoder.input");
    BOOST_CHECK(success);
    BOOST_CHECK_EQUAL(pd.connections.size(), 1u);
    BOOST_CHECK_EQUAL(pd.connections[0].from_module, "source");
    BOOST_CHECK(pd.connections[0].from_pin.empty());  // Will use default output pin
}

BOOST_AUTO_TEST_CASE(PipelineDescription_AddConnectionFailsOnEmptyModule)
{
    PipelineDescription pd;

    // Empty module name is invalid
    bool success = pd.addConnection("", "decoder.input");
    BOOST_CHECK(!success);
    BOOST_CHECK(pd.connections.empty());
}

BOOST_AUTO_TEST_CASE(PipelineDescription_FindModule)
{
    PipelineDescription pd;

    ModuleInstance m1, m2;
    m1.instance_id = "source";
    m1.module_type = "FileReaderModule";
    m2.instance_id = "decoder";
    m2.module_type = "H264Decoder";

    pd.addModule(m1);
    pd.addModule(m2);

    auto* found = pd.findModule("decoder");
    BOOST_REQUIRE(found != nullptr);
    BOOST_CHECK_EQUAL(found->module_type, "H264Decoder");

    auto* notFound = pd.findModule("nonexistent");
    BOOST_CHECK(notFound == nullptr);
}

BOOST_AUTO_TEST_CASE(PipelineDescription_SourceTracking)
{
    PipelineDescription pd;
    pd.source_format = "toml";
    pd.source_path = "pipeline.toml";

    BOOST_CHECK_EQUAL(pd.source_format, "toml");
    BOOST_CHECK_EQUAL(pd.source_path, "pipeline.toml");
}

// ============================================================
// Scenario: Create pipeline programmatically
// ============================================================

BOOST_AUTO_TEST_CASE(Scenario_CreatePipelineProgrammatically)
{
    // Given an empty PipelineDescription
    PipelineDescription pd;

    // When I add a ModuleInstance with id="source", type="FileReaderModule"
    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "FileReaderModule";
    pd.addModule(source);

    // And I add a ModuleInstance with id="decoder", type="H264Decoder"
    ModuleInstance decoder;
    decoder.instance_id = "decoder";
    decoder.module_type = "H264Decoder";
    pd.addModule(decoder);

    // And I add a Connection from "source.output" to "decoder.input"
    pd.addConnection("source.output", "decoder.input");

    // Then description.modules.size() == 2
    BOOST_CHECK_EQUAL(pd.modules.size(), 2u);

    // And description.connections.size() == 1
    BOOST_CHECK_EQUAL(pd.connections.size(), 1u);
}

// ============================================================
// Scenario: Property values support multiple types
// ============================================================

BOOST_AUTO_TEST_CASE(Scenario_PropertyValuesMultipleTypes)
{
    // Given a ModuleInstance
    ModuleInstance m;
    m.instance_id = "test";
    m.module_type = "TestModule";

    // When I set properties["port"] = int64_t(8080)
    m.properties["port"] = int64_t(8080);

    // And I set properties["enabled"] = true
    m.properties["enabled"] = true;

    // And I set properties["url"] = std::string("rtsp://...")
    m.properties["url"] = std::string("rtsp://localhost:8554/stream");

    // Then all values are retrievable with correct types via std::get<T>
    BOOST_CHECK_EQUAL(std::get<int64_t>(m.properties["port"]), 8080);
    BOOST_CHECK_EQUAL(std::get<bool>(m.properties["enabled"]), true);
    BOOST_CHECK_EQUAL(std::get<std::string>(m.properties["url"]), "rtsp://localhost:8554/stream");
}

// ============================================================
// Helper function tests
// ============================================================

BOOST_AUTO_TEST_CASE(GetProperty_ReturnsValueIfExists)
{
    std::map<std::string, PropertyValue> props;
    props["count"] = int64_t(42);

    auto result = getProperty<int64_t>(props, "count", 0);
    BOOST_CHECK_EQUAL(result, 42);
}

BOOST_AUTO_TEST_CASE(GetProperty_ReturnsDefaultIfMissing)
{
    std::map<std::string, PropertyValue> props;

    auto result = getProperty<int64_t>(props, "missing", 99);
    BOOST_CHECK_EQUAL(result, 99);
}

BOOST_AUTO_TEST_CASE(GetProperty_ReturnsDefaultOnTypeMismatch)
{
    std::map<std::string, PropertyValue> props;
    props["value"] = std::string("not an int");

    auto result = getProperty<int64_t>(props, "value", 99);
    BOOST_CHECK_EQUAL(result, 99);
}

BOOST_AUTO_TEST_CASE(PropertyValueToString_Int)
{
    PropertyValue v = int64_t(42);
    BOOST_CHECK_EQUAL(propertyValueToString(v), "42");
}

BOOST_AUTO_TEST_CASE(PropertyValueToString_Double)
{
    PropertyValue v = 3.14;
    auto s = propertyValueToString(v);
    BOOST_CHECK(s.find("3.14") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(PropertyValueToString_Bool)
{
    PropertyValue v1 = true;
    PropertyValue v2 = false;
    BOOST_CHECK_EQUAL(propertyValueToString(v1), "true");
    BOOST_CHECK_EQUAL(propertyValueToString(v2), "false");
}

BOOST_AUTO_TEST_CASE(PropertyValueToString_String)
{
    PropertyValue v = std::string("hello");
    BOOST_CHECK_EQUAL(propertyValueToString(v), "\"hello\"");
}

BOOST_AUTO_TEST_CASE(PropertyValueToString_IntVector)
{
    PropertyValue v = std::vector<int64_t>{1, 2, 3};
    BOOST_CHECK_EQUAL(propertyValueToString(v), "[1, 2, 3]");
}

BOOST_AUTO_TEST_CASE(PropertyValueToString_StringVector)
{
    PropertyValue v = std::vector<std::string>{"a", "b"};
    BOOST_CHECK_EQUAL(propertyValueToString(v), "[\"a\", \"b\"]");
}

// ============================================================
// toJson Tests
// ============================================================

BOOST_AUTO_TEST_CASE(ToJson_EmptyPipeline)
{
    PipelineDescription pd;
    pd.source_format = "programmatic";
    pd.source_path = "<inline>";

    std::string json = pd.toJson();
    // Check modules and connections sections exist (arrays may have newlines)
    BOOST_CHECK(json.find("\"modules\":") != std::string::npos);
    BOOST_CHECK(json.find("\"connections\":") != std::string::npos);
    // Verify it's valid JSON structure
    BOOST_CHECK(json.find("\"settings\":") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(ToJson_WithModulesAndConnections)
{
    PipelineDescription pd;
    pd.settings.name = "test_pipeline";
    pd.source_format = "toml";
    pd.source_path = "test.toml";

    ModuleInstance m;
    m.instance_id = "source";
    m.module_type = "FileReaderModule";
    m.properties["path"] = std::string("/video.mp4");
    pd.addModule(m);

    pd.addConnection("source.output", "sink.input");

    std::string json = pd.toJson();

    BOOST_CHECK(json.find("\"name\": \"test_pipeline\"") != std::string::npos);
    BOOST_CHECK(json.find("\"instance_id\": \"source\"") != std::string::npos);
    BOOST_CHECK(json.find("\"module_type\": \"FileReaderModule\"") != std::string::npos);
    BOOST_CHECK(json.find("\"/video.mp4\"") != std::string::npos);
    BOOST_CHECK(json.find("\"from\": \"source.output\"") != std::string::npos);
    BOOST_CHECK(json.find("\"to\": \"sink.input\"") != std::string::npos);
}

BOOST_AUTO_TEST_SUITE_END()
