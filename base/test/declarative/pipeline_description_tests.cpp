#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "PipelineDescription.h"

BOOST_AUTO_TEST_SUITE(pipeline_description_tests)

// ============================================================================
// ModuleInstance Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(ModuleInstance_Construction)
{
    apra::ModuleInstance module;
    module.instance_id = "my_source";
    module.module_type = "FileReaderModule";

    BOOST_CHECK_EQUAL(module.instance_id, "my_source");
    BOOST_CHECK_EQUAL(module.module_type, "FileReaderModule");
    BOOST_CHECK(module.properties.empty());
}

BOOST_AUTO_TEST_CASE(ModuleInstance_WithProperties)
{
    apra::ModuleInstance module;
    module.instance_id = "source";
    module.module_type = "FileReaderModule";
    module.properties["path"] = std::string("/video.mp4");
    module.properties["loop"] = true;
    module.properties["framerate"] = int64_t(30);

    BOOST_CHECK_EQUAL(module.properties.size(), 3);
    BOOST_CHECK_EQUAL(std::get<std::string>(module.properties["path"]), "/video.mp4");
    BOOST_CHECK_EQUAL(std::get<bool>(module.properties["loop"]), true);
    BOOST_CHECK_EQUAL(std::get<int64_t>(module.properties["framerate"]), 30);
}

// ============================================================================
// PropertyValue Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(PropertyValue_Int64)
{
    apra::PropertyValue value = int64_t(8080);
    BOOST_CHECK_EQUAL(std::get<int64_t>(value), 8080);
}

BOOST_AUTO_TEST_CASE(PropertyValue_Double)
{
    apra::PropertyValue value = 3.14159;
    BOOST_CHECK_CLOSE(std::get<double>(value), 3.14159, 0.0001);
}

BOOST_AUTO_TEST_CASE(PropertyValue_Bool)
{
    apra::PropertyValue value = true;
    BOOST_CHECK_EQUAL(std::get<bool>(value), true);

    apra::PropertyValue value2 = false;
    BOOST_CHECK_EQUAL(std::get<bool>(value2), false);
}

BOOST_AUTO_TEST_CASE(PropertyValue_String)
{
    apra::PropertyValue value = std::string("rtsp://example.com/stream");
    BOOST_CHECK_EQUAL(std::get<std::string>(value), "rtsp://example.com/stream");
}

BOOST_AUTO_TEST_CASE(PropertyValue_VectorInt64)
{
    apra::PropertyValue value = std::vector<int64_t>{1, 2, 3, 4, 5};
    auto& vec = std::get<std::vector<int64_t>>(value);
    BOOST_CHECK_EQUAL(vec.size(), 5);
    BOOST_CHECK_EQUAL(vec[0], 1);
    BOOST_CHECK_EQUAL(vec[4], 5);
}

BOOST_AUTO_TEST_CASE(PropertyValue_VectorDouble)
{
    apra::PropertyValue value = std::vector<double>{1.1, 2.2, 3.3};
    auto& vec = std::get<std::vector<double>>(value);
    BOOST_CHECK_EQUAL(vec.size(), 3);
    BOOST_CHECK_CLOSE(vec[0], 1.1, 0.001);
    BOOST_CHECK_CLOSE(vec[2], 3.3, 0.001);
}

BOOST_AUTO_TEST_CASE(PropertyValue_VectorString)
{
    apra::PropertyValue value = std::vector<std::string>{"a", "b", "c"};
    auto& vec = std::get<std::vector<std::string>>(value);
    BOOST_CHECK_EQUAL(vec.size(), 3);
    BOOST_CHECK_EQUAL(vec[0], "a");
    BOOST_CHECK_EQUAL(vec[2], "c");
}

// ============================================================================
// Connection Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(Connection_Construction)
{
    apra::Connection conn;
    conn.from_module = "source";
    conn.from_pin = "output";
    conn.to_module = "decoder";
    conn.to_pin = "input";

    BOOST_CHECK_EQUAL(conn.from_module, "source");
    BOOST_CHECK_EQUAL(conn.from_pin, "output");
    BOOST_CHECK_EQUAL(conn.to_module, "decoder");
    BOOST_CHECK_EQUAL(conn.to_pin, "input");
}

BOOST_AUTO_TEST_CASE(Connection_Parse)
{
    auto conn = apra::Connection::parse("source.output", "decoder.input");

    BOOST_CHECK_EQUAL(conn.from_module, "source");
    BOOST_CHECK_EQUAL(conn.from_pin, "output");
    BOOST_CHECK_EQUAL(conn.to_module, "decoder");
    BOOST_CHECK_EQUAL(conn.to_pin, "input");
}

BOOST_AUTO_TEST_CASE(Connection_Parse_MultipleDots)
{
    // If module name has dots (unusual but possible)
    auto conn = apra::Connection::parse("ns.source.output", "decoder.input");
    BOOST_CHECK_EQUAL(conn.from_module, "ns");
    BOOST_CHECK_EQUAL(conn.from_pin, "source.output");  // Everything after first dot
}

BOOST_AUTO_TEST_CASE(Connection_Parse_Invalid_Throws)
{
    BOOST_CHECK_THROW(
        apra::Connection::parse("invalid_no_dot", "decoder.input"),
        std::invalid_argument
    );

    BOOST_CHECK_THROW(
        apra::Connection::parse("source.output", "invalid_no_dot"),
        std::invalid_argument
    );
}

// ============================================================================
// PipelineSettings Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(PipelineSettings_DefaultValues)
{
    apra::PipelineSettings settings;

    BOOST_CHECK(settings.name.empty());
    BOOST_CHECK_EQUAL(settings.version, "1.0");
    BOOST_CHECK(settings.description.empty());
    BOOST_CHECK_EQUAL(settings.queue_size, 10);
    BOOST_CHECK_EQUAL(settings.on_error, "restart_module");
    BOOST_CHECK_EQUAL(settings.auto_start, false);
}

BOOST_AUTO_TEST_CASE(PipelineSettings_CustomValues)
{
    apra::PipelineSettings settings;
    settings.name = "MyPipeline";
    settings.version = "2.0";
    settings.description = "Test pipeline";
    settings.queue_size = 20;
    settings.on_error = "stop_pipeline";
    settings.auto_start = true;

    BOOST_CHECK_EQUAL(settings.name, "MyPipeline");
    BOOST_CHECK_EQUAL(settings.version, "2.0");
    BOOST_CHECK_EQUAL(settings.description, "Test pipeline");
    BOOST_CHECK_EQUAL(settings.queue_size, 20);
    BOOST_CHECK_EQUAL(settings.on_error, "stop_pipeline");
    BOOST_CHECK_EQUAL(settings.auto_start, true);
}

// ============================================================================
// PipelineDescription Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(PipelineDescription_Empty)
{
    apra::PipelineDescription desc;

    BOOST_CHECK(desc.modules.empty());
    BOOST_CHECK(desc.connections.empty());
    BOOST_CHECK(desc.source_format.empty());
    BOOST_CHECK(desc.source_path.empty());
}

BOOST_AUTO_TEST_CASE(PipelineDescription_AddModulesAndConnections)
{
    apra::PipelineDescription desc;

    // Add source module
    apra::ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "FileReaderModule";
    source.properties["path"] = std::string("/video.mp4");
    desc.modules.push_back(source);

    // Add decoder module
    apra::ModuleInstance decoder;
    decoder.instance_id = "decoder";
    decoder.module_type = "H264Decoder";
    desc.modules.push_back(decoder);

    // Add connection
    desc.connections.push_back(apra::Connection::parse("source.output", "decoder.input"));

    BOOST_CHECK_EQUAL(desc.modules.size(), 2);
    BOOST_CHECK_EQUAL(desc.connections.size(), 1);
}

BOOST_AUTO_TEST_CASE(PipelineDescription_FindModule)
{
    apra::PipelineDescription desc;

    apra::ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "FileReaderModule";
    desc.modules.push_back(source);

    apra::ModuleInstance decoder;
    decoder.instance_id = "decoder";
    decoder.module_type = "H264Decoder";
    desc.modules.push_back(decoder);

    // Find existing modules
    const auto* found_source = desc.findModule("source");
    BOOST_REQUIRE(found_source != nullptr);
    BOOST_CHECK_EQUAL(found_source->module_type, "FileReaderModule");

    const auto* found_decoder = desc.findModule("decoder");
    BOOST_REQUIRE(found_decoder != nullptr);
    BOOST_CHECK_EQUAL(found_decoder->module_type, "H264Decoder");

    // Find non-existent module
    const auto* not_found = desc.findModule("nonexistent");
    BOOST_CHECK(not_found == nullptr);
}

BOOST_AUTO_TEST_CASE(PipelineDescription_SourceTracking)
{
    apra::PipelineDescription desc;
    desc.source_format = "toml";
    desc.source_path = "pipeline.toml";

    BOOST_CHECK_EQUAL(desc.source_format, "toml");
    BOOST_CHECK_EQUAL(desc.source_path, "pipeline.toml");
}

BOOST_AUTO_TEST_CASE(PipelineDescription_ToJson)
{
    apra::PipelineDescription desc;
    desc.settings.name = "TestPipeline";
    desc.settings.version = "1.0";
    desc.source_format = "toml";
    desc.source_path = "test.toml";

    apra::ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "FileReaderModule";
    source.properties["path"] = std::string("/video.mp4");
    source.properties["port"] = int64_t(8080);
    source.properties["enabled"] = true;
    desc.modules.push_back(source);

    desc.connections.push_back(apra::Connection::parse("source.output", "decoder.input"));

    std::string json = desc.toJson();

    // Check that JSON contains expected keys
    BOOST_CHECK(json.find("\"settings\"") != std::string::npos);
    BOOST_CHECK(json.find("\"modules\"") != std::string::npos);
    BOOST_CHECK(json.find("\"connections\"") != std::string::npos);
    BOOST_CHECK(json.find("\"TestPipeline\"") != std::string::npos);
    BOOST_CHECK(json.find("\"source\"") != std::string::npos);
    BOOST_CHECK(json.find("\"FileReaderModule\"") != std::string::npos);
    BOOST_CHECK(json.find("\"/video.mp4\"") != std::string::npos);
    BOOST_CHECK(json.find("8080") != std::string::npos);
    BOOST_CHECK(json.find("true") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(PipelineDescription_ToJson_EmptyPipeline)
{
    apra::PipelineDescription desc;
    std::string json = desc.toJson();

    // Should produce valid JSON even when empty
    BOOST_CHECK(json.find("\"modules\":[]") != std::string::npos);
    BOOST_CHECK(json.find("\"connections\":[]") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(PipelineDescription_ToJson_EscapesSpecialCharacters)
{
    apra::PipelineDescription desc;

    apra::ModuleInstance module;
    module.instance_id = "test";
    module.module_type = "TestModule";
    module.properties["text"] = std::string("Hello\nWorld\t\"quoted\"");
    desc.modules.push_back(module);

    std::string json = desc.toJson();

    // Check escape sequences
    BOOST_CHECK(json.find("\\n") != std::string::npos);
    BOOST_CHECK(json.find("\\t") != std::string::npos);
    BOOST_CHECK(json.find("\\\"") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(PipelineDescription_Scenario_Programmatic)
{
    // Scenario: Create pipeline description programmatically
    // Given an empty PipelineDescription
    apra::PipelineDescription desc;

    // When I add a ModuleInstance with id="source", type="FileReaderModule"
    apra::ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "FileReaderModule";
    desc.modules.push_back(source);

    // And I add a ModuleInstance with id="decoder", type="H264Decoder"
    apra::ModuleInstance decoder;
    decoder.instance_id = "decoder";
    decoder.module_type = "H264Decoder";
    desc.modules.push_back(decoder);

    // And I add a Connection from "source.output" to "decoder.input"
    desc.connections.push_back(apra::Connection::parse("source.output", "decoder.input"));

    // Then description.modules.size() == 2
    BOOST_CHECK_EQUAL(desc.modules.size(), 2);
    // And description.connections.size() == 1
    BOOST_CHECK_EQUAL(desc.connections.size(), 1);
}

BOOST_AUTO_TEST_CASE(PipelineDescription_Scenario_MultipleTypes)
{
    // Scenario: Property values support multiple types
    // Given a ModuleInstance
    apra::ModuleInstance module;
    module.instance_id = "test";
    module.module_type = "TestModule";

    // When I set properties["port"] = int64_t(8080)
    module.properties["port"] = int64_t(8080);
    // And I set properties["enabled"] = true
    module.properties["enabled"] = true;
    // And I set properties["url"] = std::string("rtsp://...")
    module.properties["url"] = std::string("rtsp://example.com/stream");

    // Then all values are retrievable with correct types via std::get<T>
    BOOST_CHECK_EQUAL(std::get<int64_t>(module.properties["port"]), 8080);
    BOOST_CHECK_EQUAL(std::get<bool>(module.properties["enabled"]), true);
    BOOST_CHECK_EQUAL(std::get<std::string>(module.properties["url"]), "rtsp://example.com/stream");
}

BOOST_AUTO_TEST_CASE(PipelineDescription_Scenario_SourceTracking)
{
    // Scenario: Track source format for error messages
    // Given a PipelineDescription parsed from "pipeline.toml"
    apra::PipelineDescription desc;

    // When parsing completes
    desc.source_format = "toml";
    desc.source_path = "pipeline.toml";

    // Then source_format == "toml"
    BOOST_CHECK_EQUAL(desc.source_format, "toml");
    // And source_path == "pipeline.toml"
    BOOST_CHECK_EQUAL(desc.source_path, "pipeline.toml");
}

BOOST_AUTO_TEST_SUITE_END()
