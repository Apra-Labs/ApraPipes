#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "declarative/TomlParser.h"
#include <fstream>
#include <cstdlib>

// Helper to get the test data directory path
namespace {
std::string getTestDataPath(const std::string& filename) {
    // Try relative path from build directory
    std::string paths[] = {
        "../base/test/data/pipelines/" + filename,
        "base/test/data/pipelines/" + filename,
        "test/data/pipelines/" + filename,
    };

    for (const auto& path : paths) {
        std::ifstream f(path);
        if (f.good()) {
            return path;
        }
    }

    // Fallback
    return "../base/test/data/pipelines/" + filename;
}
}

BOOST_AUTO_TEST_SUITE(toml_parser_tests)

// ============================================================================
// Basic Parsing Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(ParseMinimalPipeline)
{
    apra::TomlParser parser;
    auto result = parser.parseFile(getTestDataPath("minimal.toml"));

    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.modules.size(), 1);
    BOOST_CHECK_EQUAL(result.description.connections.size(), 0);
    BOOST_CHECK_EQUAL(result.description.modules[0].instance_id, "source");
    BOOST_CHECK_EQUAL(result.description.modules[0].module_type, "FileReaderModule");
}

BOOST_AUTO_TEST_CASE(ParseCompletePipeline)
{
    apra::TomlParser parser;
    auto result = parser.parseFile(getTestDataPath("complete.toml"));

    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.modules.size(), 3);
    BOOST_CHECK_EQUAL(result.description.connections.size(), 2);

    // Check settings
    BOOST_CHECK_EQUAL(result.description.settings.name, "TestPipeline");
    BOOST_CHECK_EQUAL(result.description.settings.version, "1.0");
    BOOST_CHECK_EQUAL(result.description.settings.description, "A complete test pipeline");
    BOOST_CHECK_EQUAL(result.description.settings.queue_size, 20);
    BOOST_CHECK_EQUAL(result.description.settings.on_error, "stop_pipeline");
    BOOST_CHECK_EQUAL(result.description.settings.auto_start, true);
}

BOOST_AUTO_TEST_CASE(ParseAllPropertyTypes)
{
    apra::TomlParser parser;
    auto result = parser.parseFile(getTestDataPath("all_property_types.toml"));

    BOOST_CHECK(result.success);
    BOOST_REQUIRE_EQUAL(result.description.modules.size(), 1);

    const auto& props = result.description.modules[0].properties;

    // Integer
    BOOST_CHECK_EQUAL(std::get<int64_t>(props.at("int_val")), 42);
    BOOST_CHECK_EQUAL(std::get<int64_t>(props.at("negative_int")), -100);

    // Float
    BOOST_CHECK_CLOSE(std::get<double>(props.at("float_val")), 3.14159, 0.00001);

    // Boolean
    BOOST_CHECK_EQUAL(std::get<bool>(props.at("bool_true")), true);
    BOOST_CHECK_EQUAL(std::get<bool>(props.at("bool_false")), false);

    // String
    BOOST_CHECK_EQUAL(std::get<std::string>(props.at("string_val")), "hello world");

    // Integer array
    auto& int_arr = std::get<std::vector<int64_t>>(props.at("int_array"));
    BOOST_CHECK_EQUAL(int_arr.size(), 5);
    BOOST_CHECK_EQUAL(int_arr[0], 1);
    BOOST_CHECK_EQUAL(int_arr[4], 5);

    // Float array
    auto& float_arr = std::get<std::vector<double>>(props.at("float_array"));
    BOOST_CHECK_EQUAL(float_arr.size(), 3);
    BOOST_CHECK_CLOSE(float_arr[0], 1.1, 0.001);

    // String array
    auto& str_arr = std::get<std::vector<std::string>>(props.at("string_array"));
    BOOST_CHECK_EQUAL(str_arr.size(), 3);
    BOOST_CHECK_EQUAL(str_arr[0], "a");
    BOOST_CHECK_EQUAL(str_arr[2], "c");

    // Empty array (defaults to int64_t vector)
    auto& empty_arr = std::get<std::vector<int64_t>>(props.at("empty_array"));
    BOOST_CHECK_EQUAL(empty_arr.size(), 0);
}

// ============================================================================
// Connection Parsing Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(ParseConnections)
{
    apra::TomlParser parser;
    auto result = parser.parseFile(getTestDataPath("complete.toml"));

    BOOST_CHECK(result.success);
    BOOST_REQUIRE_EQUAL(result.description.connections.size(), 2);

    // First connection
    BOOST_CHECK_EQUAL(result.description.connections[0].from_module, "source");
    BOOST_CHECK_EQUAL(result.description.connections[0].from_pin, "output");
    BOOST_CHECK_EQUAL(result.description.connections[0].to_module, "decoder");
    BOOST_CHECK_EQUAL(result.description.connections[0].to_pin, "input");

    // Second connection
    BOOST_CHECK_EQUAL(result.description.connections[1].from_module, "decoder");
    BOOST_CHECK_EQUAL(result.description.connections[1].from_pin, "output");
    BOOST_CHECK_EQUAL(result.description.connections[1].to_module, "writer");
    BOOST_CHECK_EQUAL(result.description.connections[1].to_pin, "input");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(SyntaxErrorReporting)
{
    apra::TomlParser parser;
    auto result = parser.parseFile(getTestDataPath("syntax_error.toml"));

    BOOST_CHECK(!result.success);
    BOOST_CHECK(!result.error.empty());
    BOOST_CHECK(result.error_line > 0);  // Should report line number
}

BOOST_AUTO_TEST_CASE(MissingTypeField)
{
    apra::TomlParser parser;
    auto result = parser.parseFile(getTestDataPath("missing_type.toml"));

    BOOST_CHECK(!result.success);
    BOOST_CHECK(result.error.find("type") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(FileNotFound)
{
    apra::TomlParser parser;
    auto result = parser.parseFile("nonexistent_file.toml");

    BOOST_CHECK(!result.success);
    BOOST_CHECK(!result.error.empty());
}

BOOST_AUTO_TEST_CASE(EmptyFile)
{
    apra::TomlParser parser;
    // Parse empty TOML string
    auto result = parser.parseString("");

    // Empty file should be valid (just an empty pipeline)
    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.modules.size(), 0);
    BOOST_CHECK_EQUAL(result.description.connections.size(), 0);
}

// ============================================================================
// String Parsing Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(ParseFromString)
{
    apra::TomlParser parser;

    std::string toml = R"(
[pipeline]
name = "InlineTest"

[modules.source]
type = "FileReaderModule"
    [modules.source.props]
    path = "/video.mp4"

[[connections]]
from = "source.output"
to = "decoder.input"
)";

    auto result = parser.parseString(toml, "inline_test");

    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.settings.name, "InlineTest");
    BOOST_CHECK_EQUAL(result.description.modules.size(), 1);
    BOOST_CHECK_EQUAL(result.description.connections.size(), 1);
    BOOST_CHECK_EQUAL(result.description.source_path, "inline_test");
}

BOOST_AUTO_TEST_CASE(ParseStringWithSyntaxError)
{
    apra::TomlParser parser;

    std::string toml = R"(
[modules.broken]
type = "unclosed
)";

    auto result = parser.parseString(toml);

    BOOST_CHECK(!result.success);
    BOOST_CHECK(!result.error.empty());
    BOOST_CHECK(result.error_line > 0);
}

// ============================================================================
// Source Tracking Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(SourceTrackingFromFile)
{
    apra::TomlParser parser;
    auto result = parser.parseFile(getTestDataPath("minimal.toml"));

    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.source_format, "toml");
    BOOST_CHECK(result.description.source_path.find("minimal.toml") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(SourceTrackingFromString)
{
    apra::TomlParser parser;
    auto result = parser.parseString("[modules.test]\ntype = \"Test\"", "custom_source");

    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.source_format, "toml");
    BOOST_CHECK_EQUAL(result.description.source_path, "custom_source");
}

// ============================================================================
// Parser Interface Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(FormatName)
{
    apra::TomlParser parser;
    BOOST_CHECK_EQUAL(parser.formatName(), "toml");
}

BOOST_AUTO_TEST_CASE(FileExtensions)
{
    apra::TomlParser parser;
    auto extensions = parser.fileExtensions();
    BOOST_REQUIRE_EQUAL(extensions.size(), 1);
    BOOST_CHECK_EQUAL(extensions[0], ".toml");
}

// ============================================================================
// Edge Case Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(ModuleWithNoProps)
{
    apra::TomlParser parser;
    auto result = parser.parseString(R"(
[modules.simple]
type = "SimpleModule"
)");

    BOOST_CHECK(result.success);
    BOOST_REQUIRE_EQUAL(result.description.modules.size(), 1);
    BOOST_CHECK_EQUAL(result.description.modules[0].properties.size(), 0);
}

BOOST_AUTO_TEST_CASE(PipelineWithOnlySettings)
{
    apra::TomlParser parser;
    auto result = parser.parseString(R"(
[pipeline]
name = "EmptyPipeline"
version = "2.0"
)");

    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.settings.name, "EmptyPipeline");
    BOOST_CHECK_EQUAL(result.description.settings.version, "2.0");
    BOOST_CHECK_EQUAL(result.description.modules.size(), 0);
}

BOOST_AUTO_TEST_CASE(MultilineStrings)
{
    apra::TomlParser parser;
    auto result = parser.parseString(R"(
[modules.test]
type = "TestModule"
    [modules.test.props]
    description = """
This is a
multiline string
"""
)");

    BOOST_CHECK(result.success);
    auto& desc = std::get<std::string>(result.description.modules[0].properties.at("description"));
    BOOST_CHECK(desc.find("\n") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(DefaultSettings)
{
    apra::TomlParser parser;
    auto result = parser.parseString(R"(
[modules.test]
type = "TestModule"
)");

    BOOST_CHECK(result.success);
    // Should have default values
    BOOST_CHECK_EQUAL(result.description.settings.version, "1.0");
    BOOST_CHECK_EQUAL(result.description.settings.queue_size, 10);
    BOOST_CHECK_EQUAL(result.description.settings.on_error, "restart_module");
    BOOST_CHECK_EQUAL(result.description.settings.auto_start, false);
}

BOOST_AUTO_TEST_SUITE_END()
