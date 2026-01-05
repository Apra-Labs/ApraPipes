// ============================================================
// File: json_parser_tests.cpp
// Unit tests for JSON pipeline parser
// Task J2: JSON Parser Tests
// ============================================================

#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "declarative/JsonParser.h"
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

BOOST_AUTO_TEST_SUITE(json_parser_tests)

// ============================================================================
// Basic Parsing Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(ParseMinimalPipeline)
{
    auto result = apra::JsonParser::parseFile(getTestDataPath("minimal.json"));

    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.modules.size(), 1);
    BOOST_CHECK_EQUAL(result.description.connections.size(), 0);
    BOOST_CHECK_EQUAL(result.description.modules[0].instance_id, "source");
    BOOST_CHECK_EQUAL(result.description.modules[0].module_type, "FileReaderModule");
}

BOOST_AUTO_TEST_CASE(ParseCompletePipeline)
{
    auto result = apra::JsonParser::parseFile(getTestDataPath("complete.json"));

    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.modules.size(), 3);
    BOOST_CHECK_EQUAL(result.description.connections.size(), 2);

    // Check settings
    BOOST_CHECK_EQUAL(result.description.settings.name, "TestPipeline");
    BOOST_CHECK_EQUAL(result.description.settings.version, "1.0");
    BOOST_CHECK_EQUAL(result.description.settings.description, "A complete test pipeline with multiple modules and connections");
    BOOST_CHECK_EQUAL(result.description.settings.queue_size, 20);
    BOOST_CHECK_EQUAL(result.description.settings.on_error, "stop_pipeline");
    BOOST_CHECK_EQUAL(result.description.settings.auto_start, true);
}

BOOST_AUTO_TEST_CASE(ParseAllPropertyTypes)
{
    auto result = apra::JsonParser::parseFile(getTestDataPath("all_property_types.json"));

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
    auto result = apra::JsonParser::parseFile(getTestDataPath("complete.json"));

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
    auto result = apra::JsonParser::parseFile(getTestDataPath("syntax_error.json"));

    BOOST_CHECK(!result.success);
    BOOST_CHECK(!result.error.empty());
}

BOOST_AUTO_TEST_CASE(MissingTypeField)
{
    auto result = apra::JsonParser::parseFile(getTestDataPath("missing_type.json"));

    BOOST_CHECK(!result.success);
    BOOST_CHECK(result.error.find("type") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(FileNotFound)
{
    auto result = apra::JsonParser::parseFile("nonexistent_file.json");

    BOOST_CHECK(!result.success);
    BOOST_CHECK(!result.error.empty());
}

BOOST_AUTO_TEST_CASE(EmptyFile)
{
    // Parse empty JSON object
    auto result = apra::JsonParser::parseString("{}");

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
    std::string json = R"({
        "pipeline": {
            "name": "InlineTest"
        },
        "modules": {
            "source": {
                "type": "FileReaderModule",
                "props": {
                    "path": "/video.mp4"
                }
            }
        },
        "connections": [
            { "from": "source.output", "to": "decoder.input" }
        ]
    })";

    auto result = apra::JsonParser::parseString(json, "inline_test");

    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.settings.name, "InlineTest");
    BOOST_CHECK_EQUAL(result.description.modules.size(), 1);
    BOOST_CHECK_EQUAL(result.description.connections.size(), 1);
    BOOST_CHECK_EQUAL(result.description.source_path, "inline_test");
}

BOOST_AUTO_TEST_CASE(ParseStringWithSyntaxError)
{
    std::string json = R"({
        "modules": {
            "broken": {
                "type": "unclosed
            }
        }
    })";

    auto result = apra::JsonParser::parseString(json);

    BOOST_CHECK(!result.success);
    BOOST_CHECK(!result.error.empty());
}

// ============================================================================
// Source Tracking Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(SourceTrackingFromFile)
{
    auto result = apra::JsonParser::parseFile(getTestDataPath("minimal.json"));

    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.source_format, "json");
    BOOST_CHECK(result.description.source_path.find("minimal.json") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(SourceTrackingFromString)
{
    auto result = apra::JsonParser::parseString(R"({"modules": {"test": {"type": "Test"}}})", "custom_source");

    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.source_format, "json");
    BOOST_CHECK_EQUAL(result.description.source_path, "custom_source");
}

// ============================================================================
// Parser Interface Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(FormatName)
{
    BOOST_CHECK_EQUAL(apra::JsonParser::formatName(), "json");
}

BOOST_AUTO_TEST_CASE(FileExtensions)
{
    auto extensions = apra::JsonParser::fileExtensions();
    BOOST_REQUIRE_EQUAL(extensions.size(), 1);
    BOOST_CHECK_EQUAL(extensions[0], ".json");
}

// ============================================================================
// Edge Case Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(ModuleWithNoProps)
{
    auto result = apra::JsonParser::parseString(R"({
        "modules": {
            "simple": {
                "type": "SimpleModule"
            }
        }
    })");

    BOOST_CHECK(result.success);
    BOOST_REQUIRE_EQUAL(result.description.modules.size(), 1);
    BOOST_CHECK_EQUAL(result.description.modules[0].properties.size(), 0);
}

BOOST_AUTO_TEST_CASE(PipelineWithOnlySettings)
{
    auto result = apra::JsonParser::parseString(R"({
        "pipeline": {
            "name": "EmptyPipeline",
            "version": "2.0"
        }
    })");

    BOOST_CHECK(result.success);
    BOOST_CHECK_EQUAL(result.description.settings.name, "EmptyPipeline");
    BOOST_CHECK_EQUAL(result.description.settings.version, "2.0");
    BOOST_CHECK_EQUAL(result.description.modules.size(), 0);
}

BOOST_AUTO_TEST_CASE(DefaultSettings)
{
    auto result = apra::JsonParser::parseString(R"({
        "modules": {
            "test": {
                "type": "TestModule"
            }
        }
    })");

    BOOST_CHECK(result.success);
    // Should have default values
    BOOST_CHECK_EQUAL(result.description.settings.version, "1.0");
    BOOST_CHECK_EQUAL(result.description.settings.queue_size, 10);
    BOOST_CHECK_EQUAL(result.description.settings.on_error, "restart_module");
    BOOST_CHECK_EQUAL(result.description.settings.auto_start, false);
}

// ============================================================================
// JSON-Specific Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(CommentFieldsIgnored)
{
    // Fields starting with # are treated as comments and ignored in props
    auto result = apra::JsonParser::parseString(R"({
        "modules": {
            "test": {
                "type": "TestModule",
                "props": {
                    "#comment": "This is a comment field",
                    "value": 42
                }
            }
        }
    })");

    BOOST_CHECK(result.success);
    BOOST_REQUIRE_EQUAL(result.description.modules.size(), 1);
    // Comment field should not be in properties
    BOOST_CHECK(result.description.modules[0].properties.find("#comment") ==
                result.description.modules[0].properties.end());
    // Regular property should be present
    BOOST_CHECK_EQUAL(std::get<int64_t>(result.description.modules[0].properties.at("value")), 42);
}

BOOST_AUTO_TEST_CASE(CamelCaseAndSnakeCaseSettings)
{
    // Test camelCase
    auto result1 = apra::JsonParser::parseString(R"({
        "settings": {
            "queueSize": 30,
            "onError": "stop",
            "autoStart": true
        },
        "modules": {}
    })");

    BOOST_CHECK(result1.success);
    BOOST_CHECK_EQUAL(result1.description.settings.queue_size, 30);
    BOOST_CHECK_EQUAL(result1.description.settings.on_error, "stop");
    BOOST_CHECK_EQUAL(result1.description.settings.auto_start, true);

    // Test snake_case
    auto result2 = apra::JsonParser::parseString(R"({
        "settings": {
            "queue_size": 40,
            "on_error": "skip",
            "auto_start": false
        },
        "modules": {}
    })");

    BOOST_CHECK(result2.success);
    BOOST_CHECK_EQUAL(result2.description.settings.queue_size, 40);
    BOOST_CHECK_EQUAL(result2.description.settings.on_error, "skip");
    BOOST_CHECK_EQUAL(result2.description.settings.auto_start, false);
}

BOOST_AUTO_TEST_CASE(InvalidModulesNotObject)
{
    auto result = apra::JsonParser::parseString(R"({
        "modules": "not_an_object"
    })");

    BOOST_CHECK(!result.success);
    BOOST_CHECK(result.error.find("object") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(InvalidConnectionsNotArray)
{
    auto result = apra::JsonParser::parseString(R"({
        "modules": {},
        "connections": "not_an_array"
    })");

    BOOST_CHECK(!result.success);
    BOOST_CHECK(result.error.find("array") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(ConnectionMissingFrom)
{
    auto result = apra::JsonParser::parseString(R"({
        "modules": {},
        "connections": [
            { "to": "sink.input" }
        ]
    })");

    BOOST_CHECK(!result.success);
    BOOST_CHECK(result.error.find("from") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(ConnectionMissingTo)
{
    auto result = apra::JsonParser::parseString(R"({
        "modules": {},
        "connections": [
            { "from": "source.output" }
        ]
    })");

    BOOST_CHECK(!result.success);
    BOOST_CHECK(result.error.find("to") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(MixedTypeArrayRejected)
{
    auto result = apra::JsonParser::parseString(R"({
        "modules": {
            "test": {
                "type": "TestModule",
                "props": {
                    "mixed": [1, "two", 3]
                }
            }
        }
    })");

    BOOST_CHECK(!result.success);
    BOOST_CHECK(result.error.find("Mixed") != std::string::npos ||
                result.error.find("type") != std::string::npos);
}

BOOST_AUTO_TEST_SUITE_END()
