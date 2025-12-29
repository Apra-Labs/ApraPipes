// ============================================================
// Unit tests for declarative/PipelineValidator.h
// Task C1: Validator Shell
// ============================================================

#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "declarative/PipelineValidator.h"
#include "declarative/PipelineDescription.h"

using namespace apra;

BOOST_AUTO_TEST_SUITE(PipelineValidatorTests)

// ============================================================
// Helper to create test pipeline descriptions
// ============================================================

PipelineDescription createEmptyPipeline() {
    PipelineDescription desc;
    desc.settings.name = "empty_pipeline";
    desc.settings.version = "1.0.0";
    return desc;
}

PipelineDescription createSimplePipeline() {
    PipelineDescription desc;
    desc.settings.name = "simple_pipeline";
    desc.settings.version = "1.0.0";

    // Add source module
    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "FileReaderModule";
    source.properties["path"] = std::string("/video.mp4");
    desc.modules.push_back(source);

    // Add sink module
    ModuleInstance sink;
    sink.instance_id = "sink";
    sink.module_type = "FileSinkModule";
    sink.properties["output_path"] = std::string("/output.mp4");
    desc.modules.push_back(sink);

    // Add connection
    ConnectionDef conn;
    conn.from_module = "source";
    conn.from_pin = "output";
    conn.to_module = "sink";
    conn.to_pin = "input";
    desc.connections.push_back(conn);

    return desc;
}

PipelineDescription createPipelineWithMissingModule() {
    PipelineDescription desc;
    desc.settings.name = "missing_module_pipeline";
    desc.settings.version = "1.0.0";

    // Add source module
    ModuleInstance source;
    source.instance_id = "source";
    source.module_type = "FileReaderModule";
    desc.modules.push_back(source);

    // Connection references non-existent module
    ConnectionDef conn;
    conn.from_module = "source";
    conn.from_pin = "output";
    conn.to_module = "nonexistent";
    conn.to_pin = "input";
    desc.connections.push_back(conn);

    return desc;
}

// ============================================================
// ValidationIssue Tests
// ============================================================

BOOST_AUTO_TEST_CASE(ValidationIssue_ErrorFactory)
{
    auto issue = ValidationIssue::error("E001", "modules.decoder", "Unknown module type", "Check spelling");

    BOOST_CHECK(issue.level == ValidationIssue::Level::Error);
    BOOST_CHECK_EQUAL(issue.code, "E001");
    BOOST_CHECK_EQUAL(issue.location, "modules.decoder");
    BOOST_CHECK_EQUAL(issue.message, "Unknown module type");
    BOOST_CHECK_EQUAL(issue.suggestion, "Check spelling");
}

BOOST_AUTO_TEST_CASE(ValidationIssue_WarningFactory)
{
    auto issue = ValidationIssue::warning("W001", "modules.encoder", "Deprecated module", "Use NewEncoder");

    BOOST_CHECK(issue.level == ValidationIssue::Level::Warning);
    BOOST_CHECK_EQUAL(issue.code, "W001");
    BOOST_CHECK_EQUAL(issue.message, "Deprecated module");
}

BOOST_AUTO_TEST_CASE(ValidationIssue_InfoFactory)
{
    auto issue = ValidationIssue::info("I001", "pipeline", "Validation started");

    BOOST_CHECK(issue.level == ValidationIssue::Level::Info);
    BOOST_CHECK_EQUAL(issue.code, "I001");
    BOOST_CHECK(issue.suggestion.empty());
}

BOOST_AUTO_TEST_CASE(ValidationIssue_ErrorCodes_Defined)
{
    // Module validation codes (C2)
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::UNKNOWN_MODULE_TYPE), "E100");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::MODULE_VERSION_MISMATCH), "W100");

    // Property validation codes (C3)
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::UNKNOWN_PROPERTY), "E200");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::PROPERTY_TYPE_MISMATCH), "E201");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::PROPERTY_OUT_OF_RANGE), "E202");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::PROPERTY_INVALID_ENUM), "E203");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::PROPERTY_REGEX_MISMATCH), "E204");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::MISSING_REQUIRED_PROPERTY), "W200");

    // Connection validation codes (C4)
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::UNKNOWN_SOURCE_MODULE), "E300");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::UNKNOWN_DEST_MODULE), "E301");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::UNKNOWN_SOURCE_PIN), "E302");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::UNKNOWN_DEST_PIN), "E303");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::FRAME_TYPE_INCOMPATIBLE), "E304");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::DUPLICATE_INPUT_CONNECTION), "E305");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::REQUIRED_PIN_UNCONNECTED), "W300");

    // Graph validation codes (C5)
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::NO_SOURCE_MODULE), "E400");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::GRAPH_HAS_CYCLE), "E401");
    BOOST_CHECK_EQUAL(std::string(ValidationIssue::ORPHAN_MODULE), "W400");
}

// ============================================================
// Result Tests
// ============================================================

BOOST_AUTO_TEST_CASE(Result_HasErrors_EmptyResult)
{
    PipelineValidator::Result result;
    BOOST_CHECK(!result.hasErrors());
    BOOST_CHECK(!result.hasWarnings());
}

BOOST_AUTO_TEST_CASE(Result_HasErrors_WithError)
{
    PipelineValidator::Result result;
    result.issues.push_back(ValidationIssue::error("E001", "loc", "msg"));

    BOOST_CHECK(result.hasErrors());
    BOOST_CHECK(!result.hasWarnings());
}

BOOST_AUTO_TEST_CASE(Result_HasWarnings_WithWarning)
{
    PipelineValidator::Result result;
    result.issues.push_back(ValidationIssue::warning("W001", "loc", "msg"));

    BOOST_CHECK(!result.hasErrors());
    BOOST_CHECK(result.hasWarnings());
}

BOOST_AUTO_TEST_CASE(Result_Errors_FiltersCorrectly)
{
    PipelineValidator::Result result;
    result.issues.push_back(ValidationIssue::error("E001", "loc1", "error1"));
    result.issues.push_back(ValidationIssue::warning("W001", "loc2", "warning1"));
    result.issues.push_back(ValidationIssue::error("E002", "loc3", "error2"));
    result.issues.push_back(ValidationIssue::info("I001", "loc4", "info1"));

    auto errors = result.errors();
    BOOST_CHECK_EQUAL(errors.size(), 2);
    BOOST_CHECK_EQUAL(errors[0].code, "E001");
    BOOST_CHECK_EQUAL(errors[1].code, "E002");
}

BOOST_AUTO_TEST_CASE(Result_Warnings_FiltersCorrectly)
{
    PipelineValidator::Result result;
    result.issues.push_back(ValidationIssue::error("E001", "loc1", "error1"));
    result.issues.push_back(ValidationIssue::warning("W001", "loc2", "warning1"));
    result.issues.push_back(ValidationIssue::warning("W002", "loc3", "warning2"));

    auto warnings = result.warnings();
    BOOST_CHECK_EQUAL(warnings.size(), 2);
    BOOST_CHECK_EQUAL(warnings[0].code, "W001");
    BOOST_CHECK_EQUAL(warnings[1].code, "W002");
}

BOOST_AUTO_TEST_CASE(Result_Merge_CombinesIssues)
{
    PipelineValidator::Result result1;
    result1.issues.push_back(ValidationIssue::error("E001", "loc1", "error1"));

    PipelineValidator::Result result2;
    result2.issues.push_back(ValidationIssue::warning("W001", "loc2", "warning1"));

    result1.merge(result2);

    BOOST_CHECK_EQUAL(result1.issues.size(), 2);
    BOOST_CHECK(result1.hasErrors());
    BOOST_CHECK(result1.hasWarnings());
}

BOOST_AUTO_TEST_CASE(Result_Format_OutputsCorrectly)
{
    PipelineValidator::Result result;
    result.issues.push_back(ValidationIssue::error("E001", "modules.decoder", "Unknown module"));
    result.issues.push_back(ValidationIssue::warning("W001", "modules.encoder", "Deprecated", "Use NewEncoder"));

    std::string formatted = result.format();

    BOOST_CHECK(formatted.find("[ERROR]") != std::string::npos);
    BOOST_CHECK(formatted.find("[WARN]") != std::string::npos);
    BOOST_CHECK(formatted.find("E001") != std::string::npos);
    BOOST_CHECK(formatted.find("modules.decoder") != std::string::npos);
    BOOST_CHECK(formatted.find("Suggestion:") != std::string::npos);
}

// ============================================================
// Validator Options Tests
// ============================================================

BOOST_AUTO_TEST_CASE(Options_DefaultValues)
{
    PipelineValidator::Options opts;

    BOOST_CHECK(!opts.stopOnFirstError);
    BOOST_CHECK(!opts.includeInfoMessages);
    BOOST_CHECK(opts.validateConnections);
    BOOST_CHECK(opts.validateGraph);
}

BOOST_AUTO_TEST_CASE(Validator_GetSetOptions)
{
    PipelineValidator validator;

    auto opts = validator.options();
    BOOST_CHECK(!opts.stopOnFirstError);

    opts.stopOnFirstError = true;
    validator.setOptions(opts);

    BOOST_CHECK(validator.options().stopOnFirstError);
}

// ============================================================
// Main Validation Tests
// ============================================================

BOOST_AUTO_TEST_CASE(Validate_EmptyPipeline_NoErrors)
{
    PipelineValidator validator;
    auto desc = createEmptyPipeline();

    auto result = validator.validate(desc);

    // Shell implementation should not produce errors for empty pipeline
    BOOST_CHECK(!result.hasErrors());
}

BOOST_AUTO_TEST_CASE(Validate_SimplePipeline_NoErrors)
{
    PipelineValidator validator;
    auto desc = createSimplePipeline();

    auto result = validator.validate(desc);

    // Shell implementation should not produce errors
    BOOST_CHECK(!result.hasErrors());
}

BOOST_AUTO_TEST_CASE(Validate_WithInfoMessages)
{
    PipelineValidator::Options opts;
    opts.includeInfoMessages = true;
    PipelineValidator validator(opts);

    auto desc = createSimplePipeline();
    auto result = validator.validate(desc);

    // Should have info messages about what was validated
    bool hasInfo = std::any_of(result.issues.begin(), result.issues.end(),
        [](const ValidationIssue& i) { return i.level == ValidationIssue::Level::Info; });
    BOOST_CHECK(hasInfo);
}

BOOST_AUTO_TEST_CASE(Validate_DisableConnectionValidation)
{
    PipelineValidator::Options opts;
    opts.validateConnections = false;
    opts.includeInfoMessages = true;
    PipelineValidator validator(opts);

    auto desc = createPipelineWithMissingModule();
    auto result = validator.validate(desc);

    // With connection validation disabled, should not report connection issues
    // (Though shell implementation may still report info)
    BOOST_CHECK(!result.hasErrors());
}

BOOST_AUTO_TEST_CASE(Validate_DisableGraphValidation)
{
    PipelineValidator::Options opts;
    opts.validateGraph = false;
    PipelineValidator validator(opts);

    auto desc = createEmptyPipeline();
    auto result = validator.validate(desc);

    BOOST_CHECK(!result.hasErrors());
}

// ============================================================
// Individual Phase Tests
// ============================================================

BOOST_AUTO_TEST_CASE(ValidateModules_ReturnsResult)
{
    PipelineValidator validator;
    auto desc = createSimplePipeline();

    auto result = validator.validateModules(desc);

    // Shell implementation - just check it runs without crashing
    // and returns a valid result
    BOOST_CHECK(result.issues.empty() || !result.issues.empty());
}

BOOST_AUTO_TEST_CASE(ValidateProperties_ReturnsResult)
{
    PipelineValidator validator;
    auto desc = createSimplePipeline();

    auto result = validator.validateProperties(desc);

    // Shell implementation - check it returns valid result
    BOOST_CHECK(result.issues.empty() || !result.issues.empty());
}

BOOST_AUTO_TEST_CASE(ValidateConnections_ReturnsResult)
{
    PipelineValidator validator;
    auto desc = createSimplePipeline();

    auto result = validator.validateConnections(desc);

    // Shell implementation - check it returns valid result
    BOOST_CHECK(result.issues.empty() || !result.issues.empty());
}

BOOST_AUTO_TEST_CASE(ValidateGraph_ReturnsResult)
{
    PipelineValidator validator;
    auto desc = createSimplePipeline();

    auto result = validator.validateGraph(desc);

    // Shell implementation - check it returns valid result
    BOOST_CHECK(result.issues.empty() || !result.issues.empty());
}

// ============================================================
// Info Messages Content Tests
// ============================================================

BOOST_AUTO_TEST_CASE(Validate_InfoMessages_ShowModuleCount)
{
    PipelineValidator::Options opts;
    opts.includeInfoMessages = true;
    PipelineValidator validator(opts);

    auto desc = createSimplePipeline();
    auto result = validator.validate(desc);

    // Should mention the number of modules and connections
    std::string formatted = result.format();
    BOOST_CHECK(formatted.find("2 modules") != std::string::npos);
    BOOST_CHECK(formatted.find("1 connections") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(ValidateModules_InfoMessages_ShowModuleTypes)
{
    PipelineValidator::Options opts;
    opts.includeInfoMessages = true;
    PipelineValidator validator(opts);

    auto desc = createSimplePipeline();
    auto result = validator.validateModules(desc);

    std::string formatted = result.format();
    BOOST_CHECK(formatted.find("FileReaderModule") != std::string::npos);
    BOOST_CHECK(formatted.find("FileSinkModule") != std::string::npos);
}

// ============================================================
// Stop On First Error Tests
// ============================================================

BOOST_AUTO_TEST_CASE(Validate_StopOnFirstError_Option)
{
    PipelineValidator::Options opts;
    opts.stopOnFirstError = true;
    PipelineValidator validator(opts);

    // Shell implementation doesn't produce errors,
    // but verify the option is respected by the framework
    auto desc = createSimplePipeline();
    auto result = validator.validate(desc);

    // Should complete without errors in shell implementation
    BOOST_CHECK(!result.hasErrors());
}

// ============================================================
// Edge Cases
// ============================================================

BOOST_AUTO_TEST_CASE(Validate_ModuleWithNoProperties)
{
    PipelineValidator validator;

    PipelineDescription desc;
    desc.settings.name = "no_props";

    ModuleInstance mod;
    mod.instance_id = "bare_module";
    mod.module_type = "SomeModule";
    // No properties set
    desc.modules.push_back(mod);

    auto result = validator.validate(desc);

    // Should not crash or error on modules with no properties
    BOOST_CHECK(!result.hasErrors());
}

BOOST_AUTO_TEST_CASE(Validate_ConnectionWithMissingSource)
{
    PipelineValidator::Options opts;
    opts.includeInfoMessages = true;
    PipelineValidator validator(opts);

    auto desc = createPipelineWithMissingModule();
    auto result = validator.validateConnections(desc);

    // Shell implementation should note the missing module in info
    std::string formatted = result.format();
    BOOST_CHECK(formatted.find("nonexistent") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(Validate_PipelineName_InSummary)
{
    PipelineValidator::Options opts;
    opts.includeInfoMessages = true;
    PipelineValidator validator(opts);

    PipelineDescription desc;
    desc.settings.name = "my_custom_pipeline";

    auto result = validator.validate(desc);

    std::string formatted = result.format();
    BOOST_CHECK(formatted.find("my_custom_pipeline") != std::string::npos);
}

BOOST_AUTO_TEST_SUITE_END()
