// ============================================================
// Unit tests for declarative/PropertyMacros.h
// Task D2: Property Binding System
// ============================================================

#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "declarative/PropertyMacros.h"
#include <string>
#include <map>

// ============================================================
// Test Props Classes using DECLARE_PROPS
// ============================================================

// Simple props with all static properties
#define SIMPLE_PROPS(P) \
    P(std::string, path,      Static,  Required, "",    "File path") \
    P(bool,        enabled,   Static,  Optional, true,  "Enable flag") \
    P(int,         count,     Static,  Optional, 10,    "Item count") \
    P(float,       threshold, Static,  Optional, 0.5f,  "Threshold value")

class SimpleTestProps {
public:
    DECLARE_PROPS(SIMPLE_PROPS)
    SimpleTestProps() {}
};

// Props with dynamic properties (can change at runtime)
#define DYNAMIC_PROPS(P) \
    P(std::string, modelPath,   Static,  Required, "",    "Model file path") \
    P(int,         deviceId,    Static,  Optional, 0,     "Device ID") \
    P(float,       scaleFactor, Dynamic, Optional, 1.0f,  "Scale factor") \
    P(float,       confidence,  Dynamic, Optional, 0.5f,  "Confidence threshold") \
    P(bool,        debugMode,   Dynamic, Optional, false, "Debug mode")

class DynamicTestProps {
public:
    DECLARE_PROPS(DYNAMIC_PROPS)
    DynamicTestProps() {}
};

// Props with various integer types
#define INT_TYPES_PROPS(P) \
    P(int,      intVal,    Static, Optional, 42,   "int value") \
    P(int64_t,  int64Val,  Static, Optional, 100,  "int64 value") \
    P(uint32_t, uint32Val, Static, Optional, 200,  "uint32 value") \
    P(double,   doubleVal, Static, Optional, 3.14, "double value")

class IntTypesTestProps {
public:
    DECLARE_PROPS(INT_TYPES_PROPS)
    IntTypesTestProps() {}
};

BOOST_AUTO_TEST_SUITE(PropertyMacrosTests)

// ============================================================
// Member Declaration Tests
// ============================================================

BOOST_AUTO_TEST_CASE(DeclareProps_GeneratesMembers) {
    SimpleTestProps props;

    // Members should exist with default values
    BOOST_CHECK_EQUAL(props.path, "");
    BOOST_CHECK_EQUAL(props.enabled, true);
    BOOST_CHECK_EQUAL(props.count, 10);
    BOOST_CHECK_CLOSE(props.threshold, 0.5f, 0.001f);
}

BOOST_AUTO_TEST_CASE(DeclareProps_DynamicPropsGeneratesMembers) {
    DynamicTestProps props;

    BOOST_CHECK_EQUAL(props.modelPath, "");
    BOOST_CHECK_EQUAL(props.deviceId, 0);
    BOOST_CHECK_CLOSE(props.scaleFactor, 1.0f, 0.001f);
    BOOST_CHECK_CLOSE(props.confidence, 0.5f, 0.001f);
    BOOST_CHECK_EQUAL(props.debugMode, false);
}

// ============================================================
// applyProperties Tests
// ============================================================

BOOST_AUTO_TEST_CASE(ApplyProperties_AllOptional) {
    SimpleTestProps props;
    std::map<std::string, apra::ScalarPropertyValue> values = {
        {"path", std::string("/test/path.mp4")},
        {"enabled", false},
        {"count", int64_t(42)},
        {"threshold", 0.75}
    };
    std::vector<std::string> missing;

    SimpleTestProps::applyProperties(props, values, missing);

    BOOST_CHECK(missing.empty());
    BOOST_CHECK_EQUAL(props.path, "/test/path.mp4");
    BOOST_CHECK_EQUAL(props.enabled, false);
    BOOST_CHECK_EQUAL(props.count, 42);
    BOOST_CHECK_CLOSE(props.threshold, 0.75f, 0.001f);
}

BOOST_AUTO_TEST_CASE(ApplyProperties_MissingRequired) {
    SimpleTestProps props;
    std::map<std::string, apra::ScalarPropertyValue> values = {
        // "path" is required but missing
        {"enabled", true}
    };
    std::vector<std::string> missing;

    SimpleTestProps::applyProperties(props, values, missing);

    BOOST_CHECK_EQUAL(missing.size(), 1);
    BOOST_CHECK_EQUAL(missing[0], "path");
}

BOOST_AUTO_TEST_CASE(ApplyProperties_MissingOptionalUsesDefault) {
    SimpleTestProps props;
    std::map<std::string, apra::ScalarPropertyValue> values = {
        {"path", std::string("/video.mp4")}
        // Optional properties not provided - should use defaults
    };
    std::vector<std::string> missing;

    SimpleTestProps::applyProperties(props, values, missing);

    BOOST_CHECK(missing.empty());
    BOOST_CHECK_EQUAL(props.path, "/video.mp4");
    BOOST_CHECK_EQUAL(props.enabled, true);   // default
    BOOST_CHECK_EQUAL(props.count, 10);       // default
    BOOST_CHECK_CLOSE(props.threshold, 0.5f, 0.001f);  // default
}

BOOST_AUTO_TEST_CASE(ApplyProperties_TypeConversion_IntToSmaller) {
    IntTypesTestProps props;
    std::map<std::string, apra::ScalarPropertyValue> values = {
        {"intVal", int64_t(123)},
        {"int64Val", int64_t(456789)},
        {"uint32Val", int64_t(999)},
        {"doubleVal", 2.718}
    };
    std::vector<std::string> missing;

    IntTypesTestProps::applyProperties(props, values, missing);

    BOOST_CHECK(missing.empty());
    BOOST_CHECK_EQUAL(props.intVal, 123);
    BOOST_CHECK_EQUAL(props.int64Val, 456789);
    BOOST_CHECK_EQUAL(props.uint32Val, 999);
    BOOST_CHECK_CLOSE(props.doubleVal, 2.718, 0.001);
}

// ============================================================
// getProperty Tests
// ============================================================

BOOST_AUTO_TEST_CASE(GetProperty_StringProperty) {
    SimpleTestProps props;
    props.path = "/test/video.mp4";

    auto value = props.getProperty("path");

    BOOST_CHECK(std::holds_alternative<std::string>(value));
    BOOST_CHECK_EQUAL(std::get<std::string>(value), "/test/video.mp4");
}

BOOST_AUTO_TEST_CASE(GetProperty_BoolProperty) {
    SimpleTestProps props;
    props.enabled = false;

    auto value = props.getProperty("enabled");

    BOOST_CHECK(std::holds_alternative<bool>(value));
    BOOST_CHECK_EQUAL(std::get<bool>(value), false);
}

BOOST_AUTO_TEST_CASE(GetProperty_IntProperty) {
    SimpleTestProps props;
    props.count = 42;

    auto value = props.getProperty("count");

    BOOST_CHECK(std::holds_alternative<int64_t>(value));
    BOOST_CHECK_EQUAL(std::get<int64_t>(value), 42);
}

BOOST_AUTO_TEST_CASE(GetProperty_FloatProperty) {
    SimpleTestProps props;
    props.threshold = 0.75f;

    auto value = props.getProperty("threshold");

    BOOST_CHECK(std::holds_alternative<double>(value));
    BOOST_CHECK_CLOSE(std::get<double>(value), 0.75, 0.001);
}

BOOST_AUTO_TEST_CASE(GetProperty_UnknownPropertyThrows) {
    SimpleTestProps props;

    BOOST_CHECK_THROW(props.getProperty("nonexistent"), std::runtime_error);
}

// ============================================================
// setProperty Tests
// ============================================================

BOOST_AUTO_TEST_CASE(SetProperty_DynamicPropertySucceeds) {
    DynamicTestProps props;
    props.scaleFactor = 1.0f;

    bool result = props.setProperty("scaleFactor", 2.5);

    BOOST_CHECK(result);
    BOOST_CHECK_CLOSE(props.scaleFactor, 2.5f, 0.001f);
}

BOOST_AUTO_TEST_CASE(SetProperty_MultipleDynamicProperties) {
    DynamicTestProps props;

    props.setProperty("scaleFactor", 1.5);
    props.setProperty("confidence", 0.8);
    props.setProperty("debugMode", true);

    BOOST_CHECK_CLOSE(props.scaleFactor, 1.5f, 0.001f);
    BOOST_CHECK_CLOSE(props.confidence, 0.8f, 0.001f);
    BOOST_CHECK_EQUAL(props.debugMode, true);
}

BOOST_AUTO_TEST_CASE(SetProperty_StaticPropertyThrows) {
    DynamicTestProps props;

    // modelPath is Static - should throw
    BOOST_CHECK_THROW(
        props.setProperty("modelPath", std::string("/new/path")),
        std::runtime_error
    );

    // deviceId is Static - should throw
    BOOST_CHECK_THROW(
        props.setProperty("deviceId", int64_t(1)),
        std::runtime_error
    );
}

BOOST_AUTO_TEST_CASE(SetProperty_UnknownPropertyThrows) {
    DynamicTestProps props;

    BOOST_CHECK_THROW(
        props.setProperty("nonexistent", int64_t(42)),
        std::runtime_error
    );
}

BOOST_AUTO_TEST_CASE(SetProperty_AllStaticPropsThrow) {
    SimpleTestProps props;

    // All properties in SimpleTestProps are Static
    BOOST_CHECK_THROW(props.setProperty("path", std::string("x")), std::runtime_error);
    BOOST_CHECK_THROW(props.setProperty("enabled", true), std::runtime_error);
    BOOST_CHECK_THROW(props.setProperty("count", int64_t(1)), std::runtime_error);
    BOOST_CHECK_THROW(props.setProperty("threshold", 0.1), std::runtime_error);
}

// ============================================================
// dynamicPropertyNames Tests
// ============================================================

BOOST_AUTO_TEST_CASE(DynamicPropertyNames_NoDynamicProps) {
    auto names = SimpleTestProps::dynamicPropertyNames();

    BOOST_CHECK(names.empty());
}

BOOST_AUTO_TEST_CASE(DynamicPropertyNames_WithDynamicProps) {
    auto names = DynamicTestProps::dynamicPropertyNames();

    BOOST_CHECK_EQUAL(names.size(), 3);
    BOOST_CHECK(std::find(names.begin(), names.end(), "scaleFactor") != names.end());
    BOOST_CHECK(std::find(names.begin(), names.end(), "confidence") != names.end());
    BOOST_CHECK(std::find(names.begin(), names.end(), "debugMode") != names.end());
}

BOOST_AUTO_TEST_CASE(IsPropertyDynamic_Static) {
    BOOST_CHECK(!DynamicTestProps::isPropertyDynamic("modelPath"));
    BOOST_CHECK(!DynamicTestProps::isPropertyDynamic("deviceId"));
}

BOOST_AUTO_TEST_CASE(IsPropertyDynamic_Dynamic) {
    BOOST_CHECK(DynamicTestProps::isPropertyDynamic("scaleFactor"));
    BOOST_CHECK(DynamicTestProps::isPropertyDynamic("confidence"));
    BOOST_CHECK(DynamicTestProps::isPropertyDynamic("debugMode"));
}

// ============================================================
// getPropertyInfos Tests
// ============================================================

BOOST_AUTO_TEST_CASE(GetPropertyInfos_ReturnsAllProps) {
    auto infos = SimpleTestProps::getPropertyInfos();

    BOOST_CHECK_EQUAL(infos.size(), 4);
}

BOOST_AUTO_TEST_CASE(GetPropertyInfos_ContainsCorrectData) {
    auto infos = DynamicTestProps::getPropertyInfos();

    BOOST_CHECK_EQUAL(infos.size(), 5);

    // Find modelPath
    auto it = std::find_if(infos.begin(), infos.end(),
        [](const apra::PropertyInfo& p) { return p.name == "modelPath"; });
    BOOST_REQUIRE(it != infos.end());
    BOOST_CHECK_EQUAL(it->required, true);
    BOOST_CHECK_EQUAL(it->dynamic, false);

    // Find scaleFactor
    it = std::find_if(infos.begin(), infos.end(),
        [](const apra::PropertyInfo& p) { return p.name == "scaleFactor"; });
    BOOST_REQUIRE(it != infos.end());
    BOOST_CHECK_EQUAL(it->required, false);
    BOOST_CHECK_EQUAL(it->dynamic, true);
}

// ============================================================
// Edge Cases
// ============================================================

BOOST_AUTO_TEST_CASE(ApplyProperties_EmptyMap) {
    DynamicTestProps props;
    std::map<std::string, apra::ScalarPropertyValue> values;  // empty
    std::vector<std::string> missing;

    DynamicTestProps::applyProperties(props, values, missing);

    // Only modelPath is required
    BOOST_CHECK_EQUAL(missing.size(), 1);
    BOOST_CHECK_EQUAL(missing[0], "modelPath");

    // Optional properties should have defaults
    BOOST_CHECK_EQUAL(props.deviceId, 0);
    BOOST_CHECK_CLOSE(props.scaleFactor, 1.0f, 0.001f);
}

BOOST_AUTO_TEST_CASE(ApplyProperties_ExtraPropertiesIgnored) {
    SimpleTestProps props;
    std::map<std::string, apra::ScalarPropertyValue> values = {
        {"path", std::string("/video.mp4")},
        {"unknownProp", int64_t(999)},      // Should be ignored
        {"anotherUnknown", std::string("x")} // Should be ignored
    };
    std::vector<std::string> missing;

    SimpleTestProps::applyProperties(props, values, missing);

    // Should succeed without error
    BOOST_CHECK(missing.empty());
    BOOST_CHECK_EQUAL(props.path, "/video.mp4");
}

BOOST_AUTO_TEST_CASE(SetProperty_IntToFloat) {
    DynamicTestProps props;

    // Setting float property with int value
    bool result = props.setProperty("scaleFactor", int64_t(2));

    BOOST_CHECK(result);
    BOOST_CHECK_CLOSE(props.scaleFactor, 2.0f, 0.001f);
}

BOOST_AUTO_TEST_SUITE_END()
