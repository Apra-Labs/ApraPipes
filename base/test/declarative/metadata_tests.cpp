// ============================================================
// Unit tests for declarative/Metadata.h
// Task A1: Core Metadata Types
// ============================================================

#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "declarative/Metadata.h"
#include <array>

using namespace apra;

// ============================================================
// Sample Module Metadata Struct (must be at namespace scope for constexpr static members)
// ============================================================
struct SampleModuleMetadata {
    static constexpr std::string_view name = "SampleModule";
    static constexpr ModuleCategory category = ModuleCategory::Transform;
    static constexpr std::string_view description = "A sample module for testing";
    static constexpr std::string_view version = "1.0";

    static constexpr std::array<std::string_view, 2> tags = {
        "sample",
        "test"
    };

    static constexpr std::array<PinDef, 1> inputs = {
        PinDef::create("input", "RawImagePlanar", true, "Input frames")
    };

    static constexpr std::array<PinDef, 1> outputs = {
        PinDef::create("output", "RawImagePlanar", true, "Output frames")
    };

    static constexpr std::array<PropDef, 3> properties = {
        PropDef::Int("buffer_size", 5, 1, 100, "Frame buffer size"),
        PropDef::DynamicFloat("scale", 1.0, 0.1, 10.0, "Scale factor"),
        PropDef::Bool("enabled", true, "Enable processing")
    };
};

BOOST_AUTO_TEST_SUITE(MetadataTests)

// ============================================================
// ModuleCategory Tests
// ============================================================

BOOST_AUTO_TEST_CASE(ModuleCategory_HasAllSixValues)
{
    // Verify all 6 category values exist and are distinct
    BOOST_CHECK(ModuleCategory::Source != ModuleCategory::Sink);
    BOOST_CHECK(ModuleCategory::Sink != ModuleCategory::Transform);
    BOOST_CHECK(ModuleCategory::Transform != ModuleCategory::Analytics);
    BOOST_CHECK(ModuleCategory::Analytics != ModuleCategory::Controller);
    BOOST_CHECK(ModuleCategory::Controller != ModuleCategory::Utility);

    // Verify we can use them in a switch (compile-time check)
    auto getCategory = [](ModuleCategory cat) -> int {
        switch (cat) {
            case ModuleCategory::Source: return 0;
            case ModuleCategory::Sink: return 1;
            case ModuleCategory::Transform: return 2;
            case ModuleCategory::Analytics: return 3;
            case ModuleCategory::Controller: return 4;
            case ModuleCategory::Utility: return 5;
        }
        return -1;
    };

    BOOST_CHECK_EQUAL(getCategory(ModuleCategory::Source), 0);
    BOOST_CHECK_EQUAL(getCategory(ModuleCategory::Sink), 1);
    BOOST_CHECK_EQUAL(getCategory(ModuleCategory::Transform), 2);
    BOOST_CHECK_EQUAL(getCategory(ModuleCategory::Analytics), 3);
    BOOST_CHECK_EQUAL(getCategory(ModuleCategory::Controller), 4);
    BOOST_CHECK_EQUAL(getCategory(ModuleCategory::Utility), 5);
}

// ============================================================
// PinDef Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PinDef_SingleFrameType)
{
    constexpr auto pin = PinDef::create("output", "H264Frame", true, "Encoded output");

    BOOST_CHECK_EQUAL(pin.name, "output");
    BOOST_CHECK_EQUAL(pin.required, true);
    BOOST_CHECK_EQUAL(pin.description, "Encoded output");
    BOOST_CHECK_EQUAL(pin.frameTypeCount(), 1);
    BOOST_CHECK_EQUAL(pin.frame_types[0], "H264Frame");
}

BOOST_AUTO_TEST_CASE(PinDef_TwoFrameTypes)
{
    constexpr auto pin = PinDef::create(
        "input",
        "RawImagePlanar", "RawImagePacked",
        true,
        "Video input pin"
    );

    BOOST_CHECK_EQUAL(pin.name, "input");
    BOOST_CHECK_EQUAL(pin.required, true);
    BOOST_CHECK_EQUAL(pin.description, "Video input pin");
    BOOST_CHECK_EQUAL(pin.frameTypeCount(), 2);
    BOOST_CHECK_EQUAL(pin.frame_types[0], "RawImagePlanar");
    BOOST_CHECK_EQUAL(pin.frame_types[1], "RawImagePacked");
}

BOOST_AUTO_TEST_CASE(PinDef_ThreeFrameTypes)
{
    constexpr auto pin = PinDef::create(
        "video_in",
        "RawImagePlanar", "RawImagePacked", "H264Frame",
        true,
        "Accepts multiple video formats"
    );

    BOOST_CHECK_EQUAL(pin.frameTypeCount(), 3);
    BOOST_CHECK_EQUAL(pin.frame_types[2], "H264Frame");
}

BOOST_AUTO_TEST_CASE(PinDef_FourFrameTypes)
{
    constexpr auto pin = PinDef::create(
        "multi_in",
        "Type1", "Type2", "Type3", "Type4",
        false,
        "Multi-type optional pin"
    );

    BOOST_CHECK_EQUAL(pin.frameTypeCount(), 4);
    BOOST_CHECK_EQUAL(pin.required, false);
}

BOOST_AUTO_TEST_CASE(PinDef_AcceptsFrameType)
{
    // Explicit 'true' for required to avoid ambiguous overload with 1-frame factory
    // (const char* can implicitly convert to bool)
    constexpr auto pin = PinDef::create("input", "RawImagePlanar", "RawImagePacked", true);

    BOOST_CHECK(pin.acceptsFrameType("RawImagePlanar"));
    BOOST_CHECK(pin.acceptsFrameType("RawImagePacked"));
    BOOST_CHECK(!pin.acceptsFrameType("H264Frame"));
    BOOST_CHECK(!pin.acceptsFrameType("Unknown"));
}

BOOST_AUTO_TEST_CASE(PinDef_DefaultRequired_IsTrue)
{
    constexpr auto pin = PinDef::create("output", "H264Frame");
    BOOST_CHECK_EQUAL(pin.required, true);
}

// ============================================================
// PropDef::Int Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PropDef_Int_CreatesIntegerProperty)
{
    constexpr auto prop = PropDef::Int("device_id", 0, 0, 7, "CUDA device index");

    BOOST_CHECK_EQUAL(prop.name, "device_id");
    BOOST_CHECK(prop.type == PropDef::Type::Integer);
    BOOST_CHECK(prop.mutability == PropDef::Mutability::Static);
    BOOST_CHECK_EQUAL(prop.int_default, 0);
    BOOST_CHECK_EQUAL(prop.int_min, 0);
    BOOST_CHECK_EQUAL(prop.int_max, 7);
    BOOST_CHECK_EQUAL(prop.description, "CUDA device index");
}

BOOST_AUTO_TEST_CASE(PropDef_Int_RangeValidation)
{
    constexpr auto prop = PropDef::Int("buffer_size", 10, 1, 100, "Frame buffer size");

    BOOST_CHECK_EQUAL(prop.int_min, 1);
    BOOST_CHECK_EQUAL(prop.int_max, 100);
    BOOST_CHECK_EQUAL(prop.int_default, 10);
}

// ============================================================
// PropDef::Float Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PropDef_Float_CreatesFloatProperty)
{
    constexpr auto prop = PropDef::Float("sensitivity", 0.5, 0.0, 1.0, "Motion sensitivity");

    BOOST_CHECK_EQUAL(prop.name, "sensitivity");
    BOOST_CHECK(prop.type == PropDef::Type::Floating);
    BOOST_CHECK(prop.mutability == PropDef::Mutability::Static);
    BOOST_CHECK_EQUAL(prop.float_default, 0.5);
    BOOST_CHECK_EQUAL(prop.float_min, 0.0);
    BOOST_CHECK_EQUAL(prop.float_max, 1.0);
    BOOST_CHECK_EQUAL(prop.description, "Motion sensitivity");
}

BOOST_AUTO_TEST_CASE(PropDef_Float_RangeValidation)
{
    constexpr auto prop = PropDef::Float("threshold", 0.75, 0.1, 0.99, "Detection threshold");

    BOOST_CHECK_EQUAL(prop.float_min, 0.1);
    BOOST_CHECK_EQUAL(prop.float_max, 0.99);
    BOOST_CHECK_EQUAL(prop.float_default, 0.75);
}

// ============================================================
// PropDef::Bool Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PropDef_Bool_CreatesBooleanProperty)
{
    constexpr auto prop = PropDef::Bool("enabled", true, "Enable processing");

    BOOST_CHECK_EQUAL(prop.name, "enabled");
    BOOST_CHECK(prop.type == PropDef::Type::Boolean);
    BOOST_CHECK(prop.mutability == PropDef::Mutability::Static);
    BOOST_CHECK_EQUAL(prop.bool_default, true);
    BOOST_CHECK_EQUAL(prop.description, "Enable processing");
}

BOOST_AUTO_TEST_CASE(PropDef_Bool_DefaultFalse)
{
    constexpr auto prop = PropDef::Bool("debug_mode", false, "Enable debug output");

    BOOST_CHECK_EQUAL(prop.bool_default, false);
}

// ============================================================
// PropDef::String Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PropDef_String_CreatesStringProperty)
{
    constexpr auto prop = PropDef::String("url", "rtsp://localhost", "RTSP stream URL");

    BOOST_CHECK_EQUAL(prop.name, "url");
    BOOST_CHECK(prop.type == PropDef::Type::Text);
    BOOST_CHECK(prop.mutability == PropDef::Mutability::Static);
    BOOST_CHECK_EQUAL(prop.string_default, "rtsp://localhost");
    BOOST_CHECK_EQUAL(prop.description, "RTSP stream URL");
}

BOOST_AUTO_TEST_CASE(PropDef_String_WithRegexPattern)
{
    constexpr auto prop = PropDef::String(
        "ip_address",
        "192.168.1.1",
        "Target IP address",
        R"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    );

    BOOST_CHECK_EQUAL(prop.name, "ip_address");
    BOOST_CHECK(prop.type == PropDef::Type::Text);
    BOOST_CHECK(!prop.regex_pattern.empty());
}

// ============================================================
// PropDef::Enum Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PropDef_Enum_TwoValues)
{
    constexpr auto prop = PropDef::Enum(
        "mode",
        "auto",
        "auto", "manual",
        "Operation mode",
        PropDef::Mutability::Static
    );

    BOOST_CHECK_EQUAL(prop.name, "mode");
    BOOST_CHECK(prop.type == PropDef::Type::Enumeration);
    BOOST_CHECK(prop.mutability == PropDef::Mutability::Static);
    BOOST_CHECK_EQUAL(prop.string_default, "auto");
    BOOST_CHECK_EQUAL(prop.enum_value_count, 2);
    BOOST_CHECK_EQUAL(prop.enum_values[0], "auto");
    BOOST_CHECK_EQUAL(prop.enum_values[1], "manual");
}

BOOST_AUTO_TEST_CASE(PropDef_Enum_ThreeValues)
{
    constexpr auto prop = PropDef::Enum(
        "codec",
        "h264",
        "h264", "h265", "jpeg",
        "Video codec to use",
        PropDef::Mutability::Static
    );

    BOOST_CHECK_EQUAL(prop.name, "codec");
    BOOST_CHECK_EQUAL(prop.enum_value_count, 3);
    BOOST_CHECK_EQUAL(prop.enum_values[0], "h264");
    BOOST_CHECK_EQUAL(prop.enum_values[1], "h265");
    BOOST_CHECK_EQUAL(prop.enum_values[2], "jpeg");
}

BOOST_AUTO_TEST_CASE(PropDef_Enum_FourValues)
{
    constexpr auto prop = PropDef::Enum(
        "quality",
        "medium",
        "low", "medium", "high", "ultra",
        "Encoding quality"
    );

    BOOST_CHECK_EQUAL(prop.enum_value_count, 4);
}

// ============================================================
// PropDef Dynamic Variants Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PropDef_DynamicInt_HasDynamicMutability)
{
    constexpr auto prop = PropDef::DynamicInt("frame_skip", 1, 0, 10, "Frames to skip");

    BOOST_CHECK(prop.type == PropDef::Type::Integer);
    BOOST_CHECK(prop.mutability == PropDef::Mutability::Dynamic);
    BOOST_CHECK_EQUAL(prop.int_default, 1);
    BOOST_CHECK_EQUAL(prop.int_min, 0);
    BOOST_CHECK_EQUAL(prop.int_max, 10);
}

BOOST_AUTO_TEST_CASE(PropDef_DynamicFloat_HasDynamicMutability)
{
    constexpr auto prop = PropDef::DynamicFloat("brightness", 0.5, 0.0, 2.0, "Brightness adjustment");

    BOOST_CHECK(prop.type == PropDef::Type::Floating);
    BOOST_CHECK(prop.mutability == PropDef::Mutability::Dynamic);
    BOOST_CHECK_EQUAL(prop.float_default, 0.5);
}

BOOST_AUTO_TEST_CASE(PropDef_DynamicBool_HasDynamicMutability)
{
    constexpr auto prop = PropDef::DynamicBool("paused", false, "Pause processing");

    BOOST_CHECK(prop.type == PropDef::Type::Boolean);
    BOOST_CHECK(prop.mutability == PropDef::Mutability::Dynamic);
    BOOST_CHECK_EQUAL(prop.bool_default, false);
}

BOOST_AUTO_TEST_CASE(PropDef_DynamicString_HasDynamicMutability)
{
    constexpr auto prop = PropDef::DynamicString("overlay_text", "", "Text overlay");

    BOOST_CHECK(prop.type == PropDef::Type::Text);
    BOOST_CHECK(prop.mutability == PropDef::Mutability::Dynamic);
}

BOOST_AUTO_TEST_CASE(PropDef_DynamicEnum_HasDynamicMutability)
{
    constexpr auto prop = PropDef::DynamicEnum(
        "quality",
        "medium",
        "low", "medium", "high",
        "Encoding quality"
    );

    BOOST_CHECK(prop.type == PropDef::Type::Enumeration);
    BOOST_CHECK(prop.mutability == PropDef::Mutability::Dynamic);
}

// ============================================================
// AttrDef Tests
// ============================================================

BOOST_AUTO_TEST_CASE(AttrDef_Int_CreatesIntAttribute)
{
    constexpr auto attr = AttrDef::Int("width", true, "Image width in pixels");

    BOOST_CHECK_EQUAL(attr.name, "width");
    BOOST_CHECK(attr.type == AttrDef::Type::Integer);
    BOOST_CHECK_EQUAL(attr.required, true);
    BOOST_CHECK_EQUAL(attr.description, "Image width in pixels");
}

BOOST_AUTO_TEST_CASE(AttrDef_Int64_CreatesInt64Attribute)
{
    constexpr auto attr = AttrDef::Int64("timestamp", true, "Frame timestamp");

    BOOST_CHECK_EQUAL(attr.name, "timestamp");
    BOOST_CHECK(attr.type == AttrDef::Type::Integer64);
}

BOOST_AUTO_TEST_CASE(AttrDef_Float_CreatesFloatAttribute)
{
    constexpr auto attr = AttrDef::Float("fps", false, "Frames per second");

    BOOST_CHECK_EQUAL(attr.name, "fps");
    BOOST_CHECK(attr.type == AttrDef::Type::Floating);
    BOOST_CHECK_EQUAL(attr.required, false);
}

BOOST_AUTO_TEST_CASE(AttrDef_Bool_CreatesBoolAttribute)
{
    constexpr auto attr = AttrDef::Bool("is_keyframe", true, "Is this a keyframe");

    BOOST_CHECK_EQUAL(attr.name, "is_keyframe");
    BOOST_CHECK(attr.type == AttrDef::Type::Boolean);
}

BOOST_AUTO_TEST_CASE(AttrDef_String_CreatesStringAttribute)
{
    constexpr auto attr = AttrDef::String("codec_name", true, "Codec identifier");

    BOOST_CHECK_EQUAL(attr.name, "codec_name");
    BOOST_CHECK(attr.type == AttrDef::Type::Text);
}

BOOST_AUTO_TEST_CASE(AttrDef_Enum_TwoValues)
{
    constexpr auto attr = AttrDef::Enum(
        "color_space",
        "RGB", "YUV",
        true,
        "Color space"
    );

    BOOST_CHECK_EQUAL(attr.name, "color_space");
    BOOST_CHECK(attr.type == AttrDef::Type::Enumeration);
    BOOST_CHECK_EQUAL(attr.enum_value_count, 2);
}

BOOST_AUTO_TEST_CASE(AttrDef_Enum_FourValues)
{
    constexpr auto attr = AttrDef::Enum(
        "pixel_format",
        "RGB", "BGR", "NV12", "YUV420",
        true,
        "Pixel format"
    );

    BOOST_CHECK_EQUAL(attr.name, "pixel_format");
    BOOST_CHECK(attr.type == AttrDef::Type::Enumeration);
    BOOST_CHECK_EQUAL(attr.enum_value_count, 4);
}

BOOST_AUTO_TEST_CASE(AttrDef_IntArray_CreatesIntArrayAttribute)
{
    constexpr auto attr = AttrDef::IntArray("dimensions", true, "Array dimensions");

    BOOST_CHECK_EQUAL(attr.name, "dimensions");
    BOOST_CHECK(attr.type == AttrDef::Type::IntegerArray);
}

// ============================================================
// Constexpr Compile-Time Tests
// ============================================================

BOOST_AUTO_TEST_CASE(AllTypes_AreConstexprConstructible)
{
    // These compile-time constants prove constexpr works
    constexpr ModuleCategory cat = ModuleCategory::Transform;
    constexpr auto pin = PinDef::create("test", "Frame");
    constexpr auto pin2 = PinDef::create("test2", "F1", "F2");
    constexpr PropDef intProp = PropDef::Int("x", 0, 0, 100);
    constexpr PropDef floatProp = PropDef::Float("y", 0.5, 0.0, 1.0);
    constexpr PropDef boolProp = PropDef::Bool("z", true);
    constexpr PropDef strProp = PropDef::String("s", "default");
    constexpr PropDef enumProp = PropDef::Enum("e", "a", "a", "b");
    constexpr AttrDef intAttr = AttrDef::Int("a");

    // If this compiles, constexpr is working
    BOOST_CHECK(true);

    // Suppress unused variable warnings
    (void)cat;
    (void)pin;
    (void)pin2;
    (void)intProp;
    (void)floatProp;
    (void)boolProp;
    (void)strProp;
    (void)enumProp;
    (void)intAttr;
}

// ============================================================
// Tags as std::array Tests
// ============================================================

BOOST_AUTO_TEST_CASE(Tags_CanBeDeclaredAsStdArray)
{
    // Test that tags can be declared as constexpr std::array
    constexpr std::array<std::string_view, 4> tags = {
        "decoder",
        "h264",
        "nvidia",
        "cuda_required"
    };

    BOOST_CHECK_EQUAL(tags.size(), 4);
    BOOST_CHECK_EQUAL(tags[0], "decoder");
    BOOST_CHECK_EQUAL(tags[1], "h264");
    BOOST_CHECK_EQUAL(tags[2], "nvidia");
    BOOST_CHECK_EQUAL(tags[3], "cuda_required");
}

BOOST_AUTO_TEST_CASE(Tags_EmptyArray)
{
    constexpr std::array<std::string_view, 0> tags = {};
    BOOST_CHECK_EQUAL(tags.size(), 0);
}

// ============================================================
// Sample Module Metadata Struct Test
// ============================================================

BOOST_AUTO_TEST_CASE(SampleModuleMetadata_CanBeDeclared)
{
    // Test that the namespace-scope SampleModuleMetadata struct is accessible
    // (struct is defined at namespace scope above, before BOOST_AUTO_TEST_SUITE)

    // Verify we can access all metadata at compile time
    BOOST_CHECK_EQUAL(SampleModuleMetadata::name, "SampleModule");
    BOOST_CHECK(SampleModuleMetadata::category == ModuleCategory::Transform);
    BOOST_CHECK_EQUAL(SampleModuleMetadata::description, "A sample module for testing");
    BOOST_CHECK_EQUAL(SampleModuleMetadata::version, "1.0");
    BOOST_CHECK_EQUAL(SampleModuleMetadata::tags.size(), 2);
    BOOST_CHECK_EQUAL(SampleModuleMetadata::inputs.size(), 1);
    BOOST_CHECK_EQUAL(SampleModuleMetadata::outputs.size(), 1);
    BOOST_CHECK_EQUAL(SampleModuleMetadata::properties.size(), 3);

    // Check specific property details
    BOOST_CHECK_EQUAL(SampleModuleMetadata::properties[0].name, "buffer_size");
    BOOST_CHECK(SampleModuleMetadata::properties[0].mutability == PropDef::Mutability::Static);
    BOOST_CHECK_EQUAL(SampleModuleMetadata::properties[1].name, "scale");
    BOOST_CHECK(SampleModuleMetadata::properties[1].mutability == PropDef::Mutability::Dynamic);
}

// ============================================================
// Constants Tests
// ============================================================

BOOST_AUTO_TEST_CASE(Constants_HaveExpectedValues)
{
    BOOST_CHECK_EQUAL(MAX_FRAME_TYPES, 8);
    BOOST_CHECK_EQUAL(MAX_ENUM_VALUES, 16);
}

BOOST_AUTO_TEST_SUITE_END()
