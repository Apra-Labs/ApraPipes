// ============================================================
// Unit tests for declarative/PropertyValidators.h
// Task D2: Property Binding System - Validation Tests
// ============================================================

#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "declarative/PropertyValidators.h"

BOOST_AUTO_TEST_SUITE(PropertyValidatorTests)

// ============================================================
// RangeValidator Tests
// ============================================================

BOOST_AUTO_TEST_CASE(IntRangeValidator_InRange_Valid) {
    apra::IntRangeValidator validator(0, 100);

    auto result = validator.validate("port", int64_t(50));
    BOOST_CHECK(result.valid);
    BOOST_CHECK(result.error.empty());
}

BOOST_AUTO_TEST_CASE(IntRangeValidator_AtMin_Valid) {
    apra::IntRangeValidator validator(0, 100);

    auto result = validator.validate("port", int64_t(0));
    BOOST_CHECK(result.valid);
}

BOOST_AUTO_TEST_CASE(IntRangeValidator_AtMax_Valid) {
    apra::IntRangeValidator validator(0, 100);

    auto result = validator.validate("port", int64_t(100));
    BOOST_CHECK(result.valid);
}

BOOST_AUTO_TEST_CASE(IntRangeValidator_BelowMin_Invalid) {
    apra::IntRangeValidator validator(0, 100);

    auto result = validator.validate("port", int64_t(-1));
    BOOST_CHECK(!result.valid);
    BOOST_CHECK_EQUAL(result.propertyName, "port");
    BOOST_CHECK(result.error.find("out of range") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(IntRangeValidator_AboveMax_Invalid) {
    apra::IntRangeValidator validator(0, 100);

    auto result = validator.validate("port", int64_t(101));
    BOOST_CHECK(!result.valid);
    BOOST_CHECK(result.error.find("out of range") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(IntRangeValidator_ExclusiveBounds) {
    // Exclusive bounds: (0, 100) - not including 0 and 100
    apra::IntRangeValidator validator(0, 100, false, false);

    auto result0 = validator.validate("val", int64_t(0));
    BOOST_CHECK(!result0.valid);

    auto result100 = validator.validate("val", int64_t(100));
    BOOST_CHECK(!result100.valid);

    auto result50 = validator.validate("val", int64_t(50));
    BOOST_CHECK(result50.valid);
}

BOOST_AUTO_TEST_CASE(FloatRangeValidator_InRange_Valid) {
    apra::FloatRangeValidator validator(0.0, 1.0);

    auto result = validator.validate("confidence", 0.5);
    BOOST_CHECK(result.valid);
}

BOOST_AUTO_TEST_CASE(FloatRangeValidator_Precision) {
    apra::FloatRangeValidator validator(0.0, 1.0);

    auto result = validator.validate("val", 0.999999);
    BOOST_CHECK(result.valid);

    auto resultOver = validator.validate("val", 1.000001);
    BOOST_CHECK(!resultOver.valid);
}

BOOST_AUTO_TEST_CASE(RangeValidator_Describe) {
    apra::IntRangeValidator validator(1, 65535);
    BOOST_CHECK_EQUAL(validator.describe(), "range[1,65535]");
}

BOOST_AUTO_TEST_CASE(RangeValidator_WrongType_Invalid) {
    apra::IntRangeValidator validator(0, 100);

    auto result = validator.validate("port", std::string("not a number"));
    BOOST_CHECK(!result.valid);
    BOOST_CHECK(result.error.find("Expected numeric") != std::string::npos);
}

// ============================================================
// RegexValidator Tests
// ============================================================

BOOST_AUTO_TEST_CASE(RegexValidator_ValidMatch) {
    apra::RegexValidator validator(apra::patterns::IPv4, "IPv4 address");

    auto result = validator.validate("ip", std::string("192.168.1.1"));
    BOOST_CHECK(result.valid);
}

BOOST_AUTO_TEST_CASE(RegexValidator_InvalidMatch) {
    apra::RegexValidator validator(apra::patterns::IPv4, "IPv4 address");

    auto result = validator.validate("ip", std::string("not.an.ip"));
    BOOST_CHECK(!result.valid);
    BOOST_CHECK(result.error.find("does not match") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(RegexValidator_EmptyString) {
    apra::RegexValidator validator(R"(.+)", "non-empty");

    auto result = validator.validate("name", std::string(""));
    BOOST_CHECK(!result.valid);
}

BOOST_AUTO_TEST_CASE(RegexValidator_CustomPattern) {
    apra::RegexValidator validator(R"(^\d{3}-\d{4}$)", "phone format XXX-XXXX");

    auto valid = validator.validate("phone", std::string("123-4567"));
    BOOST_CHECK(valid.valid);

    auto invalid = validator.validate("phone", std::string("12-34567"));
    BOOST_CHECK(!invalid.valid);
}

BOOST_AUTO_TEST_CASE(RegexValidator_EmailPattern) {
    apra::RegexValidator validator(apra::patterns::Email, "email address");

    auto valid = validator.validate("email", std::string("user@example.com"));
    BOOST_CHECK(valid.valid);

    auto invalid = validator.validate("email", std::string("not-an-email"));
    BOOST_CHECK(!invalid.valid);
}

BOOST_AUTO_TEST_CASE(RegexValidator_IdentifierPattern) {
    apra::RegexValidator validator(apra::patterns::Identifier, "identifier");

    auto valid1 = validator.validate("name", std::string("myVar"));
    BOOST_CHECK(valid1.valid);

    auto valid2 = validator.validate("name", std::string("_private"));
    BOOST_CHECK(valid2.valid);

    auto invalid = validator.validate("name", std::string("123start"));
    BOOST_CHECK(!invalid.valid);
}

BOOST_AUTO_TEST_CASE(RegexValidator_WrongType_Invalid) {
    apra::RegexValidator validator(R"(.+)");

    auto result = validator.validate("name", int64_t(42));
    BOOST_CHECK(!result.valid);
    BOOST_CHECK(result.error.find("Expected string") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(RegexValidator_Describe) {
    apra::RegexValidator validator(R"(\d+)", "digits");
    BOOST_CHECK_EQUAL(validator.describe(), "digits");

    apra::RegexValidator validator2(R"(\d+)");
    BOOST_CHECK(validator2.describe().find("regex") != std::string::npos);
}

// ============================================================
// EnumValidator Tests
// ============================================================

BOOST_AUTO_TEST_CASE(EnumValidator_ValidValue) {
    apra::EnumValidator validator({"mog2", "knn", "gsoc"});

    auto result = validator.validate("algorithm", std::string("mog2"));
    BOOST_CHECK(result.valid);
}

BOOST_AUTO_TEST_CASE(EnumValidator_InvalidValue) {
    apra::EnumValidator validator({"mog2", "knn", "gsoc"});

    auto result = validator.validate("algorithm", std::string("unknown"));
    BOOST_CHECK(!result.valid);
    BOOST_CHECK(result.error.find("not in allowed values") != std::string::npos);
    BOOST_CHECK(result.error.find("mog2") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(EnumValidator_CaseSensitive) {
    apra::EnumValidator validator({"LOW", "MEDIUM", "HIGH"}, true);

    auto valid = validator.validate("level", std::string("LOW"));
    BOOST_CHECK(valid.valid);

    auto invalid = validator.validate("level", std::string("low"));
    BOOST_CHECK(!invalid.valid);
}

BOOST_AUTO_TEST_CASE(EnumValidator_CaseInsensitive) {
    apra::EnumValidator validator({"LOW", "MEDIUM", "HIGH"}, false);

    auto result1 = validator.validate("level", std::string("low"));
    BOOST_CHECK(result1.valid);

    auto result2 = validator.validate("level", std::string("Low"));
    BOOST_CHECK(result2.valid);

    auto result3 = validator.validate("level", std::string("HIGH"));
    BOOST_CHECK(result3.valid);
}

BOOST_AUTO_TEST_CASE(EnumValidator_WrongType_Invalid) {
    apra::EnumValidator validator({"a", "b", "c"});

    auto result = validator.validate("opt", int64_t(1));
    BOOST_CHECK(!result.valid);
    BOOST_CHECK(result.error.find("Expected string") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(EnumValidator_Describe) {
    apra::EnumValidator validator({"a", "b", "c"});
    BOOST_CHECK_EQUAL(validator.describe(), "enum{a,b,c}");
}

BOOST_AUTO_TEST_CASE(EnumValidator_AllowedValues) {
    apra::EnumValidator validator({"x", "y", "z"});
    auto values = validator.allowedValues();
    BOOST_CHECK_EQUAL(values.size(), 3);
    BOOST_CHECK_EQUAL(values[0], "x");
    BOOST_CHECK_EQUAL(values[1], "y");
    BOOST_CHECK_EQUAL(values[2], "z");
}

// ============================================================
// CompositeValidator Tests
// ============================================================

BOOST_AUTO_TEST_CASE(CompositeValidator_AllPass) {
    apra::CompositeValidator composite;
    composite.add(std::make_shared<apra::IntRangeValidator>(1, 65535));

    auto result = composite.validate("port", int64_t(8080));
    BOOST_CHECK(result.valid);
}

BOOST_AUTO_TEST_CASE(CompositeValidator_OneFails) {
    apra::CompositeValidator composite;
    composite.add(std::make_shared<apra::IntRangeValidator>(1, 1024));
    composite.add(std::make_shared<apra::IntRangeValidator>(100, 200));

    // 500 is in [1,1024] but not in [100,200]
    auto result = composite.validate("port", int64_t(500));
    BOOST_CHECK(!result.valid);
}

// ============================================================
// PropertyValidatorRegistry Tests
// ============================================================

BOOST_AUTO_TEST_CASE(ValidatorRegistry_RegisterAndValidate) {
    apra::PropertyValidatorRegistry registry;
    registry.registerValidator("port", std::make_shared<apra::IntRangeValidator>(1, 65535));
    registry.registerValidator("level", std::make_shared<apra::EnumValidator>(
        std::initializer_list<std::string>{"low", "medium", "high"}));

    auto portOk = registry.validate("port", int64_t(8080));
    BOOST_CHECK(portOk.valid);

    auto portBad = registry.validate("port", int64_t(0));
    BOOST_CHECK(!portBad.valid);

    auto levelOk = registry.validate("level", std::string("medium"));
    BOOST_CHECK(levelOk.valid);

    auto levelBad = registry.validate("level", std::string("extreme"));
    BOOST_CHECK(!levelBad.valid);
}

BOOST_AUTO_TEST_CASE(ValidatorRegistry_NoValidator_AlwaysValid) {
    apra::PropertyValidatorRegistry registry;

    // Property without validator is always valid
    auto result = registry.validate("unknown", std::string("anything"));
    BOOST_CHECK(result.valid);
}

BOOST_AUTO_TEST_CASE(ValidatorRegistry_HasValidator) {
    apra::PropertyValidatorRegistry registry;
    registry.registerValidator("port", std::make_shared<apra::IntRangeValidator>(1, 65535));

    BOOST_CHECK(registry.hasValidator("port"));
    BOOST_CHECK(!registry.hasValidator("unknown"));
}

BOOST_AUTO_TEST_CASE(ValidatorRegistry_ValidateAll) {
    apra::PropertyValidatorRegistry registry;
    registry.registerValidator("port", std::make_shared<apra::IntRangeValidator>(1, 65535));
    registry.registerValidator("level", std::make_shared<apra::EnumValidator>(
        std::initializer_list<std::string>{"low", "high"}));

    std::map<std::string, apra::ScalarPropertyValue> props;
    props["port"] = int64_t(0);           // Invalid
    props["level"] = std::string("bad");  // Invalid
    props["name"] = std::string("test");  // No validator - valid

    auto errors = registry.validateAll(props);
    BOOST_CHECK_EQUAL(errors.size(), 2);
}

// ============================================================
// Helper Function Tests
// ============================================================

BOOST_AUTO_TEST_CASE(MakeRangeValidator_Works) {
    auto validator = apra::makeRangeValidator<int64_t>(0, 100);
    auto result = validator->validate("x", int64_t(50));
    BOOST_CHECK(result.valid);
}

BOOST_AUTO_TEST_CASE(MakeRegexValidator_Works) {
    auto validator = apra::makeRegexValidator(R"(\d+)", "digits only");
    auto result = validator->validate("num", std::string("12345"));
    BOOST_CHECK(result.valid);
}

BOOST_AUTO_TEST_CASE(MakeEnumValidator_Works) {
    auto validator = apra::makeEnumValidator({"a", "b", "c"});
    auto result = validator->validate("opt", std::string("b"));
    BOOST_CHECK(result.valid);
}

BOOST_AUTO_TEST_SUITE_END()
