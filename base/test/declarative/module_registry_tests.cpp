// ============================================================
// Unit tests for declarative/ModuleRegistry.h
// Task A2: Module Registry
// ============================================================

#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "declarative/ModuleRegistry.h"
#include <thread>
#include <vector>
#include <atomic>

using namespace apra;

// ============================================================
// Test Module with Metadata struct (for testing REGISTER_MODULE pattern)
// ============================================================
namespace test_modules {

// A mock module class for testing - doesn't inherit from real Module
// since we just want to test the registry machinery
class MockModule {
public:
    struct Metadata {
        static constexpr std::string_view name = "MockModule";
        static constexpr ModuleCategory category = ModuleCategory::Transform;
        static constexpr std::string_view version = "1.0.0";
        static constexpr std::string_view description = "A mock module for testing the registry";

        static constexpr std::array<std::string_view, 2> tags = {
            "mock",
            "test"
        };

        static constexpr std::array<PinDef, 1> inputs = {
            PinDef::create("input", "RawImagePlanar", "RawImagePacked", true, "Input frames")
        };

        static constexpr std::array<PinDef, 1> outputs = {
            PinDef::create("output", "RawImagePlanar", true, "Output frames")
        };

        static constexpr std::array<PropDef, 3> properties = {
            PropDef::Integer("buffer_size", 10, 1, 100, "Frame buffer size"),
            PropDef::DynamicFloat("scale", 1.0, 0.1, 10.0, "Scale factor"),
            PropDef::Enum("mode", "auto", "auto", "manual", "Operation mode", PropDef::Mutability::Static)
        };
    };
};

// Another mock module for category filtering tests
class SourceMockModule {
public:
    struct Metadata {
        static constexpr std::string_view name = "SourceMockModule";
        static constexpr ModuleCategory category = ModuleCategory::Source;
        static constexpr std::string_view version = "2.0.0";
        static constexpr std::string_view description = "A source module for testing";

        static constexpr std::array<std::string_view, 3> tags = {
            "source",
            "test",
            "file"
        };

        static constexpr std::array<PinDef, 0> inputs = {};

        static constexpr std::array<PinDef, 1> outputs = {
            PinDef::create("output", "H264Frame", true, "Encoded output")
        };

        static constexpr std::array<PropDef, 1> properties = {
            PropDef::Text("path", "/tmp/video.mp4", "File path")
        };
    };
};

// Helper function to create ModuleInfo from Metadata
template<typename ModuleClass>
ModuleInfo createModuleInfo() {
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

    return info;
}

} // namespace test_modules

BOOST_AUTO_TEST_SUITE(ModuleRegistryTests)

// ============================================================
// Fixture to clear registry between tests
// ============================================================
struct RegistryFixture {
    RegistryFixture() {
        ModuleRegistry::instance().clear();
    }
    ~RegistryFixture() {
        ModuleRegistry::instance().clear();
    }
};

// ============================================================
// Singleton Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(Instance_ReturnsSingleton, RegistryFixture)
{
    auto& reg1 = ModuleRegistry::instance();
    auto& reg2 = ModuleRegistry::instance();
    BOOST_CHECK_EQUAL(&reg1, &reg2);
}

// ============================================================
// Registration Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(RegisterModule_AddsModuleToRegistry, RegistryFixture)
{
    auto info = test_modules::createModuleInfo<test_modules::MockModule>();

    ModuleRegistry::instance().registerModule(std::move(info));

    BOOST_CHECK(ModuleRegistry::instance().hasModule("MockModule"));
    BOOST_CHECK_EQUAL(ModuleRegistry::instance().size(), 1);
}

BOOST_FIXTURE_TEST_CASE(RegisterModule_DuplicateDoesNotOverwrite, RegistryFixture)
{
    auto info1 = test_modules::createModuleInfo<test_modules::MockModule>();
    info1.version = "1.0.0";

    auto info2 = test_modules::createModuleInfo<test_modules::MockModule>();
    info2.version = "2.0.0";

    ModuleRegistry::instance().registerModule(std::move(info1));
    ModuleRegistry::instance().registerModule(std::move(info2));

    // First registration wins
    const auto* registered = ModuleRegistry::instance().getModule("MockModule");
    BOOST_REQUIRE(registered != nullptr);
    BOOST_CHECK_EQUAL(registered->version, "1.0.0");
    BOOST_CHECK_EQUAL(ModuleRegistry::instance().size(), 1);
}

// ============================================================
// Query Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(HasModule_ReturnsTrueForRegistered, RegistryFixture)
{
    auto info = test_modules::createModuleInfo<test_modules::MockModule>();
    ModuleRegistry::instance().registerModule(std::move(info));

    BOOST_CHECK(ModuleRegistry::instance().hasModule("MockModule"));
}

BOOST_FIXTURE_TEST_CASE(HasModule_ReturnsFalseForUnregistered, RegistryFixture)
{
    BOOST_CHECK(!ModuleRegistry::instance().hasModule("NonExistent"));
}

BOOST_FIXTURE_TEST_CASE(GetModule_ReturnsPointerForRegistered, RegistryFixture)
{
    auto info = test_modules::createModuleInfo<test_modules::MockModule>();
    ModuleRegistry::instance().registerModule(std::move(info));

    const auto* module = ModuleRegistry::instance().getModule("MockModule");
    BOOST_REQUIRE(module != nullptr);
    BOOST_CHECK_EQUAL(module->name, "MockModule");
    BOOST_CHECK(module->category == ModuleCategory::Transform);
    BOOST_CHECK_EQUAL(module->version, "1.0.0");
}

BOOST_FIXTURE_TEST_CASE(GetModule_ReturnsNullptrForUnregistered, RegistryFixture)
{
    const auto* module = ModuleRegistry::instance().getModule("NonExistent");
    BOOST_CHECK(module == nullptr);
}

BOOST_FIXTURE_TEST_CASE(GetAllModules_ReturnsAllRegisteredNames, RegistryFixture)
{
    auto info1 = test_modules::createModuleInfo<test_modules::MockModule>();
    auto info2 = test_modules::createModuleInfo<test_modules::SourceMockModule>();

    ModuleRegistry::instance().registerModule(std::move(info1));
    ModuleRegistry::instance().registerModule(std::move(info2));

    auto modules = ModuleRegistry::instance().getAllModules();
    BOOST_CHECK_EQUAL(modules.size(), 2);

    // Check both are present (order may vary)
    bool hasMock = std::find(modules.begin(), modules.end(), "MockModule") != modules.end();
    bool hasSource = std::find(modules.begin(), modules.end(), "SourceMockModule") != modules.end();
    BOOST_CHECK(hasMock);
    BOOST_CHECK(hasSource);
}

BOOST_FIXTURE_TEST_CASE(GetModulesByCategory_FiltersCorrectly, RegistryFixture)
{
    auto info1 = test_modules::createModuleInfo<test_modules::MockModule>();
    auto info2 = test_modules::createModuleInfo<test_modules::SourceMockModule>();

    ModuleRegistry::instance().registerModule(std::move(info1));
    ModuleRegistry::instance().registerModule(std::move(info2));

    auto sourceModules = ModuleRegistry::instance().getModulesByCategory(ModuleCategory::Source);
    BOOST_CHECK_EQUAL(sourceModules.size(), 1);
    BOOST_CHECK_EQUAL(sourceModules[0], "SourceMockModule");

    auto transformModules = ModuleRegistry::instance().getModulesByCategory(ModuleCategory::Transform);
    BOOST_CHECK_EQUAL(transformModules.size(), 1);
    BOOST_CHECK_EQUAL(transformModules[0], "MockModule");

    auto sinkModules = ModuleRegistry::instance().getModulesByCategory(ModuleCategory::Sink);
    BOOST_CHECK_EQUAL(sinkModules.size(), 0);
}

BOOST_FIXTURE_TEST_CASE(GetModulesByTag_FiltersCorrectly, RegistryFixture)
{
    auto info1 = test_modules::createModuleInfo<test_modules::MockModule>();
    auto info2 = test_modules::createModuleInfo<test_modules::SourceMockModule>();

    ModuleRegistry::instance().registerModule(std::move(info1));
    ModuleRegistry::instance().registerModule(std::move(info2));

    // "test" tag is on both
    auto testModules = ModuleRegistry::instance().getModulesByTag("test");
    BOOST_CHECK_EQUAL(testModules.size(), 2);

    // "mock" tag is only on MockModule
    auto mockModules = ModuleRegistry::instance().getModulesByTag("mock");
    BOOST_CHECK_EQUAL(mockModules.size(), 1);
    BOOST_CHECK_EQUAL(mockModules[0], "MockModule");

    // "file" tag is only on SourceMockModule
    auto fileModules = ModuleRegistry::instance().getModulesByTag("file");
    BOOST_CHECK_EQUAL(fileModules.size(), 1);
    BOOST_CHECK_EQUAL(fileModules[0], "SourceMockModule");

    // Non-existent tag
    auto noModules = ModuleRegistry::instance().getModulesByTag("nonexistent");
    BOOST_CHECK_EQUAL(noModules.size(), 0);
}

// ============================================================
// ModuleInfo Content Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(ModuleInfo_HasCorrectPins, RegistryFixture)
{
    auto info = test_modules::createModuleInfo<test_modules::MockModule>();
    ModuleRegistry::instance().registerModule(std::move(info));

    const auto* module = ModuleRegistry::instance().getModule("MockModule");
    BOOST_REQUIRE(module != nullptr);

    // Check inputs
    BOOST_REQUIRE_EQUAL(module->inputs.size(), 1);
    BOOST_CHECK_EQUAL(module->inputs[0].name, "input");
    BOOST_CHECK_EQUAL(module->inputs[0].required, true);
    BOOST_REQUIRE_EQUAL(module->inputs[0].frame_types.size(), 2);
    BOOST_CHECK_EQUAL(module->inputs[0].frame_types[0], "RawImagePlanar");
    BOOST_CHECK_EQUAL(module->inputs[0].frame_types[1], "RawImagePacked");

    // Check outputs
    BOOST_REQUIRE_EQUAL(module->outputs.size(), 1);
    BOOST_CHECK_EQUAL(module->outputs[0].name, "output");
    BOOST_CHECK_EQUAL(module->outputs[0].frame_types[0], "RawImagePlanar");
}

BOOST_FIXTURE_TEST_CASE(ModuleInfo_HasCorrectProperties, RegistryFixture)
{
    auto info = test_modules::createModuleInfo<test_modules::MockModule>();
    ModuleRegistry::instance().registerModule(std::move(info));

    const auto* module = ModuleRegistry::instance().getModule("MockModule");
    BOOST_REQUIRE(module != nullptr);
    BOOST_REQUIRE_EQUAL(module->properties.size(), 3);

    // Check int property
    BOOST_CHECK_EQUAL(module->properties[0].name, "buffer_size");
    BOOST_CHECK_EQUAL(module->properties[0].type, "int");
    BOOST_CHECK_EQUAL(module->properties[0].mutability, "static");
    BOOST_CHECK_EQUAL(module->properties[0].default_value, "10");
    BOOST_CHECK_EQUAL(module->properties[0].min_value, "1");
    BOOST_CHECK_EQUAL(module->properties[0].max_value, "100");

    // Check dynamic float property
    BOOST_CHECK_EQUAL(module->properties[1].name, "scale");
    BOOST_CHECK_EQUAL(module->properties[1].type, "float");
    BOOST_CHECK_EQUAL(module->properties[1].mutability, "dynamic");

    // Check enum property
    BOOST_CHECK_EQUAL(module->properties[2].name, "mode");
    BOOST_CHECK_EQUAL(module->properties[2].type, "enum");
    BOOST_REQUIRE_EQUAL(module->properties[2].enum_values.size(), 2);
    BOOST_CHECK_EQUAL(module->properties[2].enum_values[0], "auto");
    BOOST_CHECK_EQUAL(module->properties[2].enum_values[1], "manual");
}

// ============================================================
// Export Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(ToJson_ReturnsValidJson, RegistryFixture)
{
    auto info = test_modules::createModuleInfo<test_modules::MockModule>();
    ModuleRegistry::instance().registerModule(std::move(info));

    std::string json = ModuleRegistry::instance().toJson();

    // Basic structure checks
    BOOST_CHECK(json.find("\"modules\"") != std::string::npos);
    BOOST_CHECK(json.find("\"MockModule\"") != std::string::npos);
    BOOST_CHECK(json.find("\"transform\"") != std::string::npos);
    BOOST_CHECK(json.find("\"buffer_size\"") != std::string::npos);
}

BOOST_FIXTURE_TEST_CASE(ToToml_ReturnsValidToml, RegistryFixture)
{
    auto info = test_modules::createModuleInfo<test_modules::MockModule>();
    ModuleRegistry::instance().registerModule(std::move(info));

    std::string toml = ModuleRegistry::instance().toToml();

    // Basic structure checks
    BOOST_CHECK(toml.find("[[module]]") != std::string::npos);
    BOOST_CHECK(toml.find("name = \"MockModule\"") != std::string::npos);
    BOOST_CHECK(toml.find("category = \"transform\"") != std::string::npos);
}

// ============================================================
// Thread Safety Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(ConcurrentRegistration_IsThreadSafe, RegistryFixture)
{
    const int numThreads = 10;
    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([i]() {
            ModuleInfo info;
            info.name = "Module" + std::to_string(i);
            info.category = ModuleCategory::Transform;
            info.version = "1.0";
            info.description = "Thread test module";
            ModuleRegistry::instance().registerModule(std::move(info));
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    BOOST_CHECK_EQUAL(ModuleRegistry::instance().size(), numThreads);
}

BOOST_FIXTURE_TEST_CASE(ConcurrentQuery_IsThreadSafe, RegistryFixture)
{
    // Register some modules first
    for (int i = 0; i < 5; ++i) {
        ModuleInfo info;
        info.name = "Module" + std::to_string(i);
        info.category = ModuleCategory::Transform;
        info.version = "1.0";
        info.tags.push_back("test");
        ModuleRegistry::instance().registerModule(std::move(info));
    }

    const int numThreads = 10;
    std::vector<std::thread> threads;
    std::atomic<int> successCount{0};

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&successCount]() {
            // Multiple concurrent queries
            for (int j = 0; j < 100; ++j) {
                auto all = ModuleRegistry::instance().getAllModules();
                auto byCategory = ModuleRegistry::instance().getModulesByCategory(ModuleCategory::Transform);
                auto byTag = ModuleRegistry::instance().getModulesByTag("test");

                if (all.size() == 5 && byCategory.size() == 5 && byTag.size() == 5) {
                    ++successCount;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All queries should have succeeded
    BOOST_CHECK_EQUAL(successCount.load(), numThreads * 100);
}

// ============================================================
// Clear and Size Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(Clear_RemovesAllModules, RegistryFixture)
{
    auto info = test_modules::createModuleInfo<test_modules::MockModule>();
    ModuleRegistry::instance().registerModule(std::move(info));
    BOOST_CHECK_EQUAL(ModuleRegistry::instance().size(), 1);

    ModuleRegistry::instance().clear();

    BOOST_CHECK_EQUAL(ModuleRegistry::instance().size(), 0);
    BOOST_CHECK(!ModuleRegistry::instance().hasModule("MockModule"));
}

BOOST_FIXTURE_TEST_CASE(Size_ReturnsCorrectCount, RegistryFixture)
{
    BOOST_CHECK_EQUAL(ModuleRegistry::instance().size(), 0);

    auto info1 = test_modules::createModuleInfo<test_modules::MockModule>();
    ModuleRegistry::instance().registerModule(std::move(info1));
    BOOST_CHECK_EQUAL(ModuleRegistry::instance().size(), 1);

    auto info2 = test_modules::createModuleInfo<test_modules::SourceMockModule>();
    ModuleRegistry::instance().registerModule(std::move(info2));
    BOOST_CHECK_EQUAL(ModuleRegistry::instance().size(), 2);
}

// ============================================================
// Helper Function Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PropTypeToString_ReturnsCorrectStrings)
{
    BOOST_CHECK_EQUAL(detail::propTypeToString(PropDef::Type::Integer), "int");
    BOOST_CHECK_EQUAL(detail::propTypeToString(PropDef::Type::Floating), "float");
    BOOST_CHECK_EQUAL(detail::propTypeToString(PropDef::Type::Boolean), "bool");
    BOOST_CHECK_EQUAL(detail::propTypeToString(PropDef::Type::Text), "string");
    BOOST_CHECK_EQUAL(detail::propTypeToString(PropDef::Type::Enumeration), "enum");
}

BOOST_AUTO_TEST_CASE(MutabilityToString_ReturnsCorrectStrings)
{
    BOOST_CHECK_EQUAL(detail::mutabilityToString(PropDef::Mutability::Static), "static");
    BOOST_CHECK_EQUAL(detail::mutabilityToString(PropDef::Mutability::Dynamic), "dynamic");
}

BOOST_AUTO_TEST_CASE(CategoryToString_ReturnsCorrectStrings)
{
    BOOST_CHECK_EQUAL(detail::categoryToString(ModuleCategory::Source), "source");
    BOOST_CHECK_EQUAL(detail::categoryToString(ModuleCategory::Sink), "sink");
    BOOST_CHECK_EQUAL(detail::categoryToString(ModuleCategory::Transform), "transform");
    BOOST_CHECK_EQUAL(detail::categoryToString(ModuleCategory::Analytics), "analytics");
    BOOST_CHECK_EQUAL(detail::categoryToString(ModuleCategory::Controller), "controller");
    BOOST_CHECK_EQUAL(detail::categoryToString(ModuleCategory::Utility), "utility");
}

BOOST_AUTO_TEST_SUITE_END()
