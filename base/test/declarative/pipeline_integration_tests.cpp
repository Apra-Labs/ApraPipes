// ============================================================
// Pipeline Integration Tests
// Verifies end-to-end declarative pipeline execution using TOML files
// ============================================================

#include <boost/test/unit_test.hpp>
#include <fstream>
#include <thread>
#include <chrono>
#include "declarative/TomlParser.h"
#include "declarative/ModuleFactory.h"
#include "declarative/PipelineDescription.h"
#include "FaceDetectsInfo.h"
#include "Utils.h"

using namespace apra;

BOOST_AUTO_TEST_SUITE(PipelineIntegrationTests, *boost::unit_test::disabled())

namespace {
    const std::string FACE_DETECTION_TOML = "./docs/declarative-pipeline/examples/working/09_face_detection_demo.toml";
    const std::string TEST_OUTPUT_PATH = "./data/testOutput/declarative_face_result.raw";

    std::vector<uint8_t> readFile(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) return {};
        auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<uint8_t> data(size);
        file.read(reinterpret_cast<char*>(data.data()), size);
        return data;
    }
}

BOOST_AUTO_TEST_CASE(FaceDetectionPipeline_FromToml_ProducesValidOutput)
{
    // Clean up any existing output file
    std::remove(TEST_OUTPUT_PATH.c_str());

    // Parse TOML file
    TomlParser parser;
    auto parseResult = parser.parseFile(FACE_DETECTION_TOML);
    BOOST_REQUIRE_MESSAGE(parseResult.success,
        "Failed to parse TOML: " + parseResult.error);

    // Build pipeline from parsed description
    ModuleFactory factory;
    auto buildResult = factory.build(parseResult.description);
    BOOST_REQUIRE_MESSAGE(buildResult.success(), buildResult.formatIssues());

    // Initialize
    BOOST_REQUIRE(buildResult.pipeline->init());

    // Run with threads and wait for completion
    buildResult.pipeline->run_all_threaded();

    // Wait for pipeline to process (single image, should be quick)
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Stop
    buildResult.pipeline->stop();
    buildResult.pipeline->term();
    buildResult.pipeline->wait_for_all();

    // Verify output file was created and contains valid face detection data
    auto data = readFile(TEST_OUTPUT_PATH);
    BOOST_REQUIRE_MESSAGE(!data.empty(), "Output file was not created or is empty");

    // Deserialize FaceDetectsInfo
    FaceDetectsInfo result;
    Utils::deSerialize<FaceDetectsInfo>(result, data.data(), data.size());

    // Verify faces were detected (faces.jpg has 5 faces)
    BOOST_TEST(result.facesFound == true);
    BOOST_TEST(result.faces.size() == 5);

    // Verify face data is valid
    for (const auto& face : result.faces) {
        BOOST_TEST(face.x2 > face.x1);   // Width > 0
        BOOST_TEST(face.score > 0.5f);   // Reasonable confidence
    }

    // Clean up
    std::remove(TEST_OUTPUT_PATH.c_str());
}

BOOST_AUTO_TEST_SUITE_END()
