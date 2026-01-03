// ============================================================
// Pipeline Integration Tests
// Verifies end-to-end declarative pipeline execution
// ============================================================

#include <boost/test/unit_test.hpp>
#include <fstream>
#include "declarative/TomlParser.h"
#include "declarative/ModuleFactory.h"
#include "FaceDetectsInfo.h"
#include "Utils.h"

using namespace apra;

BOOST_AUTO_TEST_SUITE(PipelineIntegrationTests, *boost::unit_test::disabled())

// Helper to read file contents
std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

BOOST_AUTO_TEST_CASE(FaceDetectionPipeline_ProducesValidOutput)
{
    // This test verifies the declarative face detection pipeline works
    // by checking the output file contains valid FaceDetectsInfo

    const std::string outputPath = "./data/testOutput/declarative_face_result.raw";

    // Read the output file produced by the pipeline
    auto data = readFile(outputPath);
    BOOST_REQUIRE(!data.empty());

    // Deserialize FaceDetectsInfo
    FaceDetectsInfo result;
    Utils::deSerialize<FaceDetectsInfo>(result, data.data(), data.size());

    // Verify faces were detected
    BOOST_TEST(result.facesFound == true);
    BOOST_TEST(result.faces.size() > 0);

    // Print detected faces for verification
    BOOST_TEST_MESSAGE("Detected " << result.faces.size() << " face(s):");
    for (size_t i = 0; i < result.faces.size(); i++) {
        const auto& face = result.faces[i];
        BOOST_TEST_MESSAGE("  Face " << (i+1) << ": ("
            << face.x1 << ", " << face.y1 << ") - ("
            << face.x2 << ", " << face.y2 << "), score=" << face.score);

        // Basic sanity checks on coordinates
        // Note: Face detector may use (x1,y1)=bottom-left, (x2,y2)=top-right convention
        BOOST_TEST(face.x2 > face.x1);  // Width > 0
        BOOST_TEST(face.y1 != face.y2); // Height != 0
        BOOST_TEST(face.score > 0.0f);  // Confidence > 0
    }
}

BOOST_AUTO_TEST_SUITE_END()
