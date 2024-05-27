#include <boost/test/unit_test.hpp>
#include <chrono>
#include "../../base/test/test_utils.h"
#include "GenerateThumbnailsPipeline.h"

BOOST_AUTO_TEST_SUITE(generateThumbnails)

BOOST_AUTO_TEST_CASE(generateThumbnails_from_mp4)
{
    auto generateThumbnailPipeline = boost::shared_ptr<GenerateThumbnailsPipeline>(new GenerateThumbnailsPipeline());
    generateThumbnailPipeline->setUpPipeLine();
    generateThumbnailPipeline->startPipeLine();
    
    boost::this_thread::sleep_for(boost::chrono::seconds(5));

   const uint8_t *pReadDataTest = nullptr;
    unsigned int readDataSizeTest = 0U;

    BOOST_TEST(
        Test_Utils::readFile("./data/thumbnail/sample_thumbnail.jpg",
                             pReadDataTest, readDataSizeTest));
    Test_Utils::saveOrCompare(
        "./data/mp4Reader_saveOrCompare/h264/test_thumbnail.jpg",
        pReadDataTest, readDataSizeTest, 0);


    generateThumbnailPipeline->stopPipeLine();
}

BOOST_AUTO_TEST_SUITE_END()