#include <boost/test/unit_test.hpp>

#include "GenerateThumbnailsPipeline.h"

BOOST_AUTO_TEST_SUITE(generateThumbnails)

BOOST_AUTO_TEST_CASE(generateThumbnails_from_mp4)
{
    auto generateThumbnailPipeline = boost::shared_ptr<GenerateThumbnailsPipeline>(new GenerateThumbnailsPipeline());
    generateThumbnailPipeline->setUpPipeLine();
    generateThumbnailPipeline->startPipeLine();
    
    boost::this_thread::sleep_for(boost::chrono::seconds(5));

    generateThumbnailPipeline->stopPipeLine();
}

BOOST_AUTO_TEST_SUITE_END()