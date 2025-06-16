#include <boost/test/unit_test.hpp>
#include <chrono>
#include "test_utils.h"
#include "timelapse_summary.h"

BOOST_AUTO_TEST_SUITE(generate_timelapse)

BOOST_AUTO_TEST_CASE(generateTimeLapse)
{
	auto timelapsePipeline = boost::shared_ptr<TimelapsePipeline>(new TimelapsePipeline());
	
	std::string inputVideoPath = "./data/timelapse_sample_videos";
	std::string outputPath = "./data/timelapse_videos";

	BOOST_ASSERT(timelapsePipeline->setupPipeline(inputVideoPath,outputPath));
	BOOST_ASSERT(timelapsePipeline->startPipeline());
	boost::this_thread::sleep(boost::chrono::seconds(10));

	BOOST_ASSERT(timelapsePipeline->stopPipeline());
}

BOOST_AUTO_TEST_SUITE_END()