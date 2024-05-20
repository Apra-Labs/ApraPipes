#include <boost/test/unit_test.hpp>
#include <chrono>
#include "timelapse_summary.h"

BOOST_AUTO_TEST_SUITE(generate_timelapse)

BOOST_AUTO_TEST_CASE(generateTimeLapse)
{
	auto timelapsePipeline = boost::shared_ptr<TimelapsePipeline>(new TimelapsePipeline());
	timelapsePipeline -> setupPipeline();

	timelapsePipeline -> startPipeline();

	timelapsePipeline -> stopPipeline();
}

BOOST_AUTO_TEST_SUITE_END()