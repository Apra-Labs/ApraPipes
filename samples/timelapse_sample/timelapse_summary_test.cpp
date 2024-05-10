#include <boost/test/unit_test.hpp>

#include "TimelapsePipeline.h"

BOOST_AUTO_TEST_SUITE(timeplase_summary)

BOOST_AUTO_TEST_CASE(generateTimeLapse)
{
	auto timelapsePipeline = boost::shared_ptr<TimelapsePipeline>(new TimelapsePipeline());
	timelapsePipeline.setupPipeline();

	timelapsePipeline.startPipeline();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	timelapsePipeline.stopPipeline();
}

BOOST_AUTO_TEST_SUITE_END()