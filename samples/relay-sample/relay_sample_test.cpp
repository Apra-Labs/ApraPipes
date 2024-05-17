#include <boost/test/unit_test.hpp>
#include <chrono>
#include "relay_sample.h"

BOOST_AUTO_TEST_SUITE(relay_sample)

BOOST_AUTO_TEST_CASE(relaySample)
{
    auto relayPipeline = boost::shared_ptr<RelayPipeline>(new RelayPipeline());

    relayPipeline->setupPipeline();
    relayPipeline->startPipeline();
}

BOOST_AUTO_TEST_SUITE_END()
