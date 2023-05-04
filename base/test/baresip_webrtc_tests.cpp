#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "BaresipWebRTC.h"

BOOST_AUTO_TEST_SUITE(baresip_webrtc_tests)

BOOST_AUTO_TEST_CASE(basic,* boost::unit_test::disabled())
{
    auto baresip = boost::shared_ptr<BaresipWebRTC>(new BaresipWebRTC(BaresipWebRTCProps()));
    baresip->init(0,{});
    baresip->processSOS();
    baresip->process();
}

BOOST_AUTO_TEST_SUITE_END()