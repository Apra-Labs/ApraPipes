#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "BaresipAdapter.h"

BOOST_AUTO_TEST_SUITE(baresip_adapter_tests)

BOOST_AUTO_TEST_CASE(basic)
{
    auto baresip = boost::shared_ptr<BaresipAdapter>(new BaresipAdapter(BaresipAdapterProps()));
    baresip->init(0,{});
    baresip->processSOS();
    baresip->process();
}

BOOST_AUTO_TEST_SUITE_END()