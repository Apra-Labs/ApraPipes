#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FIndexStrategy.h"


BOOST_AUTO_TEST_SUITE(findexstrategy_tests)


BOOST_AUTO_TEST_CASE(case1)
{
    auto strategy = FIndexStrategy::create(FIndexStrategy::FIndexStrategyType::AUTO_INCREMENT);
    BOOST_TEST(strategy->getFIndex(0)==0);
    BOOST_TEST(strategy->getFIndex(0)==1);
    BOOST_TEST(strategy->getFIndex(0)==2);
}

BOOST_AUTO_TEST_CASE(case2)
{
    auto strategy = FIndexStrategy::create(FIndexStrategy::FIndexStrategyType::AUTO_INCREMENT);
    BOOST_TEST(strategy->getFIndex(1)==1);
    BOOST_TEST(strategy->getFIndex(1)==2);
    BOOST_TEST(strategy->getFIndex(1)==3);
}

BOOST_AUTO_TEST_CASE(case3)
{
    auto strategy = FIndexStrategy::create(FIndexStrategy::FIndexStrategyType::AUTO_INCREMENT);
    BOOST_TEST(strategy->getFIndex(10)==10);
    BOOST_TEST(strategy->getFIndex(11)==11);
    BOOST_TEST(strategy->getFIndex(11)==12);
    BOOST_TEST(strategy->getFIndex(12)==13);
}
BOOST_AUTO_TEST_SUITE_END()