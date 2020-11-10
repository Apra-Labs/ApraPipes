#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "Logger.h"
#include "PaceMaker.h"

// NOTE: TESTS WHICH REQUIRE ANY ENVIRONMENT TO BE PRESENT BEFORE RUNNING ARE NOT UNIT TESTS !!!

BOOST_AUTO_TEST_SUITE(logger_tests)

BOOST_AUTO_TEST_CASE(basic)
{
	LoggerProps props;
	props.enableConsoleLog = false;
	Logger::initLogger(props);

	for (auto i = 0; i < 100; i++)
	{
		LOG_ERROR << "HELLO WORLD " << i;
	}
		
	boost::this_thread::sleep_for(boost::chrono::seconds(1));		
}

void listener(std::string& msg)
{
	std::cout << msg << "-------------------------" << std::endl;
}

BOOST_AUTO_TEST_CASE(listener_test)
{
	LoggerProps props;
	Logger::initLogger(props);

	for (auto i = 0; i < 10; i++)
	{
		LOG_ERROR << "HELLO WORLD " << i;
	}

	Logger::setListener(listener);

	for (auto i = 10; i < 20; i++)
	{
		LOG_ERROR << "HELLO WORLD " << i;
	}

	Logger::setListener(nullptr);

	boost::this_thread::sleep_for(boost::chrono::seconds(1));
}

BOOST_AUTO_TEST_SUITE_END()