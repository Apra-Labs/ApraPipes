#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "Logger.h"
#include "PaceMaker.h"

// NOTE: TESTS WHICH REQUIRE ANY ENVIRONMENT TO BE PRESENT BEFORE RUNNING ARE NOT UNIT TESTS !!!

BOOST_AUTO_TEST_SUITE(logger_tests)

BOOST_AUTO_TEST_CASE(basic)
{
	LoggerProps props;
	props.enableConsoleLog = true;
	props.enableFileLog = true;
	
	Logger::initLogger(props);
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	for (auto i = 0; i < 1000; i++)
	{
		LOG_INFO << "HELLO WORLD " << i;
	}
}

void listener(const std::string& msg)
{
	std::cout << msg << "-------------------------" << std::endl;
}

BOOST_AUTO_TEST_CASE(listener_test)
{
	LoggerProps props;
	props.enableConsoleLog = true;
	props.enableFileLog = true;
	
	Logger::initLogger(props);
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	for (auto i = 0; i < 10; i++)
	{
		LOG_INFO << "HELLO WORLD " << i;
	}

	Logger::setListener(listener);

	for (auto i = 10; i < 20; i++)
	{
		LOG_INFO << "HELLO WORLD " << i;
	}

	Logger::setListener(nullptr);
}

BOOST_AUTO_TEST_SUITE_END()