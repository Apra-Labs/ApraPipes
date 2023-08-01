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

BOOST_AUTO_TEST_CASE(severity)
{
	LoggerProps props;
	props.enableConsoleLog = true;
	Logger::initLogger(props);

	int i=1;
	BOOST_TEST(!Logger::setLogLevel("foo-bar"),"bad setting test");
	LOG_DEBUG <<"should see this on console "<<i++; 

	BOOST_TEST(Logger::setLogLevel("info"),"info setting test");
	LOG_INFO <<"should see this on console with INFO "<<i++; 
	LOG_DEBUG <<"should not see this on console "<<i++; 
	

	BOOST_TEST(Logger::setLogLevel("fatal"),"fatal setting test");
	LOG_FATAL <<"should see this on console with FATAL "<<i++; 
	LOG_DEBUG <<"should not see this on console "<<i++; 
	LOG_INFO <<"should not see this on console "<<i++; 
	LOG_WARNING <<"should not see this on console "<<i++; 
	
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