#include <boost/test/unit_test.hpp>

#include "NvArgusCamera.h"
#include "StatSink.h"
#include "PipeLine.h"

BOOST_AUTO_TEST_SUITE(nvarguscamera_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	auto source = boost::shared_ptr<Module>(new NvArgusCamera(NvArgusCameraProps(1280, 720, 30)));

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	source->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
