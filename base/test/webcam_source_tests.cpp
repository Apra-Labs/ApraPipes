#include <boost/test/unit_test.hpp>

#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "PipeLine.h"
#include "WebCamSource.h"
#include "ExternalSinkModule.h"
#include "ImageViewerModule.h"

BOOST_AUTO_TEST_SUITE(webcam_tests, * boost::unit_test::disabled())

BOOST_AUTO_TEST_CASE(basic)
{

	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);
	WebCamSourceProps webCamSourceprops(-1, 640, 480);
	auto source = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	source->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(sink->init());
	source->step();
	auto frames = sink->pop();
	auto outFrame = frames.begin()->second;
	BOOST_TEST(outFrame->size() == 640 * 480 * 3);

	webCamSourceprops.width = 352;
	webCamSourceprops.height = 288;
	source->setProps(webCamSourceprops);
	source->step();
	frames = sink->pop();
	outFrame = frames.begin()->second;
	BOOST_TEST(outFrame->isEOS());
	source->step();
	frames = sink->pop();
	outFrame = frames.begin()->second;
	BOOST_TEST(outFrame->size() == 352 * 288 * 3);
}

BOOST_AUTO_TEST_CASE(basic_small)
{

	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);
	WebCamSourceProps webCamSourceprops(-1, 352, 288);
	auto source = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	source->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(sink->init());
	source->step();
	auto frames = sink->pop();
	auto outFrame = frames.begin()->second;
	BOOST_TEST(outFrame->size() == 352 * 288 * 3);
	BOOST_TEST(source->term());
	BOOST_TEST(sink->term());
}

BOOST_AUTO_TEST_CASE(viewer_test)
{   
	WebCamSourceProps webCamSourceprops(-1, 640, 480);
	auto source = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("imageview")));
	source->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	std::this_thread::sleep_for(std::chrono::seconds(10));

	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();

}

BOOST_AUTO_TEST_SUITE_END()