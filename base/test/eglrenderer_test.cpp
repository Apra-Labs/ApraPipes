#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "EglRenderer.h"
#include "PipeLine.h"

BOOST_AUTO_TEST_SUITE(eglrenderer_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	int width = 640;
	int height = 480;

    FileReaderModuleProps fileReaderProps("./data/ArgusCamera");
	fileReaderProps.fps = 30;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::UYVY, CV_8UC1, 0, CV_8U, FrameMetadata::MemType::DMABUF, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	fileReader->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
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