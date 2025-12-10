#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "BMPConverter.h"
#include "PipeLine.h"
#include "StatSink.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(bmpconverter_tests)

BOOST_AUTO_TEST_CASE(rgb)
{
	// metadata is known
	auto width = 1280;
	auto height = 720;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, CV_8UC3, width*3, CV_8U, FrameMetadata::HOST));
	auto rawImagePin = fileReader->addOutputPin(metadata);
	
	auto m2 = boost::shared_ptr<BMPConverter>(new BMPConverter(BMPConverterProps()));
	fileReader->setNext(m2);
	auto outputPin = m2->getAllOutputPinsByType(FrameMetadata::BMP_IMAGE)[0];

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());
		
	fileReader->step();
	m2->step();
	auto frames = m3->pop();
	BOOST_TEST((frames.find(outputPin) != frames.end()));
	auto outputFrame = frames[outputPin];
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::BMP_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/bmpconverter_tests_1280x720_rgb.bmp", (const uint8_t *)outputFrame->data(), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(perf, *boost::unit_test::disabled())
{

	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 1280;
	auto height = 720;

	FileReaderModuleProps fileReaderProps("./data/frame_1280x720_rgb.raw");
	fileReaderProps.fps = 1000;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, CV_8UC3, width*3, CV_8U, FrameMetadata::HOST));
	fileReader->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<Module>(new BMPConverter(BMPConverterProps()));
	fileReader->setNext(m2);	

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto m3 = boost::shared_ptr<Module>(new StatSink(sinkProps));
	m2->setNext(m3);

	PipeLine p("test");
	p.appendModule(fileReader);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(60));
	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
