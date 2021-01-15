#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "H264EncoderV4L2.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "StatSink.h"

BOOST_AUTO_TEST_SUITE(h264encoderv4l2_tests)

BOOST_AUTO_TEST_CASE(yuv420_640x360)
{
	// metadata is known
	auto width = 640;
	auto height = 360;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 1024;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	fileReader->setNext(encoder);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/Raw_YUV420_640x360.h264", true)));
	encoder->setNext(fileWriter);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(fileWriter->init());

	fileReader->play(true);


	for (auto i = 0; i < 42; i++)
	{
		fileReader->step();
		encoder->step();
		fileWriter->step();
	}

	Test_Utils::saveOrCompare("./data/testOutput/Raw_YUV420_640x360.h264", 0);
}

BOOST_AUTO_TEST_CASE(yuv420_640x360_profiling, *boost::unit_test::disabled())
{
	// metadata is known
	auto width = 640;
	auto height = 360;

	FileReaderModuleProps fileReaderProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw");
	fileReaderProps.fps = 1000;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 1024;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	fileReader->setNext(encoder);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	encoder->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
