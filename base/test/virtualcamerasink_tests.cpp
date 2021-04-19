#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "VirtualCameraSink.h"
#include "PipeLine.h"
#include "StatSink.h"

BOOST_AUTO_TEST_SUITE(virtualcamerasink_tests)

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

	VirtualCameraSinkProps sinkProps("/dev/video10");
	// sinkProps.logHealth = true;
	// sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new VirtualCameraSink(sinkProps));
	fileReader->setNext(sink);	

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(yuv420, *boost::unit_test::disabled())
{

	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 640;
	auto height = 360;

	FileReaderModuleProps fileReaderProps("./data/Raw_YUV420_640x360");
	fileReaderProps.fps = 30;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));	
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U, FrameMetadata::MemType::HOST));

	fileReader->addOutputPin(metadata);

	VirtualCameraSinkProps sinkProps("/dev/video10");
	// sinkProps.logHealth = true;
	// sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new VirtualCameraSink(sinkProps));
	fileReader->setNext(sink);	

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(nv12, *boost::unit_test::disabled())
{
	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 1280;
	auto height = 720;

	FileReaderModuleProps fileReaderProps("./data/Raw_NV12_1280x720");
	fileReaderProps.fps = 30;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));	
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::NV12, size_t(0), CV_8U, FrameMetadata::MemType::HOST));

	fileReader->addOutputPin(metadata);

	VirtualCameraSinkProps sinkProps("/dev/video10");
	// sinkProps.logHealth = true;
	// sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new VirtualCameraSink(sinkProps));
	fileReader->setNext(sink);	

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000));
	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(bgra, *boost::unit_test::disabled())
{
	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 1280;
	auto height = 720;

	FileReaderModuleProps fileReaderProps("./data/Raw_BGRA_1280x720");
	fileReaderProps.fps = 30;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));	
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::BGRA, CV_8UC4, width*4, CV_8U, FrameMetadata::HOST));

	fileReader->addOutputPin(metadata);

	VirtualCameraSinkProps sinkProps("/dev/video10");
	// sinkProps.logHealth = true;
	// sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new VirtualCameraSink(sinkProps));
	fileReader->setNext(sink);	

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000));
	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
