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
#include "CudaMemCopy.h"
#include "RTSPPusher.h"

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

BOOST_AUTO_TEST_CASE(rgb24_1280x720)
{
	// metadata is known
	auto width = 1280;
	auto height = 720;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/Raw_RGB24_1280x720")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::RGB, CV_8UC3, size_t(0), CV_8U, FrameMetadata::HOST, true));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, stream);
	copyProps.sync = true;
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 1024;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	copy->setNext(encoder);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/Raw_RGB24_1280x720.h264", true)));
	encoder->setNext(fileWriter);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(fileWriter->init());

	fileReader->play(true);


	for (auto i = 0; i < 42; i++)
	{
		fileReader->step();
		copy->step();
		encoder->step();
		fileWriter->step();
	}

	Test_Utils::saveOrCompare("./data/testOutput/Raw_RGB24_1280x720.h264", 0);
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

BOOST_AUTO_TEST_CASE(rgb24_1280x720_profiling, *boost::unit_test::disabled())
{
	// metadata is known
	auto width = 1280;
	auto height = 720;

	FileReaderModuleProps fileReaderProps("./data/Raw_RGB24_1280x720");
	fileReaderProps.fps = 1000;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::RGB, CV_8UC3, size_t(0), CV_8U, FrameMetadata::HOST, true));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, stream);
	copyProps.sync = true;
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 1024;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	copy->setNext(encoder);

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


BOOST_AUTO_TEST_CASE(encodepush, *boost::unit_test::disabled())
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

	auto sink = boost::shared_ptr<Module>(new RTSPPusher(RTSPPusherProps("rtsp://10.102.10.129:5544", "aprapipes_h264")));
	encoder->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());
	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	LOG_ERROR << "STOPPING";

	p.stop();
	p.term();
	LOG_ERROR << "WAITING";
	p.wait_for_all();
	LOG_ERROR << "TEST DONE";
}

BOOST_AUTO_TEST_SUITE_END()
