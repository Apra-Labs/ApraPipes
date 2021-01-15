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

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	p.stop();
	p.term();

	p.wait_for_all();

	// Test_Utils::saveOrCompare("./data/testOutput/Raw_YUV420_640x360.h264", 0);
}

BOOST_AUTO_TEST_CASE(yuv420_640x360_pipeline, *boost::unit_test::disabled())
{
	// std::cout << "starting performance measurement" << std::endl;
	// auto cuContext = apracucontext_sp(new ApraCUcontext());

	// // metadata is known
	// auto width = 640;
	// auto height = 360;

	// auto fileReaderProps = FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw");
	// fileReaderProps.fps = 10000;
	// auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	// auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	// auto rawImagePin = fileReader->addOutputPin(metadata);

	// cudaStream_t stream;
	// cudaStreamCreate(&stream);
	// auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, stream);
	// copyProps.sync = true;
	// auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	// fileReader->setNext(copy);

	// H264EncoderNVCodecProps encoderProps(cuContext);
	// encoderProps.logHealth = true;
	// auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(encoderProps));
	// copy->setNext(encoder);

	// auto sink = boost::shared_ptr<Module>(new StatSink());
	// encoder->setNext(sink);

	// PipeLine p("test");
	// p.appendModule(fileReader);
	// BOOST_TEST(p.init());

	// Logger::setLogLevel(boost::log::trivial::severity_level::info);
	// p.run_all_threaded();

	// boost::this_thread::sleep_for(boost::chrono::seconds(10));
	// Logger::setLogLevel(boost::log::trivial::severity_level::error);
	// p.stop();
	// p.term();

	// p.wait_for_all();
	// cudaStreamDestroy(stream);
}

BOOST_AUTO_TEST_SUITE_END()
