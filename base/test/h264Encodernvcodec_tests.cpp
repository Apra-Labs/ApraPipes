#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "CudaMemCopy.h"
#include "CCNPPI.h"
#include "CudaStreamSynchronize.h"
#include "H264EncoderNVCodec.h"
#include "ResizeNPPI.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "ExternalSinkModule.h"
#include "StatSink.h"
#include "H264EncoderNVCodecHelper.h"


namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(h264encodernvcodec_tests)

//preempt test failure if the platform does not support H264 encode
struct if_h264_encoder_supported{
  tt::assertion_result operator()(utf::test_unit_id)
  {
	try{
		auto cuContext = apracucontext_sp(new ApraCUcontext());
		H264EncoderNVCodecHelper h(1000, cuContext, 30, 30, H264EncoderNVCodecProps::BASELINE, false);
		
	}
	catch(AIP_Exception& ex)
	{
		LOG_ERROR << ex.what();
		LOG_ERROR << "skipping tests";
		return false;
	}
	return true;
  }
};

BOOST_AUTO_TEST_CASE(yuv420_640x360,
* utf::precondition(if_h264_encoder_supported()))
{
	Logger::setLogLevel("info");
	auto cuContext = apracucontext_sp(new ApraCUcontext());

	// metadata is known
	auto width = 640;
	auto height = 360;
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::BASELINE;
	bool enableBFrames = true;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	
	auto cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());

	auto copyProps = CudaMemCopyProps(cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStream_);
	copyProps.sync = true;
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);
	auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	copy->setNext(encoder);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/h264images/Raw_YUV420_640x360????.h264")));
	encoder->setNext(fileWriter);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(fileWriter->init());

	fileReader->play(true);

	for (auto i = 0; i < 43; i++)
	{
		fileReader->step();
		copy->step();
		encoder->step();
		fileWriter->step();
	}	
}

BOOST_AUTO_TEST_CASE(yuv420_640x360_resize,
* utf::precondition(if_h264_encoder_supported()))
{
	std::vector<std::string> outFile = { "./data/testOutput/Raw_YUV420_640x360_to_160x90.h264" };
	Test_Utils::FileCleaner f(outFile);
	auto cuContext = apracucontext_sp(new ApraCUcontext());
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::MAIN;
	bool enableBFrames = true;
	auto width = 640;
	auto height = 360;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(width >> 2, height >> 2, cudaStream_)));
	copy->setNext(resize);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream_)));
	resize->setNext(sync);

	auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	sync->setNext(encoder);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0],true)));
	encoder->setNext(fileWriter);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(resize->init());
	BOOST_TEST(sync->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(fileWriter->init());

	fileReader->play(true);


	for (auto i = 0; i < 42; i++)
	{
		fileReader->step();
		copy->step();
		resize->step();
		sync->step();
		encoder->step();
		fileWriter->step();
	}

	Test_Utils::saveOrCompare(outFile[0], 0);
}

BOOST_AUTO_TEST_CASE(yuv420_640x360_sync,
* utf::precondition(if_h264_encoder_supported()))
{
	std::vector<std::string> outFile = { "./data/testOutput/Raw_YUV420_640x360.h264" };
	Test_Utils::FileCleaner f(outFile);

	auto cuContext = apracucontext_sp(new ApraCUcontext());
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	bool enableBFrames = 1;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::HIGH;
	// metadata is known
	auto width = 640;
	auto height = 360;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_)));
	fileReader->setNext(copy);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream_)));
	copy->setNext(sync);

	auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	sync->setNext(encoder);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0],true)));
	encoder->setNext(fileWriter);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(sync->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(fileWriter->init());

	fileReader->play(true);


	for (auto i = 0; i < 42; i++)
	{
		fileReader->step();
		copy->step();
		sync->step();
		encoder->step();
		fileWriter->step();
	}

	Test_Utils::saveOrCompare(outFile[0], 0);
	
}

BOOST_AUTO_TEST_CASE(overlay_1920x960_BGRA,
* utf::precondition(if_h264_encoder_supported()))
{
	std::vector<std::string> outFile = { "./data/testOutput/overlay_1920x960_BGRA.h264" };
	Test_Utils::FileCleaner f(outFile);

	auto cuContext = apracucontext_sp(new ApraCUcontext());
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	
	bool enableBFrames = 1;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::BASELINE;
	// metadata is known
	auto width = 1920;
	auto height = 960;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_1920x960_BGRA.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	copyProps.sync = true;
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	copy->setNext(encoder);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0],true)));
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

	Test_Utils::saveOrCompare(outFile[0], 0);
	
}

BOOST_AUTO_TEST_CASE(mono_1920x960,
* utf::precondition(if_h264_encoder_supported()))
{	
	std::vector<std::string> outFile = { "./data/testOutput/mono_1920x960.h264" };
	Test_Utils::FileCleaner f(outFile);

	auto cuContext = apracucontext_sp(new ApraCUcontext());
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::BASELINE;
	bool enableBFrames = 1;
	// metadata is known
	auto width = 1920;
	auto height = 960;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	auto cc = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV420, cudaStream_)));
	copy->setNext(cc);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream_)));
	cc->setNext(sync);

	auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	sync->setNext(encoder);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0],true)));
	encoder->setNext(fileWriter);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(cc->init());
	BOOST_TEST(sync->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(fileWriter->init());

	fileReader->play(true);


	for (auto i = 0; i < 42; i++)
	{
		fileReader->step(); 
		copy->step();
		cc->step();
		sync->step();
		encoder->step();
		fileWriter->step();
	}

	Test_Utils::saveOrCompare(outFile[0], 0);
}

void mono_1920x960_ext_sink_()
{
	auto cuContext = apracucontext_sp(new ApraCUcontext());
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::BASELINE;
	bool enableBFrames = 1;
	// metadata is known
	auto width = 1920;
	auto height = 960;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	auto cc = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV420, cudaStream_)));
	copy->setNext(cc);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream_)));
	cc->setNext(sync);

	auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	sync->setNext(encoder);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(cc->init());
	BOOST_TEST(sync->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);

	frame_sp frame;

	for (auto i = 0; i < 5; i++)
	{
		fileReader->step();
		copy->step();
		cc->step();
		sync->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST(frames.size() == 1);
		frame = frames.cbegin()->second;
	}

	
}

BOOST_AUTO_TEST_CASE(yuv420_640x360_pipeline, *boost::unit_test::disabled())
{
	std::cout << "starting performance measurement" << std::endl;
	auto cuContext = apracucontext_sp(new ApraCUcontext());
	// metadata is known
	auto width = 640;
	auto height = 360;
	
	auto fileReaderProps = FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw");
	fileReaderProps.fps = 10000;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());;
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	copyProps.sync = true;
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	H264EncoderNVCodecProps encoderProps(cuContext);
	encoderProps.logHealth = true;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(encoderProps));
	copy->setNext(encoder);

	auto sink = boost::shared_ptr<Module>(new StatSink());
	encoder->setNext(sink);

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

BOOST_AUTO_TEST_CASE(mono_1920x960_pipeline, *boost::unit_test::disabled())
{
	auto cuContext = apracucontext_sp(new ApraCUcontext());
	auto width = 1920;
	auto height = 960;
	
	auto fileReaderProps = FileReaderModuleProps("./data/mono_1920x960.raw");
	fileReaderProps.fps = 10000;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	auto cc = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV420, cudaStream_)));
	copy->setNext(cc);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream_)));
	cc->setNext(sync);

	H264EncoderNVCodecProps encoderProps(cuContext);
	encoderProps.logHealth = true;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(encoderProps));
	sync->setNext(encoder);

	auto sink = boost::shared_ptr<Module>(new StatSink());
	encoder->setNext(sink);

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
