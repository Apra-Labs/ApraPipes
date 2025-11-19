#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <memory>
#include <thread>
#include <chrono>
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
#include "nv_test_utils.h"
#include "PipeLine.h"
#include "ExternalSinkModule.h"
#include "StatSink.h"
#include "H264EncoderNVCodecHelper.h"


BOOST_AUTO_TEST_SUITE(h264encodernvcodec_tests)

BOOST_AUTO_TEST_CASE(yuv420_640x360_resize,
*boost::unit_test::disabled()
*utf::precondition(if_h264_encoder_supported()))
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

	auto fileReaderProps = FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw");
	fileReaderProps.readLoop = false;
	auto fileReader = std::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = std::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	auto copy = std::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	auto resize = std::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(width >> 2, height >> 2, cudaStream_)));
	copy->setNext(resize);

	auto sync = std::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream_)));
	resize->setNext(sync);

	auto encoder = std::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	sync->setNext(encoder);

	auto fileWriter = std::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0],true)));
	encoder->setNext(fileWriter);

	std::shared_ptr<PipeLine> p;
	p = std::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);
	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(10);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
	std::string fileComparePath = "./data/H264EncoderNvCodecTests/Raw_YUV420_640x360_to_160x90.h264";
	uint8_t* frameData;
	uint frameSize;
	Test_Utils::readFile(outFile[0], (const uint8_t*&)frameData, frameSize);

	Test_Utils::saveOrCompare(fileComparePath.c_str(), (const unsigned char*)frameData, (size_t)frameSize, 0);
}

BOOST_AUTO_TEST_CASE(yuv420_640x360_sync,
*boost::unit_test::disabled()
*utf::precondition(if_h264_encoder_supported()))
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

	auto fileReaderProps = FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw");
	fileReaderProps.readLoop = false;
	auto fileReader = std::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = std::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copy = std::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_)));
	fileReader->setNext(copy);

	auto sync = std::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream_)));
	copy->setNext(sync);

	auto encoder = std::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	sync->setNext(encoder);

	auto fileWriter = std::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0],true)));
	encoder->setNext(fileWriter);

	std::shared_ptr<PipeLine> p;
	p = std::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(10);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();

	std::string fileComparePath = "./data/H264EncoderNvCodecTests/Raw_YUV420_640x360.h264";
	uint8_t* frameData;
	uint frameSize;
	Test_Utils::readFile(outFile[0], (const uint8_t*&)frameData, frameSize);

	Test_Utils::saveOrCompare(fileComparePath.c_str(), (const unsigned char*)frameData, (size_t)frameSize, 0);	
}

BOOST_AUTO_TEST_CASE(overlay_1920x960_BGRA, *boost::unit_test::disabled()
*utf::precondition(if_h264_encoder_supported()))
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

	auto fileReader = std::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_1920x960_BGRA.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = std::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	copyProps.sync = true;
	auto copy = std::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	auto encoder = std::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	copy->setNext(encoder);

	auto fileWriter = std::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0], true)));
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

BOOST_AUTO_TEST_CASE(mono_1920x960, *boost::unit_test::disabled()
*utf::precondition(if_h264_encoder_supported()))
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

	auto fileReader = std::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = std::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	auto copy = std::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	auto cc = std::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV420, cudaStream_)));
	copy->setNext(cc);

	auto sync = std::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream_)));
	cc->setNext(sync);

	auto encoder = std::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	sync->setNext(encoder);

	auto fileWriter = std::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0], true)));
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

BOOST_AUTO_TEST_CASE(yuv420_640x360_pipeline, *boost::unit_test::disabled()
*utf::precondition(if_h264_encoder_supported()))
{
	std::cout << "starting performance measurement" << std::endl;
	auto cuContext = apracucontext_sp(new ApraCUcontext());
	// metadata is known
	auto width = 640;
	auto height = 360;

	auto fileReaderProps = FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw");
	fileReaderProps.fps = 10000;
	auto fileReader = std::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = std::shared_ptr<ApraCudaStream>(new ApraCudaStream());;
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	copyProps.sync = true;
	auto copy = std::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	H264EncoderNVCodecProps encoderProps(cuContext);
	encoderProps.logHealth = true;
	auto encoder = std::shared_ptr<Module>(new H264EncoderNVCodec(encoderProps));
	copy->setNext(encoder);

	auto sink = std::shared_ptr<Module>(new StatSink());
	encoder->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.run_all_threaded();

	std::this_thread::sleep_for(std::chrono::seconds(20));
	p.stop();
	p.term();

	p.wait_for_all();

}

BOOST_AUTO_TEST_CASE(mono_1920x960_pipeline, *boost::unit_test::disabled()
*utf::precondition(if_h264_encoder_supported()))
{
	auto cuContext = apracucontext_sp(new ApraCUcontext());
	auto width = 1920;
	auto height = 960;

	auto fileReaderProps = FileReaderModuleProps("./data/mono_1920x960.raw");
	fileReaderProps.fps = 10000;
	auto fileReader = std::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	cudastream_sp cudaStream_ = std::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	auto copy = std::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	auto cc = std::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV420, cudaStream_)));
	copy->setNext(cc);

	auto sync = std::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream_)));
	cc->setNext(sync);

	H264EncoderNVCodecProps encoderProps(cuContext);
	encoderProps.logHealth = true;
	auto encoder = std::shared_ptr<Module>(new H264EncoderNVCodec(encoderProps));
	sync->setNext(encoder);

	auto sink = std::shared_ptr<Module>(new StatSink());
	encoder->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.run_all_threaded();

	std::this_thread::sleep_for(std::chrono::seconds(20));

	p.stop();
	p.term();

	p.wait_for_all();

}

BOOST_AUTO_TEST_SUITE_END()
