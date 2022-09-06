#include <boost/test/unit_test.hpp>
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Logger.h"
#include "Frame.h"
#include "AIPExceptions.h"
#include "PipeLine.h"
#include "FileWriterModule.h"
#include "test_utils.h"
#include "FileReaderModule.h"
#include "CudaCommon.h"
#include "Mp4WriterSink.h"
#include "FramesMuxer.h"
#include "StatSink.h"
#include "EncodedImageMetadata.h"
#include "Mp4VideoMetadata.h"
#include "H264Metadata.h"
#include "CudaMemCopy.h"
#include "CCNPPI.h"
#include "CudaStreamSynchronize.h"
#include "H264EncoderNVCodec.h"
#include "ResizeNPPI.h"
#include "CudaCommon.h"
#include <ExternalSinkModule.h>
#include <H264FrameUtils.h>

BOOST_AUTO_TEST_SUITE(mp4_file_tests)

BOOST_AUTO_TEST_CASE(h264EncoderNV_to_h264writer)
{
	auto cuContext = apracucontext_sp(new ApraCUcontext());

	auto width = 640;
	auto height = 360;
	uint32_t gopLength = 30;
	uint32_t bitRateKbps = 17000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::BASELINE;
	bool enableBFrames = 1;

	std::string inFolderPath = "./data/Raw_YUV420_640x360/????.raw";
	std::string outFolderPath = "./data/testOutput/mp4_videos/rgb_24bpp/";

	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1, 4 * 1024 * 1024);
	fileReaderProps.fps = 24;
	fileReaderProps.readLoop = false;
	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());

	auto copyProps = CudaMemCopyProps(cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStream_);
	copyProps.sync = true;
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);
	auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	copy->setNext(encoder);

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 1, 40, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 10;
	auto mp4WriterSinkP = boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));
	encoder->setNext(mp4WriterSinkP);


	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	LOG_ERROR << "processing folder <" << inFolderPath << ">";
	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(180));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();

}

BOOST_AUTO_TEST_CASE(h264EncoderNV_to_h264writer_Chunktime)
{
	auto cuContext = apracucontext_sp(new ApraCUcontext());

	auto width = 640;
	auto height = 360;
	uint32_t gopLength = 30;
	uint32_t bitRateKbps = 17000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::HIGH;
	bool enableBFrames = 1;

	std::string inFolderPath = "./data/Raw_YUV420_640x360/????.raw";
	std::string outFolderPath = "./data/testOutput/mp4_videos/rgb_24bpp/";

	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1, 4 * 1024 * 1024);
	fileReaderProps.fps = 24;
	fileReaderProps.readLoop = true;
	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());

	auto copyProps = CudaMemCopyProps(cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStream_);
	copyProps.sync = true;
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);
	auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	copy->setNext(encoder);

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 1, 40, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 10;
	auto mp4WriterSinkP = boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));
	encoder->setNext(mp4WriterSinkP);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);
	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	LOG_ERROR << "processing folder <" << inFolderPath << ">";
	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(180));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}
BOOST_AUTO_TEST_SUITE_END()