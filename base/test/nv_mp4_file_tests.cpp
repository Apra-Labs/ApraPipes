#include <boost/test/unit_test.hpp>
#include "Logger.h"
#include "Frame.h"
#include "PipeLine.h"
#include "FileReaderModule.h"
#include "Mp4WriterSink.h"
#include "EncodedImageMetadata.h"
#include "Mp4VideoMetadata.h"
#include "H264Metadata.h"
#include "CudaMemCopy.h"
#include "H264EncoderNVCodec.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(nv_mp4_file_tests)

void run_h264EncoderNV_to_h264writer(bool loop, int sleepTime)
{
	std::string inFolderPath = "./data/Raw_YUV420_640x360/Image???_YUV420.raw";
	std::string outFolderPath = "./data/testOutput/mp4_videos/rgb_24bpp/";
	Test_Utils::deleteFolder(outFolderPath); //make sure this does not exist when we start new test

	auto cuContext = apracucontext_sp(new ApraCUcontext());

	auto width = 640;
	auto height = 360;
	uint32_t gopLength = 30;
	uint32_t bitRateKbps = 17000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::BASELINE;
	bool enableBFrames = 1;


	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1, 4 * 1024 * 1024);
	fileReaderProps.fps = 300;
	fileReaderProps.readLoop = loop;
	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());

	auto copyProps = CudaMemCopyProps(cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStream_);
	copyProps.sync = true;
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	BOOST_TEST(fileReader->setNext(copy));
	auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	BOOST_TEST(copy->setNext(encoder));

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 1, 40, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 1000;
	auto mp4WriterSinkP = boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));
	BOOST_TEST(encoder->setNext(mp4WriterSinkP));


	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	BOOST_TEST(p->init());

	LOG_ERROR << "processing folder <" << inFolderPath << ">";
	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(sleepTime);
	
	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();

};

BOOST_AUTO_TEST_CASE(h264EncoderNV_to_h264writer) 
{
	run_h264EncoderNV_to_h264writer(false, 5);
}
BOOST_AUTO_TEST_CASE(h264EncoderNV_to_h264writer_Chunktime) 
{
	run_h264EncoderNV_to_h264writer(true, 70);
}

BOOST_AUTO_TEST_SUITE_END()