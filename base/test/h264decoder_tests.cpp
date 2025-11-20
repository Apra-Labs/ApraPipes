#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <memory>
#include <thread>
#include <chrono>
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "Logger.h"
#include "H264Decoder.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "ExternalSinkModule.h"
#include "H264Metadata.h"
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "StatSink.h"
#ifdef ARM64
#include "EglRenderer.h"

#else
#include "CudaMemCopy.h"
#include "nv_test_utils.h"
#endif

BOOST_AUTO_TEST_SUITE(h264decoder_tests)

#ifdef ARM64

BOOST_AUTO_TEST_CASE(mp4reader_decoder_eglrenderer,* boost::unit_test::disabled())
{
	Logger::setLogLevel("info");

	// metadata is known
	std::string videoPath = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	auto mp4Reader = std::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto Decoder = std::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	mp4Reader->setNext(Decoder, mImagePin);

	auto sink = std::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	Decoder->setNext(sink);

	std::shared_ptr<PipeLine> p;
	p = std::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(15);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(mp4reader_decoder_extsink)
{
	Logger::setLogLevel("info");

	// metadata is known
	std::string videoPath = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	auto mp4Reader = std::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto Decoder = std::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	mp4Reader->setNext(Decoder, mImagePin);

	auto m3 = std::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	Decoder->setNext(m3);

	std::shared_ptr<PipeLine> p;
	p = std::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(15);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

#else
BOOST_AUTO_TEST_CASE(h264_to_yuv420, *utf::precondition(if_h264_encoder_supported()))
{
	Logger::setLogLevel("info");

	// metadata is known
	auto props = FileReaderModuleProps("./data/h264_data/FVDO_Freeway_4cif_???.H264", 0, -1);
	props.readLoop = false;
	auto fileReader = std::shared_ptr<FileReaderModule>(new FileReaderModule(props));

	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));

	auto rawImagePin = fileReader->addOutputPin(h264ImageMetadata);

	auto Decoder = std::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
	fileReader->setNext(Decoder);

	auto fileWriter = std::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/yuv420Frames/Yuv420_704x576????.raw")));
	Decoder->setNext(fileWriter);
	fileReader->play(true);

	std::shared_ptr<PipeLine> p;
	p = std::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(6);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();

}

BOOST_AUTO_TEST_CASE(encoder_to_decoder, *utf::precondition(if_h264_encoder_supported()))
{
	Logger::setLogLevel("info");
	auto cuContext = apracucontext_sp(new ApraCUcontext());

	auto width = 640;
	auto height = 360;
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::BASELINE;
	bool enableBFrames = true;

	auto fileReader = std::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw", 0, -1)));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	fileReader->addOutputPin(metadata);

	auto cudaStream_ = std::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStream_);
	copyProps.sync = true;
	auto copy = std::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	auto encoder = std::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	copy->setNext(encoder);

	auto Decoder = std::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
	encoder->setNext(Decoder);

	auto m2 = std::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	Decoder->setNext(m2);

	fileReader->play(true);

	std::shared_ptr<PipeLine> p;
	p = std::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);
	
	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(8);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(mp4reader_to_decoder_extSink, *utf::precondition(if_h264_encoder_supported()))
{
	Logger::setLogLevel("info");

	std::string startingVideoPath_2 = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	auto mp4ReaderProps_2 = Mp4ReaderSourceProps(startingVideoPath_2, false, 0, true, false, false);
	mp4ReaderProps_2.logHealth = true;
	mp4ReaderProps_2.logHealthFrequency = 100;
	mp4ReaderProps_2.fps = 30;
	auto mp4Reader_2 = std::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps_2));
	auto h264ImageMetadata_2 = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader_2->addOutPutPin(h264ImageMetadata_2);
	auto mp4Metadata_2 = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader_2->addOutPutPin(mp4Metadata_2);
	// metadata is known

	auto Decoder = std::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
	mp4Reader_2->setNext(Decoder);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = std::shared_ptr<Module>(new StatSink(sinkProps));
	Decoder->setNext(sink);

	std::shared_ptr<PipeLine> p;
	p = std::shared_ptr<PipeLine>(new PipeLine("test"));

	p->appendModule(mp4Reader_2);

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
}

#endif

BOOST_AUTO_TEST_SUITE_END()