#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "Logger.h"
#include "VideoDecoder.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "ExternalSinkModule.h"
#include "H265Metadata.h"
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "StatSink.h"
#include "JPEGEncoderL4TM.h"
#include "MemTypeConversion.h"
#ifdef ARM64
// EglRenderer not linked in this SNAP build (commented out of CMakeLists)
// #include "EglRenderer.h"
#include "ApraEGLDisplay.h"

// Helper macro to skip DMA tests when EGL/DMA is not capable (headless CI)
// isDMACapable() not available in this build — on Jetson hardware DMA is always available
#define SKIP_IF_NO_DMA_CAPABLE() \
    do {} while(0)

#else
#include "CudaMemCopy.h"
#include "nv_test_utils.h"
#endif

BOOST_AUTO_TEST_SUITE(h265decoder_tests)

#ifdef ARM64

BOOST_AUTO_TEST_CASE(mp4reader_h265decoder_eglrenderer,* boost::unit_test::disabled())
{
#if 0 // EglRenderer not linked in SNAP build
	Logger::setLogLevel("info");

	// metadata is known
	std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
	mp4Reader->addOutPutPin(h265ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto Decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
	mp4Reader->setNext(Decoder, mImagePin);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	Decoder->setNext(sink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
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
#endif // EglRenderer not linked in SNAP build
}

BOOST_AUTO_TEST_CASE(mp4reader_h265decoder_extsink)
{
	SKIP_IF_NO_DMA_CAPABLE();
	Logger::setLogLevel("info");

	// metadata is known
	std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
	mp4Reader->addOutPutPin(h265ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto Decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
	mp4Reader->setNext(Decoder, mImagePin);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	Decoder->setNext(m3);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
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

BOOST_AUTO_TEST_CASE(mp4reader_h265decoder_statsink)
{
	SKIP_IF_NO_DMA_CAPABLE();
	Logger::setLogLevel("info");

	std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	mp4ReaderProps.logHealth = true;
	mp4ReaderProps.logHealthFrequency = 100;
	mp4ReaderProps.fps = 30;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
	mp4Reader->addOutPutPin(h265ImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto Decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
	mp4Reader->setNext(Decoder, mImagePin);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	Decoder->setNext(sink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));

	p->appendModule(mp4Reader);

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

BOOST_AUTO_TEST_CASE(h265_decode_save_jpegs)
{
	SKIP_IF_NO_DMA_CAPABLE();
	Logger::setLogLevel("info");

	system("mkdir -p /tmp/h265_frames");

	std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	mp4ReaderProps.fps = 30;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
	mp4Reader->addOutPutPin(h265ImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	// VideoDecoder on ARM64 outputs RGBA DMABUF
	auto Decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
	mp4Reader->setNext(Decoder, mImagePin);

	// DMABUF -> HOST conversion required before CPU-based JPEG encoder
	auto memConv = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST)));
	Decoder->setNext(memConv);

	JPEGEncoderL4TMProps encoderProps;
	encoderProps.quality = 90;
	auto jpegEncoder = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(encoderProps));
	memConv->setNext(jpegEncoder);

	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	jpegEncoder->addOutputPin(encodedImageMetadata);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(
		FileWriterModuleProps("/tmp/h265_frames/frame_????.jpg")));
	jpegEncoder->setNext(fileWriter);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

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

	LOG_INFO << "h265_decode_save_jpegs: check /tmp/h265_frames/ for saved JPEG frames";
}

#else

BOOST_AUTO_TEST_CASE(h265_basic_decode_test, *utf::precondition(if_h264_encoder_supported()))
{
	Logger::setLogLevel("info");

	std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
	mp4Reader->addOutPutPin(h265ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto Decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
	mp4Reader->setNext(Decoder, mImagePin);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/yuv420Frames_h265/Yuv420_640x360????.raw")));
	Decoder->setNext(fileWriter);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

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

BOOST_AUTO_TEST_CASE(mp4reader_h265decoder_extSink, *utf::precondition(if_h264_encoder_supported()))
{
	Logger::setLogLevel("info");

	std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	mp4ReaderProps.logHealth = true;
	mp4ReaderProps.logHealthFrequency = 100;
	mp4ReaderProps.fps = 30;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
	mp4Reader->addOutPutPin(h265ImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto Decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
	mp4Reader->setNext(Decoder, mImagePin);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	Decoder->setNext(sink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));

	p->appendModule(mp4Reader);

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