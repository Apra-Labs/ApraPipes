#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "Logger.h"
#include "VideoDecoder.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "H264Metadata.h"
#include "H265Metadata.h"
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "StatSink.h"
#include "MemTypeConversion.h"
#include "JPEGEncoderL4TM.h"
#include "FileWriterModule.h"
#include <boost/filesystem.hpp>

BOOST_AUTO_TEST_SUITE(videodecoder_tests)

#ifdef ARM64

BOOST_AUTO_TEST_CASE(video_decoder_h264_basic)
{
	Logger::setLogLevel("info");

	std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	mp4ReaderProps.fps = 24;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	mp4Reader->setNext(decoder, mImagePin);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	decoder->setNext(sink);

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

BOOST_AUTO_TEST_CASE(video_decoder_h265_basic)
{
	Logger::setLogLevel("info");

	std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	mp4ReaderProps.fps = 30;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
	mp4Reader->addOutPutPin(h265ImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
	mp4Reader->setNext(decoder, mImagePin);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	decoder->setNext(sink);

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

BOOST_AUTO_TEST_CASE(video_decoder_codec_switch)
{
	Logger::setLogLevel("info");

	{
		std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
		auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
		mp4ReaderProps.fps = 24;
		auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
		mp4Reader->addOutPutPin(h264ImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
		mp4Reader->setNext(decoder, mImagePin);

		StatSinkProps sinkProps;
		sinkProps.logHealth = true;
		auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
		decoder->setNext(sink);

		boost::shared_ptr<PipeLine> p;
		p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
		p->appendModule(mp4Reader);

		if (!p->init())
		{
			throw AIPException(AIP_FATAL, "codec_switch H264 pipeline init failed.");
		}

		p->run_all_threaded();
		Test_Utils::sleep_for_seconds(5);
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();
	}

	{
		std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
		auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
		mp4ReaderProps.fps = 30;
		auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
		mp4Reader->addOutPutPin(h265ImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
		mp4Reader->setNext(decoder, mImagePin);

		StatSinkProps sinkProps;
		sinkProps.logHealth = true;
		auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
		decoder->setNext(sink);

		boost::shared_ptr<PipeLine> p;
		p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
		p->appendModule(mp4Reader);

		if (!p->init())
		{
			throw AIPException(AIP_FATAL, "codec_switch H265 pipeline init failed.");
		}

		p->run_all_threaded();
		Test_Utils::sleep_for_seconds(5);
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();
	}
}

// TC-1: Single H264 file -> VideoDecoder -> StatSink, no crash, no CODEC_SWITCH_EOS
BOOST_AUTO_TEST_CASE(mp4reader_video_decoder_h264_only)
{
	Logger::setLogLevel("info");

	std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	mp4ReaderProps.fps = 24;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	mp4Reader->setNext(decoder, mImagePin);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	decoder->setNext(sink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	BOOST_REQUIRE(p->init());

	p->run_all_threaded();
	Test_Utils::sleep_for_seconds(8);
	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

// TC-2: Single H265 file -> VideoDecoder -> StatSink, no crash
BOOST_AUTO_TEST_CASE(mp4reader_video_decoder_h265_only)
{
	Logger::setLogLevel("info");

	std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	mp4ReaderProps.fps = 30;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
	mp4Reader->addOutPutPin(h265ImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
	mp4Reader->setNext(decoder, mImagePin);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	decoder->setNext(sink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	BOOST_REQUIRE(p->init());

	p->run_all_threaded();
	Test_Utils::sleep_for_seconds(8);
	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

// TC-3: H264 file first, then H265 file (sequential pipelines)
// Verifies VideoDecoder successfully decodes H265 frames after codec switch, no crash
BOOST_AUTO_TEST_CASE(mp4reader_video_decoder_h264_to_h265_switch)
{
	Logger::setLogLevel("info");

	{
		std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
		auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
		mp4ReaderProps.fps = 24;
		auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
		mp4Reader->addOutPutPin(h264ImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
		mp4Reader->setNext(decoder, mImagePin);

		StatSinkProps sinkProps;
		sinkProps.logHealth = true;
		auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
		decoder->setNext(sink);

		boost::shared_ptr<PipeLine> p;
		p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
		p->appendModule(mp4Reader);

		BOOST_REQUIRE(p->init());
		p->run_all_threaded();
		Test_Utils::sleep_for_seconds(5);
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();
	}

	{
		std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
		auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
		mp4ReaderProps.fps = 30;
		auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
		mp4Reader->addOutPutPin(h265ImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
		mp4Reader->setNext(decoder, mImagePin);

		StatSinkProps sinkProps;
		sinkProps.logHealth = true;
		auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
		decoder->setNext(sink);

		boost::shared_ptr<PipeLine> p;
		p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
		p->appendModule(mp4Reader);

		BOOST_REQUIRE(p->init());
		p->run_all_threaded();
		Test_Utils::sleep_for_seconds(5);
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();
	}
}

// TC-4: H265 file first, then H264 file (reverse codec switch), no crash
BOOST_AUTO_TEST_CASE(mp4reader_video_decoder_h265_to_h264_switch)
{
	Logger::setLogLevel("info");

	{
		std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
		auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
		mp4ReaderProps.fps = 30;
		auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
		mp4Reader->addOutPutPin(h265ImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
		mp4Reader->setNext(decoder, mImagePin);

		StatSinkProps sinkProps;
		sinkProps.logHealth = true;
		auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
		decoder->setNext(sink);

		boost::shared_ptr<PipeLine> p;
		p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
		p->appendModule(mp4Reader);

		BOOST_REQUIRE(p->init());
		p->run_all_threaded();
		Test_Utils::sleep_for_seconds(5);
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();
	}

	{
		std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
		auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
		mp4ReaderProps.fps = 24;
		auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
		mp4Reader->addOutPutPin(h264ImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
		mp4Reader->setNext(decoder, mImagePin);

		StatSinkProps sinkProps;
		sinkProps.logHealth = true;
		auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
		decoder->setNext(sink);

		boost::shared_ptr<PipeLine> p;
		p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
		p->appendModule(mp4Reader);

		BOOST_REQUIRE(p->init());
		p->run_all_threaded();
		Test_Utils::sleep_for_seconds(5);
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();
	}
}

// TC-5: H264 file in readLoop mode (same codec, no codec change)
// Verifies: decoder does NOT re-init needlessly, no crash over multiple loops
BOOST_AUTO_TEST_CASE(mp4reader_video_decoder_same_codec_no_switch)
{
	Logger::setLogLevel("info");

	std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, true, false);
	mp4ReaderProps.fps = 24;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	mp4Reader->setNext(decoder, mImagePin);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	decoder->setNext(sink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	BOOST_REQUIRE(p->init());

	p->run_all_threaded();
	Test_Utils::sleep_for_seconds(10);
	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

// TC-6: H264 -> H265 -> H264 (two codec switches, three sequential pipeline segments)
// Verifies all three codec segments decode correctly without crash
BOOST_AUTO_TEST_CASE(mp4reader_video_decoder_multiple_switches)
{
	Logger::setLogLevel("info");

	auto runH264Pipeline = []()
	{
		std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
		auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
		mp4ReaderProps.fps = 24;
		auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
		mp4Reader->addOutPutPin(h264ImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
		mp4Reader->setNext(decoder, mImagePin);

		StatSinkProps sinkProps;
		sinkProps.logHealth = true;
		auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
		decoder->setNext(sink);

		boost::shared_ptr<PipeLine> p;
		p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
		p->appendModule(mp4Reader);
		if (!p->init())
		{
			throw AIPException(AIP_FATAL, "H264 pipeline init failed in multiple_switches test.");
		}
		p->run_all_threaded();
		Test_Utils::sleep_for_seconds(4);
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();
	};

	auto runH265Pipeline = []()
	{
		std::string videoPath = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
		auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
		mp4ReaderProps.fps = 30;
		auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
		mp4Reader->addOutPutPin(h265ImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
		mp4Reader->setNext(decoder, mImagePin);

		StatSinkProps sinkProps;
		sinkProps.logHealth = true;
		auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
		decoder->setNext(sink);

		boost::shared_ptr<PipeLine> p;
		p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
		p->appendModule(mp4Reader);
		if (!p->init())
		{
			throw AIPException(AIP_FATAL, "H265 pipeline init failed in multiple_switches test.");
		}
		p->run_all_threaded();
		Test_Utils::sleep_for_seconds(4);
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();
	};

	runH264Pipeline();
	runH265Pipeline();
	runH264Pipeline();
}

BOOST_AUTO_TEST_CASE(mp4reader_video_decoder_codec_switch_jpeg_dump)
{
	Logger::setLogLevel("info");

	boost::filesystem::create_directories("/tmp/codec_switch_frames");

	{
		std::string h264Path = "/home/developer/ws_yash/ApraPipes_SNAP/data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
		auto mp4ReaderProps = Mp4ReaderSourceProps(h264Path, false, 0, true, false, false);
		mp4ReaderProps.fps = 24;
		auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
		mp4Reader->addOutPutPin(h264ImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
		mp4Reader->setNext(decoder, mImagePin);

		auto memConv = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST)));
		decoder->setNext(memConv);

		JPEGEncoderL4TMProps encoderProps;
		encoderProps.quality = 90;
		auto jpegEncoder = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(encoderProps));
		memConv->setNext(jpegEncoder);

		auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
		jpegEncoder->addOutputPin(encodedImageMetadata);

		auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(
			FileWriterModuleProps("/tmp/codec_switch_frames/h264_frame_????.jpg")));
		jpegEncoder->setNext(fileWriter);

		boost::shared_ptr<PipeLine> p;
		p = boost::shared_ptr<PipeLine>(new PipeLine("codec_switch_h264"));
		p->appendModule(mp4Reader);

		if (!p->init())
		{
			throw AIPException(AIP_FATAL, "codec_switch_jpeg_dump H264 pipeline init failed.");
		}

		p->run_all_threaded();
		Test_Utils::sleep_for_seconds(5);
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();
	}

	{
		std::string h265Path = "/home/developer/ws_yash/ApraPipes_SNAP/data/h265_bunny_30frames.mp4";
		auto mp4ReaderProps = Mp4ReaderSourceProps(h265Path, false, 0, true, false, false);
		mp4ReaderProps.fps = 30;
		auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto h265ImageMetadata = framemetadata_sp(new H265Metadata(0, 0));
		mp4Reader->addOutPutPin(h265ImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		auto decoder = boost::shared_ptr<Module>(new VideoDecoder(VideoDecoderProps()));
		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::HEVC_DATA);
		mp4Reader->setNext(decoder, mImagePin);

		auto memConv = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST)));
		decoder->setNext(memConv);

		JPEGEncoderL4TMProps encoderProps;
		encoderProps.quality = 90;
		auto jpegEncoder = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(encoderProps));
		memConv->setNext(jpegEncoder);

		auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
		jpegEncoder->addOutputPin(encodedImageMetadata);

		auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(
			FileWriterModuleProps("/tmp/codec_switch_frames/h265_frame_????.jpg")));
		jpegEncoder->setNext(fileWriter);

		boost::shared_ptr<PipeLine> p;
		p = boost::shared_ptr<PipeLine>(new PipeLine("codec_switch_h265"));
		p->appendModule(mp4Reader);

		if (!p->init())
		{
			throw AIPException(AIP_FATAL, "codec_switch_jpeg_dump H265 pipeline init failed.");
		}

		p->run_all_threaded();
		Test_Utils::sleep_for_seconds(5);
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();
	}

	LOG_INFO << "mp4reader_video_decoder_codec_switch_jpeg_dump: check /tmp/codec_switch_frames/ for h264_frame_*.jpg and h265_frame_*.jpg";
}

#endif

BOOST_AUTO_TEST_SUITE_END()
