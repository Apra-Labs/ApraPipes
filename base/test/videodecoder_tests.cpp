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

#endif

BOOST_AUTO_TEST_SUITE_END()
