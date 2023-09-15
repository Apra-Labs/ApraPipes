
#include <boost/test/unit_test.hpp>
#include "test_utils.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "PipeLine.h"

#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "EncodedImageMetadata.h"
#include "FrameContainerQueue.h"
#include "H264Metadata.h"

#ifdef APRA_CUDA_ENABLED 
#include "CudaCommon.h"
#include "H264Decoder.h"
#endif

BOOST_AUTO_TEST_SUITE(mp4_seek_tests)

struct SetupSeekTests
{
	SetupSeekTests(std::string& startingVideoPath, int reInitInterval, bool parseFS, bool readLoop, FrameMetadata::FrameType frameType)
	{
		framemetadata_sp imageMetadata;
		LoggerProps loggerProps;
		loggerProps.logLevel = boost::log::trivial::severity_level::info;
		Logger::initLogger(loggerProps);

		auto mp4ReaderProps = Mp4ReaderSourceProps(startingVideoPath, parseFS, reInitInterval, true, readLoop, false);
		mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		if (frameType == FrameMetadata::FrameType::ENCODED_IMAGE)
		{
			imageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
		}
		else if(frameType == FrameMetadata::FrameType::H264_DATA)
		{
			imageMetadata = framemetadata_sp(new H264Metadata(0, 0));
		}
		auto encodedImagePin = mp4Reader->addOutPutPin(imageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_2_0"));
		auto mp4MetadataPin = mp4Reader->addOutPutPin(mp4Metadata);

		auto sinkProps = ExternalSinkProps();
		sinkProps.logHealth = true;
		sinkProps.logHealthFrequency = 1000;
		sink = boost::shared_ptr<ExternalSink>(new ExternalSink(sinkProps));
		mp4Reader->setNext(sink, true, false);

		auto p = boost::shared_ptr<PipeLine>(new PipeLine("mp4reader"));
		p->appendModule(mp4Reader);

		BOOST_TEST(mp4Reader->init());
		BOOST_TEST(sink->init());
	}

#ifdef APRA_CUDA_ENABLED 
	SetupSeekTests(std::string videoPath,
		bool reInitInterval, bool direction, bool parseFS)
	{
		LoggerProps loggerProps;
		loggerProps.logLevel = boost::log::trivial::severity_level::info;
		Logger::initLogger(loggerProps);

		bool readLoop = false;
		auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, reInitInterval, direction, readLoop, false);
		mp4ReaderProps.logHealth = true;
		mp4ReaderProps.logHealthFrequency = 1000;
		mp4ReaderProps.fps = 100;
		mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

		auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
		mp4Reader->addOutPutPin(h264ImageMetadata);

		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_2_0"));
		mp4Reader->addOutPutPin(mp4Metadata);

		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::H264_DATA);

		decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
		mp4Reader->setNext(decoder, mImagePin);

		auto sinkProps = ExternalSinkProps();
		sinkProps.logHealth = true;
		sinkProps.logHealthFrequency = 1000;
		sink = boost::shared_ptr<ExternalSink>(new ExternalSink(sinkProps));
		decoder->setNext(sink);

		auto p = boost::shared_ptr<PipeLine>(new PipeLine("mp4reader"));
		p->appendModule(mp4Reader);

		BOOST_TEST(mp4Reader->init());
		BOOST_TEST(decoder->init());
		BOOST_TEST(sink->init());
	}
#endif

	~SetupSeekTests()
	{
		mp4Reader->term();
		sink->term();
	}

	uint64_t getTSFromFileName(std::string videoPath)
	{
		std::string videoFileName = boost::filesystem::path(videoPath).filename().string();
		uint64_t ts = std::stoull(videoFileName.substr(0, videoFileName.find(".")));
		return ts;
	}

	class ExternalSinkProps : public ModuleProps
	{
	public:
		ExternalSinkProps() : ModuleProps()
		{
		}
	};
	class ExternalSink : public Module
	{
	public:
		ExternalSink(ExternalSinkProps props) : Module(SINK, "ExternalSink", props)
		{
		}

		frame_container pop()
		{
			return Module::pop();
		}

		boost::shared_ptr<FrameContainerQueue> getQue()
		{
			return Module::getQue();
		}

		~ExternalSink()
		{
		}

	protected:
		bool process(frame_container& frames)
		{
			//LOG_ERROR << "ExternalSinkProcess <>";
			for (const auto& pair : frames)
			{
				LOG_INFO << pair.first << "," << pair.second;
			}

			auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::RAW_IMAGE);
			if (frame)
				LOG_INFO << "Timestamp <" << frame->timestamp << ">";
			// raise an event for the view
			//(*mDataHandler)();

			return true;
		}

		bool validateInputPins()
		{
			return true;
		}

		bool validateInputOutputPins()
		{
			return true;
		}

	}; // ExternalSink
	boost::shared_ptr<PipeLine> p = nullptr;
	boost::shared_ptr<Mp4ReaderSource> mp4Reader;
	boost::shared_ptr<ExternalSink> sink;
#ifdef APRA_CUDA_ENABLED 
	boost::shared_ptr<H264Decoder> decoder;
#endif
};

BOOST_AUTO_TEST_CASE(no_seek)
{
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	s.mp4Reader->step();
	auto frames = s.sink->pop();
	auto imgFrame = frames.begin()->second;
	// first frame ts should be same as filename
	auto ts = s.getTSFromFileName(startingVideoPath);
	BOOST_TEST(imgFrame->timestamp == ts);
}

BOOST_AUTO_TEST_CASE(seek_in_current_file)
{
	/* video length is 3 seconds */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	s.mp4Reader->step();

	auto frames = s.sink->pop();
	auto imgFrame = frames.begin()->second;

	// seek 1 sec inside the file currently being read by mp4Reader
	uint64_t skipTS = 1655895163221;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655895163229);
	LOG_INFO << "Found next available frame " << imgFrame->timestamp - skipTS << " msecs later from skipTS";

	// lets check the next frame also - used in seek_eof_reset_state
	s.mp4Reader->step();
	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655895163245);
	LOG_INFO << "Next frame in sequence " << imgFrame->timestamp - 1655895163229 << " msecs later from last frame";
}

BOOST_AUTO_TEST_CASE(seek_in_next_file)
{
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();
	
	auto imgFrame = frames.begin()->second;

	/* ts of first frame of next file is 1655895288956 - video length 10secs.
	   Seek 5 sec inside the file which is next to the currently open file. */
	uint64_t skipTS = 1655895293956;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655895293966);
	LOG_INFO << "Found next available frame " << imgFrame->timestamp - skipTS << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(seek_in_file_in_next_hr)
{
	/* seek to frame inside the file in next hr */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	/* process one frame */
	s.mp4Reader->step();
	
	auto frames = s.sink->pop();
	auto imgFrame = frames.begin()->second;

	/* ts of first frame of seeked file is 1655919060000, video length 21secs.
	   Seek 20 sec inside this file. */
	uint64_t skipTS = 1655919080000;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655919080010);
	LOG_INFO << "Found next available frame " << imgFrame->timestamp - skipTS << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(seek_in_file_in_next_day)
{
	/* seek to frame inside the file in next day */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	/* process one frame */
	s.mp4Reader->step();
	
	auto frames = s.sink->pop();
	
	auto imgFrame = frames.begin()->second;

	/* ts of first frame of seeked file is 1655926320000, video length 60secs.
	   Seek 57 sec inside this file. */
	uint64_t skipTS = 1655926377000;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655926377014);
	LOG_INFO << "Found next available frame " << imgFrame->timestamp - skipTS << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(seek_fails_no_reset)
{
	/* the last video of the same hour as the skipTS is not long enough
	 expectation is that seek will fail and continue to find the next available frame since its not EOF */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	/* process one frame */
	s.mp4Reader->step();
	
	auto frames = s.sink->pop();
	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	uint64_t skipTS = 1655895400956;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	
	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	// expectation - the seeked to ts is the ts of the first frame of next video
	BOOST_TEST(imgFrame->timestamp == 1655919060000);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(hr_missing_next_avl_hr)
{
	/* no recording for the hour exists - move to next available hour */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	LOG_INFO << "current open video before seek <" << s.mp4Reader->getOpenVideoPath() << ">";
	/* process one frame */
	s.mp4Reader->step();
	LOG_INFO << "current open video after step but before seek <" << s.mp4Reader->getOpenVideoPath() << ">";
	
	auto frames = s.sink->pop();

	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	uint64_t skipTS = 1655898288000;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	LOG_INFO << "current video after seek <" << s.mp4Reader->getOpenVideoPath() << ">";
	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655919060000);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(hr_missing_next_avl_day)
{
	/* no recording for the day exists - move to next available day */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	/* process one frame */
	s.mp4Reader->step();
	auto frames = s.sink->pop();

	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	uint64_t skipTS = 1655898288000;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655919060000);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(missing_past_day_seek)
{
	/* no recording for the past day exists - move to next available day i.e. first frame in recordings */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	/* process one frame */
	s.mp4Reader->step();
	auto frames = s.sink->pop();

	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	uint64_t skipTS = 1655805162000;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(seek_fail_eof_reset_state)
{
	/* seek beyond the last frame of of the last video - expectation is that
	seek will fail and reset the state to pre-seek state.*/
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();
	auto imgFrame = frames.begin()->second;

	// first seek 1 sec inside current file
	uint64_t skipTS = 1655895163221;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655895163229);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	// then seek beyond eof
	skipTS = 1655926444000;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step(); // preprocess command + produce
	
	frames = s.sink->pop();
	auto frame = frames.begin();
	BOOST_TEST(frame->second->isEOS());
	auto eosFrame = dynamic_cast<EoSFrame*>(frame->second.get());
	auto type = eosFrame->getEoSFrameType();
	BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_SEEK_EOS);

	// next step should give us the frame from resumed mp4Reader state
	s.mp4Reader->step();
	
	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	// expectation - the seeked to ts is equal to original state before seek i.e. read the next frame in the old sequence
	BOOST_TEST(imgFrame->timestamp == 1655895163245);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	// one more seek
	s.mp4Reader->step();
	
	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655895163260);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(seek_to_last_frame)
{
	/* seek to the exact last frame - next and exact match both */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();
	
	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	// seek to the last frame in the file
	uint64_t skipTS = 1655895165228;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	
	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655895165230);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	// seek to the last frame in the file -- exact timestamp
	skipTS = 1655895165230;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655895165230);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(reach_eof_do_eos_then_seek)
{
	/* seek to last frame of the recordings, after processing it, we should reach EOF */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();
	
	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	uint64_t skipTS = 1655926379960;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655926379980);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
	// reached eof - next step should get us EOS frame
	bool ret = s.mp4Reader->step();
	
	frames = s.sink->pop();
	auto frame = frames.begin();
	BOOST_TEST(frame->second->isEOS());
	auto eosFrame = dynamic_cast<EoSFrame*>(frame->second.get());
	auto type = eosFrame->getEoSFrameType();
	BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_PLYB_EOS);

	// mp4Reader should be allowed to seek though
	// seek should work even after reaching EOF
	skipTS = 1655898288000;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	
	frames = s.sink->pop();
	frame = frames.begin();
	BOOST_TEST(frame->second->isEOS() == false);
	
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655919060000);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(refresh_last_file_on_seek)
{
	/* seek to the exact last video twice - make sure video is reopened every time we seek to last video in cache (fwd only) */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::ENCODED_IMAGE);

	/* process one frame */
	s.mp4Reader->step();
	
	auto frames = s.sink->pop();
	
	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	// seek to the last video in cache
	uint64_t skipTS = 1655926320000;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655926320000);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	// seek to the last video in cache again & make sure it refreshes
	skipTS = 1655926320000 + 10;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	
	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655926320016);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	// seek to second last video and make sure file is not reopened
	skipTS = 1655919060000;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	
	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655919060000);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(seek_with_parseFS_disabled)
{
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/apra.mp4";
	bool parseFS = false;
	SetupSeekTests s(startingVideoPath, 0, parseFS, false, FrameMetadata::ENCODED_IMAGE);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();
	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673855454254);

	/* before first frame - go to first frame */
	uint64_t skipTS = 1673855454000;
	bool ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();
	
	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	// first frame is 1673855454254
	BOOST_TEST(imgFrame->timestamp == 1673855454254);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	/* basic case - bw start and end */
	skipTS = 1673855454900;
	ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();
	
	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673855454900);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	/* beyond EOF - returns true - should resume from before seek state */
	skipTS = 1673855454254 + 5000;
	ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();
	
	frames = s.sink->pop();
	BOOST_TEST(frames.begin()->second->isEOS());
	auto eosFrame = dynamic_cast<EoSFrame*>(frames.begin()->second.get());
	auto type = eosFrame->getEoSFrameType();
	BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_SEEK_EOS);

	//step to get next frame from resumed state
	s.mp4Reader->step();
	
	
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673855454901);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	skipTS = 1673855454254 + 200;
	ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();
	
	
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673855454462);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	/* reach eof */
	uint64_t lastTS = 0;
	while (true)
	{
		s.mp4Reader->step();
		
		
		frames = s.sink->pop();
		if (frames.begin()->second->isEOS())
		{
			auto eosFrame = dynamic_cast<EoSFrame*>(frames.begin()->second.get());
			auto type = eosFrame->getEoSFrameType();
			BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_PLYB_EOS);
			break;
		}
		imgFrame = frames.begin()->second;
		lastTS = imgFrame->timestamp;
	}
	BOOST_TEST((lastTS == 1673855456243));
	LOG_INFO << "Reached EOF!";

	// important: seeking inside this file should allow us to step through it again
	LOG_INFO << "Seeking after reaching EOF!!";
	skipTS = 1673855454462;
	ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();
	
	
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673855454462);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	/* reach eof again */
	lastTS = 0;
	while (true)
	{
		s.mp4Reader->step();
		
		
		frames = s.sink->pop();
		if (frames.begin()->second->isEOS())
		{
			auto eosFrame = dynamic_cast<EoSFrame*>(frames.begin()->second.get());
			auto type = eosFrame->getEoSFrameType();
			BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_PLYB_EOS);
			break;
		}
		imgFrame = frames.begin()->second;
		lastTS = imgFrame->timestamp;
	}
	BOOST_TEST((lastTS == 1673855456243));
	LOG_INFO << "Reached EOF!";
}

BOOST_AUTO_TEST_CASE(read_loop)
{
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/apra.mp4";
	bool parseFS = false;
	bool readLoop = true;
	SetupSeekTests s(startingVideoPath, 0, parseFS, readLoop, FrameMetadata::ENCODED_IMAGE);

	/* process one frame */
	s.mp4Reader->step();
	
	
	auto frames = s.sink->pop();
	
	
	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673855454254);

	/* before first frame - go to first frame */
	uint64_t skipTS = 1673855454000;
	bool ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();
	
	
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	// first frame is 1673855454254
	BOOST_TEST(imgFrame->timestamp == 1673855454254);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	// seek to last frame
	skipTS = 1673855456243;
	ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();
	
	
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673855456243);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	// read loop should not give EOS - it should give first frame again
	s.mp4Reader->step();
	
	
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	// first frame is 1673855454254
	BOOST_TEST(imgFrame->timestamp == 1673855454254);

	// read till end again
	while (1)
	{
		s.mp4Reader->step();
		
		
		frames = s.sink->pop();
		
		imgFrame = frames.begin()->second;
		if (imgFrame->timestamp == 1673855456243)
		{
			break;
		}
	}

	// read loop should not give EOS - it should give first frame again
	s.mp4Reader->step();
	
	
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	// first frame is 1673855454254
	BOOST_TEST(imgFrame->timestamp == 1673855454254);
}

// H264 seek test

BOOST_AUTO_TEST_CASE(seek_in_current_file_h264)
{
	/* video length is 3 seconds */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/20230501/0013/1685604896179.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::H264_DATA);

	s.mp4Reader->step();

	auto frames = s.sink->pop();
	auto imgFrame = frames.begin()->second;

	// seek 1 sec inside the file currently being read by mp4Reader
	uint64_t skipTS = 1685604897179;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604897188);
	LOG_INFO << "Found next available frame " << imgFrame->timestamp - skipTS << " msecs later from skipTS";

	// lets check the next frame also - used in seek_eof_reset_state
	s.mp4Reader->step();
	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604897218);
	LOG_INFO << "Next frame in sequence " << imgFrame->timestamp - 1685604897188 << " msecs later from last frame";
}

BOOST_AUTO_TEST_CASE(seek_in_next_file_h264)
{
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/20230501/0012/1685604318680.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::H264_DATA);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();

	auto imgFrame = frames.begin()->second;

	/* ts of first frame of next file is 1685604361723 - video length 5secs.
	   Seek 3 sec inside the file which is next to the currently open file. */
	uint64_t skipTS = 1685604364723;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604365527);
	LOG_INFO << "Found next available frame " << imgFrame->timestamp - skipTS << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(seek_in_file_in_next_hr_h264)
{
	/* seek to frame inside the file in next hr */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/20230501/0012/1685604318680.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::H264_DATA);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();
	auto imgFrame = frames.begin()->second;

	/* ts of first frame of seeked file is 1655919060000, video length 3secs.
	   Seek 1 sec inside this file. */
	uint64_t skipTS = 1685604897179;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604897188);
	LOG_INFO << "Found next available frame " << imgFrame->timestamp - skipTS << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(seek_in_file_in_next_day_h264)
{
	/* seek to frame inside the file in next day */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/20230111/0012/1673420640350.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::H264_DATA);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();

	auto imgFrame = frames.begin()->second;

	/* ts of first frame of seeked file is 1685604318680, video length 60secs.
	   Seek 3 sec inside this file. */
	uint64_t skipTS = 1685604321680;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604322484);
	LOG_INFO << "Found next available frame " << imgFrame->timestamp - skipTS << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(seek_fails_no_reset_h264)
{
	/* the last video of the same hour as the skipTS is not long enough
	 expectation is that seek will fail and continue to find the next available frame since its not EOF */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/20230501/0012/1685604318680.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::H264_DATA);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();
	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	uint64_t skipTS = 1685604391723;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	// expectation - the seeked to ts is the ts of the first frame of next video
	BOOST_TEST(imgFrame->timestamp == 1685604896179);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(hr_missing_next_avl_hr_h264)
{
	/* no recording for the hour exists - move to next available hour */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/20230501/0012/1685604318680.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::H264_DATA);

	LOG_INFO << "current open video before seek <" << s.mp4Reader->getOpenVideoPath() << ">";
	/* process one frame */
	s.mp4Reader->step();
	LOG_INFO << "current open video after step but before seek <" << s.mp4Reader->getOpenVideoPath() << ">";

	auto frames = s.sink->pop();

	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	uint64_t skipTS = 1685604395723;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();
	LOG_INFO << "current video after seek <" << s.mp4Reader->getOpenVideoPath() << ">";
	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604896179);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(missing_past_day_seek_h264)
{
	/* no recording for the past day exists - move to next available day i.e. first frame in recordings */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/20230111/0012/1673420640350.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::H264_DATA);

	/* process one frame */
	s.mp4Reader->step();
	auto frames = s.sink->pop();

	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	uint64_t skipTS = 1673350540350;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(seek_fail_eof_reset_state_h264)
{
	/* seek beyond the last frame of of the last video - expectation is that
	seek will fail and reset the state to pre-seek state.*/
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/20230501/0012/1685604318680.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::H264_DATA);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();
	auto imgFrame = frames.begin()->second;

	// first seek 1 sec inside current file
	uint64_t skipTS = 1685604319680;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604319692);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	// then seek beyond eof
	skipTS = 1685605896179;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step(); // preprocess command + produce

	frames = s.sink->pop();
	auto frame = frames.begin();
	BOOST_TEST(frame->second->isEOS());
	auto eosFrame = dynamic_cast<EoSFrame*>(frame->second.get());
	auto type = eosFrame->getEoSFrameType();
	BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_SEEK_EOS);

	// next step should give us the frame from resumed mp4Reader state
	s.mp4Reader->step();

	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	// expectation - the seeked to ts is equal to original state before seek i.e. read the next frame in the old sequence
	BOOST_TEST(imgFrame->timestamp == 1685604319721);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	// one more seek
	s.mp4Reader->step();

	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604319753);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(seek_to_last_frame_h264)
{
	/* seek to the exact last frame - next and exact match both */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/20230501/0012/1685604361723.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::H264_DATA);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();

	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	// seek to the last frame in the file , as the last frame in the file is P frame , it opens the next posisble video
	uint64_t skipTS = 1685604368100;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604896179);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(reach_eof_do_eos_then_seek_h264)
{
	/* seek to last frame of the recordings, after processing it, we should reach EOF */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/20230501/0012/1685604318680.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::H264_DATA);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();

	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	uint64_t skipTS = 1685604898878;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604898979);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
	for(int i = 0 ; i < 7; i++)
	{ 
	s.mp4Reader->step();

	frames = s.sink->pop();
	}
	
	// reached eof - next step should get us EOS frame
	auto frame = frames.begin();
	BOOST_TEST(frame->second->isEOS());
	auto eosFrame = dynamic_cast<EoSFrame*>(frame->second.get());
	auto type = eosFrame->getEoSFrameType();
	BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_PLYB_EOS);

	// mp4Reader should be allowed to seek though
	// seek should work even after reaching EOF
	skipTS = 1685604898878;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();
	frame = frames.begin();
	BOOST_TEST(frame->second->isEOS() == false);

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604898979);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(refresh_last_file_on_seek_h264)
{
	/* seek to the exact last video twice - make sure video is reopened every time we seek to last video in cache (fwd only) */
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/20230111/0012/1673420640350.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, false, FrameMetadata::H264_DATA);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();

	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == s.getTSFromFileName(startingVideoPath));

	// seek to the last video in cache
	uint64_t skipTS = 1685604896179;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604896179);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	// seek to the last video in cache again & make sure it refreshes
	skipTS = 1685604896179 + 100;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604897188);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	// seek to second last video and make sure file is not reopened
	skipTS = 1685604899000;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->step();

	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1685604898979);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";
}

BOOST_AUTO_TEST_CASE(seek_with_parseFS_disabled_h264)
{
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/apraH264.mp4";
	bool parseFS = false;
	SetupSeekTests s(startingVideoPath, 0, parseFS, false, FrameMetadata::H264_DATA);

	/* process one frame */
	s.mp4Reader->step();

	auto frames = s.sink->pop();
	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673420640350);

	/* before first frame - go to first frame */
	uint64_t skipTS = 1673420640000;
	bool ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();

	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	// first frame is 1673420640350
	BOOST_TEST(imgFrame->timestamp == 1673420640350);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	/* basic case - bw start and end */
	skipTS = 1673420640950;
	ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();

	frames = s.sink->pop();
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673420642668);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	/* beyond EOF - returns true - should resume from before seek state */
	skipTS = 1673420640350 + 6000;
	ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();

	frames = s.sink->pop();
	BOOST_TEST(frames.begin()->second->isEOS());
	auto eosFrame = dynamic_cast<EoSFrame*>(frames.begin()->second.get());
	auto type = eosFrame->getEoSFrameType();
	BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_SEEK_EOS);

	//step to get next frame from resumed state
	s.mp4Reader->step();


	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673420642684);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	skipTS = 1673420640350 + 200;
	ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();


	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673420642668);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	/* reach eof */
	uint64_t lastTS = 0;
	while (true)
	{
		s.mp4Reader->step();


		frames = s.sink->pop();
		if (frames.begin()->second->isEOS())
		{
			auto eosFrame = dynamic_cast<EoSFrame*>(frames.begin()->second.get());
			auto type = eosFrame->getEoSFrameType();
			BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_PLYB_EOS);
			break;
		}
		imgFrame = frames.begin()->second;
		lastTS = imgFrame->timestamp;
	}
	BOOST_TEST((lastTS == 1673420645353));
	LOG_INFO << "Reached EOF!";

	// important: seeking inside this file should allow us to step through it again
	LOG_INFO << "Seeking after reaching EOF!!";
	skipTS = 1673420640550 ;
	ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();


	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673420642668);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	/* reach eof again */
	lastTS = 0;
	while (true)
	{
		s.mp4Reader->step();


		frames = s.sink->pop();
		if (frames.begin()->second->isEOS())
		{
			auto eosFrame = dynamic_cast<EoSFrame*>(frames.begin()->second.get());
			auto type = eosFrame->getEoSFrameType();
			BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_PLYB_EOS);
			break;
		}
		imgFrame = frames.begin()->second;
		lastTS = imgFrame->timestamp;
	}
	BOOST_TEST((lastTS == 1673420645353));
	LOG_INFO << "Reached EOF!";
}

BOOST_AUTO_TEST_CASE(read_loop_h264)
{
	std::string startingVideoPath = "data/Mp4_videos/mp4_seeks_tests_h264/apraH264.mp4";
	bool parseFS = false;
	bool readLoop = true;
	SetupSeekTests s(startingVideoPath, 0, parseFS, readLoop, FrameMetadata::H264_DATA);

	/* process one frame */
	s.mp4Reader->step();


	auto frames = s.sink->pop();


	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673420640350);

	/* before first frame - go to first frame */
	uint64_t skipTS = 1673420640000;
	bool ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();


	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	// first frame is 1673420640350
	BOOST_TEST(imgFrame->timestamp == 1673420640350);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	// seek to last frame
	skipTS = 1673420644850;
	ret = s.mp4Reader->randomSeek(skipTS, false);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();


	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1673420644975);
	LOG_INFO << "Found next available frame " << (int)(imgFrame->timestamp - skipTS) << " msecs later from skipTS";

	s.mp4Reader->step();


	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	// first I Frame is 1673420644990
	BOOST_TEST(imgFrame->timestamp == 1673420644990);

	// read till end again
	while (1)
	{
		s.mp4Reader->step();


		frames = s.sink->pop();

		imgFrame = frames.begin()->second;
		// last frame is 1673420645353
		if (imgFrame->timestamp == 1673420645353)
		{
			break;
		}
	}

	// read loop should not give EOS - it should give first frame again
	s.mp4Reader->step();


	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	// first frame is 1673420640350
	BOOST_TEST(imgFrame->timestamp == 1673420640350);
}

#ifdef APRA_CUDA_ENABLED 
BOOST_AUTO_TEST_CASE(seek_inside_cache_fwd)
{
#ifdef __x86_64__
	auto cuContext = apracucontext_sp(new ApraCUcontext());
		int major=0,minor=0;
		if(!cuContext->getComputeCapability(major,minor))
            return;
		LOG_INFO << "Compute Cap "<<major <<"."<<minor;
		if(major<=5) 
		{
			if(minor<2) //dont support below 5.2 (some tests failed on GTX 860M which is 5.0)
			{
				LOG_ERROR << "Compute Cap should be 5.2 or above";
				return;
			}
		}
#endif
	/* video length is 3 seconds */
	std::string startingVideoPath = "./data/Mp4_videos/h264_reverse_play/20230708/0019/1691502958947.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, true);
	frame_container frames;
	uint64_t frameTs;

	while (frames.empty())
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
	}
	frameTs = frames.begin()->second->timestamp;

	BOOST_TEST(frameTs == 1691502958947);//First frame

	for (int i = 0; i < 90; i++)
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
	}

	uint64_t skipTS = 1691502961947;
	s.mp4Reader->randomSeek(skipTS, false);

	while(!frames.empty())
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
		frameTs = frames.begin()->second->timestamp;
		if (frameTs == 1691502962087)
			break;
	}
	
	BOOST_TEST(frameTs == 1691502962087);//seek frame

	s.decoder->term();
}

BOOST_AUTO_TEST_CASE(seek_outside_cache_fwd)
{
#ifdef __x86_64__
	auto cuContext = apracucontext_sp(new ApraCUcontext());
		int major=0,minor=0;
		if(!cuContext->getComputeCapability(major,minor))
            return;
		LOG_INFO << "Compute Cap "<<major <<"."<<minor;
		if(major<=5) 
		{
			if(minor<2) //dont support below 5.2 (some tests failed on GTX 860M which is 5.0)
			{
				LOG_ERROR << "Compute Cap should be 5.2 or above";
				return;
			}
		}
#endif
	std::string startingVideoPath = "./data/Mp4_videos/h264_reverse_play/20230708/0019/1691502958947.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, true);
	frame_container frames;
	uint64_t frameTs;

	while (frames.empty())
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
	}
	frameTs = frames.begin()->second->timestamp;

	BOOST_TEST(frameTs == 1691502958947);//First frame

	uint64_t skipTS = 1691502963937;
	s.mp4Reader->randomSeek(skipTS, false);

	while (!frames.empty())
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
		frameTs = frames.begin()->second->timestamp;
		if (frameTs == 1691502964160)
			break;
	}

	BOOST_TEST(frameTs == 1691502964160);
	
	s.decoder->term();
}

BOOST_AUTO_TEST_CASE(seek_inside_cache_bwd)
{
#ifdef __x86_64__
	auto cuContext = apracucontext_sp(new ApraCUcontext());
		int major=0,minor=0;
		if(!cuContext->getComputeCapability(major,minor))
            return;
		LOG_INFO << "Compute Cap "<<major <<"."<<minor;
		if(major<=5) 
		{
			if(minor<2) //dont support below 5.2 (some tests failed on GTX 860M which is 5.0)
			{
				LOG_ERROR << "Compute Cap should be 5.2 or above";
				return;
			}
		}
#endif
	std::string startingVideoPath = "./data/Mp4_videos/h264_reverse_play/20230708/0019/1691502958947.mp4";
	SetupSeekTests s(startingVideoPath, 0, false, true);
	frame_container frames;
	uint64_t frameTs;

	for (int i = 0; i < 60; i++)
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
	}

	uint64_t skipTS = 1691503016167;
	s.mp4Reader->randomSeek(skipTS, false);

	while (!frames.empty())
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
		frameTs = frames.begin()->second->timestamp;
		if (frameTs == 1691503017167)
			break;
	}
	BOOST_TEST(frameTs == 1691503017167);//seeked frame
	s.decoder->term();
}

BOOST_AUTO_TEST_CASE(seek_outside_cache_bwd)
{
#ifdef __x86_64__
	auto cuContext = apracucontext_sp(new ApraCUcontext());
		int major=0,minor=0;
		if(!cuContext->getComputeCapability(major,minor))
            return;
		LOG_INFO << "Compute Cap "<<major <<"."<<minor;
		if(major<=5) 
		{
			if(minor<2) //dont support below 5.2 (some tests failed on GTX 860M which is 5.0)
			{
				LOG_ERROR << "Compute Cap should be 5.2 or above";
				return;
			}
		}
#endif
	std::string startingVideoPath = "./data/Mp4_videos/h264_reverse_play/20230708/0019/1691502958947.mp4";
	SetupSeekTests s(startingVideoPath, 0, false, true);
	frame_container frames;
	uint64_t frameTs;

	for (int i = 0; i < 60; i++)
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
	}

	uint64_t skipTS = 1691503002167;
	s.mp4Reader->randomSeek(skipTS, false);

	while(!frames.empty())//Process all the buffered backward frames to get the seeked I frame.
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
		frameTs = frames.begin()->second->timestamp;
		if (frameTs == 1691503001585)
			break;
	}
	BOOST_TEST(frameTs == 1691503001585);//seeked Frame

	s.decoder->term();
}

BOOST_AUTO_TEST_CASE(seek_inside_cache_and_dir_change_to_bwd)
{
#ifdef __x86_64__
	auto cuContext = apracucontext_sp(new ApraCUcontext());
		int major=0,minor=0;
		if(!cuContext->getComputeCapability(major,minor))
            return;
		LOG_INFO << "Compute Cap "<<major <<"."<<minor;
		if(major<=5) 
		{
			if(minor<2) //dont support below 5.2 (some tests failed on GTX 860M which is 5.0)
			{
				LOG_ERROR << "Compute Cap should be 5.2 or above";
				return;
			}
		}
#endif
	std::string startingVideoPath = "./data/Mp4_videos/h264_reverse_play/20230708/0019/1691502958947.mp4";
	SetupSeekTests s(startingVideoPath, 0, true, true);
	frame_container frames;
	uint64_t frameTs;

	while (frames.empty())
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
	}
	frameTs = frames.begin()->second->timestamp;

	BOOST_TEST(frameTs == 1691502958947);//First frame

	for (int i = 0; i < 90; i++)
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
	}

	uint64_t skipTS = 1691502961947;
	s.mp4Reader->randomSeek(skipTS, false);
	s.mp4Reader->changePlayback(1, false);

	while (!frames.empty())
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
		frameTs = frames.begin()->second->timestamp;
		if (frameTs == 1691502962042)
			break;
	}
	BOOST_TEST(frameTs == 1691502962042);

	s.decoder->term();
}

BOOST_AUTO_TEST_CASE(seek_outside_cache_and_dir_change_to_fwd)
{
#ifdef __x86_64__
	auto cuContext = apracucontext_sp(new ApraCUcontext());
		int major=0,minor=0;
		if(!cuContext->getComputeCapability(major,minor))
            return;
		LOG_INFO << "Compute Cap "<<major <<"."<<minor;
		if(major<=5) 
		{
			if(minor<2) //dont support below 5.2 (some tests failed on GTX 860M which is 5.0)
			{
				LOG_ERROR << "Compute Cap should be 5.2 or above";
				return;
			}
		}
#endif
	std::string startingVideoPath = "./data/Mp4_videos/h264_reverse_play/20230708/0019/1691502958947.mp4";
	SetupSeekTests s(startingVideoPath, 0, false, true);
	frame_container frames;
	uint64_t frameTs;

	while (frames.empty())
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
	}
	frameTs = frames.begin()->second->timestamp;

	BOOST_TEST(frameTs == 1691503018158);//First frame

	uint64_t skipTS = 1691502963937;
	s.mp4Reader->changePlayback(1, true);
	s.mp4Reader->randomSeek(skipTS, false);
	
	
	while (!frames.empty())
	{
		s.mp4Reader->step();
		s.decoder->step();
		auto sinkQ = s.sink->getQue();
		frames = sinkQ->try_pop();
		frameTs = frames.begin()->second->timestamp;
		if (frameTs == 1691502964160)
			break;
	}
	BOOST_TEST(frameTs == 1691502964160);//seeked_i_Frame
	s.decoder->term();
}
#endif

BOOST_AUTO_TEST_SUITE_END()