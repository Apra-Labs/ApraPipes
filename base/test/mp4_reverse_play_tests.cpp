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
#include "StatSink.h"

#include "FrameContainerQueue.h"

BOOST_AUTO_TEST_SUITE(mp4_reverse_play)

class TestModule : public Module
{
public:
	TestModule(Kind nature, string name, ModuleProps props) : Module(nature, name, props)
	{

	}

	virtual ~TestModule() {}

	size_t getNumberOfOutputPins() { return Module::getNumberOfOutputPins(); }
	size_t getNumberOfInputPins() { return Module::getNumberOfInputPins(); }
	framemetadata_sp getFirstInputMetadata() { return Module::getFirstInputMetadata(); }
	framemetadata_sp getFirstOutputMetadata() { return Module::getFirstOutputMetadata(); }
	metadata_by_pin& getInputMetadata() { return Module::getInputMetadata(); }
	framemetadata_sp getInputMetadataByType(int type) { return Module::getInputMetadataByType(type); }
	framemetadata_sp getOutputMetadataByType(int type) { return Module::getOutputMetadataByType(type); }
	int getNumberOfInputsByType(int type) { return Module::getNumberOfInputsByType(type); }
	int getNumberOfOutputsByType(int type) { return Module::getNumberOfOutputsByType(type); }
	bool isMetadataEmpty(framemetadata_sp& metadata) { return Module::isMetadataEmpty(metadata); }
	bool isFrameEmpty(frame_sp& frame) { return Module::isFrameEmpty(frame); }
	string getInputPinIdByType(int type) { return Module::getInputPinIdByType(type); }
	string getOutputPinIdByType(int type) { return Module::getOutputPinIdByType(type); }

	void addInputPin(framemetadata_sp& metadata, string& pinId) { return Module::addInputPin(metadata, pinId); } // throws exception if validation fails
	Connections getConnections() { return Module::getConnections(); }

	boost_deque<frame_sp> getFrames(frame_container& frames) { return Module::getFrames(frames); }

	frame_sp makeFrame(size_t size, string pinId) { return Module::makeFrame(size, pinId); }
	frame_sp makeFrame(size_t size) { return Module::makeFrame(size); }

	bool send(frame_container& frames) { return Module::send(frames); }

	boost::shared_ptr<FrameContainerQueue> getQue() { return Module::getQue(); }
	frame_sp getFrameByType(frame_container& frames, int frameType) { return Module::getFrameByType(frames, frameType); }

	ModuleProps getProps() { return Module::getProps(); }
	void setProps(ModuleProps& props) { return Module::setProps(props); }
	void fillProps(ModuleProps& props) { return Module::fillProps(props); }

	bool processSourceQue() { return Module::processSourceQue(); }
	bool handlePausePlay(bool play) { return Module::handlePausePlay(play); }
	bool getPlayState() { return Module::getPlayState(); }
};

class TestModuleProps : public ModuleProps
{
public:
	TestModuleProps() :ModuleProps()
	{
	}
	TestModuleProps(int fps, size_t qlen, bool logHealth) : ModuleProps(fps, qlen, logHealth)
	{
	}
	~TestModuleProps()
	{}
};
class TestModule1 : public TestModule
{
public:
	TestModule1(TestModuleProps _props) : TestModule(SINK, "TestModule1", _props)
	{

	}

	virtual ~TestModule1() {}

protected:
	bool validateInputPins() { return true; }
};

struct SetupPlaybackTests
{
	SetupPlaybackTests(std::string videoPath,
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
		auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
		mp4Reader->addOutPutPin(encodedImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_2_0"));
		mp4Reader->addOutPutPin(mp4Metadata);

		TestModuleProps sinkProps;// (30, 100, true);
		//sinkProps.logHealth = false;
		sinkProps.logHealthFrequency = 1;
		sink = boost::shared_ptr<TestModule1>(new TestModule1(sinkProps));
		mp4Reader->setNext(sink);

		BOOST_TEST(mp4Reader->init());
		BOOST_TEST(sink->init());
	}

	boost::shared_ptr<Mp4ReaderSource> mp4Reader;
	boost::shared_ptr<TestModule1> sink = nullptr;
};

BOOST_AUTO_TEST_CASE(fwd)
{
	std::string videoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895288956.mp4";
	SetupPlaybackTests f(videoPath, 0, true, true);

	int ct = 0, total = 601;
	while (ct < total - 1)
	{
		f.mp4Reader->step();
		auto sinkQ = f.sink->getQue();
		auto frames = sinkQ->try_pop();
		auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
		LOG_INFO << "frame->timestamp <" << frame->timestamp << ">";
		ct++;
	}
	f.mp4Reader->step();
	auto sinkQ = f.sink->getQue();
	auto frames = sinkQ->try_pop();
	auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895298961);
	LOG_INFO << "frame->timestamp <" << frame->timestamp << ">";

	// new video open
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655919060000);
}

BOOST_AUTO_TEST_CASE(switch_playback)
{
	std::string videoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895288956.mp4";
	SetupPlaybackTests f(videoPath, 0, true, true);

	f.mp4Reader->step();
	auto sinkQ = f.sink->getQue();
	auto frames = sinkQ->try_pop();
	auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895288956);

	LOG_INFO << "changing playback <fwd->bwd>";
	f.mp4Reader->changePlayback(1, false);
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895288956);

	// new video open + new file parse happens 
	LOG_INFO << "new video opens";
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895165230);

	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895165215);

	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895165200);

	LOG_INFO << "chaning playback <bwd->fwd>";
	f.mp4Reader->changePlayback(1, true);
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895165200);

	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895165215);

	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895165230);

	// new video open
	LOG_INFO << "new video opens<><>";
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895288956);
	LOG_INFO << "1 frame->timestamp <" << frame->timestamp << ">";

	int nFramesInOpenVideo = 601, count = 1;
	while (count < nFramesInOpenVideo - 1)
	{
		f.mp4Reader->step();
		frames = sinkQ->try_pop();
		frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
		LOG_TRACE << "frameIdx/total <" << count << "/" << nFramesInOpenVideo << ">";
		LOG_TRACE << "frame->timestamp <" << frame->timestamp << ">";
		++count;
	}
	// last frame of open video
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895298961);

	// new video open
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655919060000);
}

BOOST_AUTO_TEST_CASE(video_coverage)
{
	std::string videoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0023/1655919060000.mp4";
	SetupPlaybackTests f(videoPath, 0, true, true);

	/* forward playback verification */
	f.mp4Reader->step();
	auto sinkQ = f.sink->getQue();
	auto frames = sinkQ->try_pop();
	auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655919060000);

	int nFramesInOpenVideo = 1270, count = 1;
	while (count < nFramesInOpenVideo - 1)
	{
		f.mp4Reader->step();
		frames = sinkQ->try_pop();
		frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
		LOG_TRACE << "frameIdx/total <" << count << "/" << nFramesInOpenVideo << ">";
		LOG_TRACE << "frame->timestamp <" << frame->timestamp << ">";
		++count;
	}
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == (1655919060000 + 21136));

	/* backward playback verification */
	LOG_INFO << "changing playback <fwd->bwd>";
	f.mp4Reader->changePlayback(1, false);
	f.mp4Reader->step();
	sinkQ = f.sink->getQue();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == (1655919060000 + 21136));

	f.mp4Reader->step();
	sinkQ = f.sink->getQue();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == (1655919060000 + 21120));

	nFramesInOpenVideo = 1270, count = 2;
	while (count < nFramesInOpenVideo - 1)
	{
		f.mp4Reader->step();
		frames = sinkQ->try_pop();
		frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
		LOG_TRACE << "frameIdx/total <" << count << "/" << nFramesInOpenVideo << ">";
		LOG_TRACE << "frame->timestamp <" << frame->timestamp << ">";
		++count;
	}
	// first frame
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655919060000);

	// new (prev) video open
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895298961);
}

BOOST_AUTO_TEST_CASE(seek_in_revPlayback_prev_hr)
{
	std::string videoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0023/1655919060000.mp4";
	bool direction = false;

	SetupPlaybackTests f(videoPath, 0, direction, true);
	auto sinkQ = f.sink->getQue();

	// last frame
	f.mp4Reader->step();
	auto frames = sinkQ->try_pop();
	auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655919060000 + 21136);

	// 2nd frame is at 1655919060015
	f.mp4Reader->randomSeek(1655919060009, false);
	f.mp4Reader->step();

	// first frame
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655919060000);

	// new (prev) video open
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655895298961);
}

BOOST_AUTO_TEST_CASE(seek_in_revPlayback_prev_day)
{
	std::string videoPath = "data/Mp4_videos/mp4_seek_tests/20220523/0001/1655926320000.mp4";
	bool direction = false;

	SetupPlaybackTests f(videoPath, 0, direction, true);
	auto sinkQ = f.sink->getQue();

	// last frame
	f.mp4Reader->step();
	auto frames = sinkQ->try_pop();
	auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655926320000 + 59980);

	f.mp4Reader->randomSeek(1655926320009, false);
	f.mp4Reader->step();

	// first frame
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655926320000);

	// last frame of the new (prev) video open 
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655919060000 + 21136);
}

BOOST_AUTO_TEST_CASE(seek_in_revPlay_prev_hr)
{
	std::string videoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	bool direction = false;

	SetupPlaybackTests f(videoPath, 0, direction, true);
	auto sinkQ = f.sink->getQue();

	// last frame
	f.mp4Reader->step();
	auto frames = sinkQ->try_pop();
	auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame != nullptr);
	//BOOST_TEST(frame->timestamp == 1655919060000 + 21136);

	f.mp4Reader->randomSeek(1655895299961, false);
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame != nullptr);
	BOOST_TEST(frame->timestamp == 1655895298961);
}

BOOST_AUTO_TEST_CASE(seek_in_revPlay_fail_to_seek_infile_restore)
{
	std::string videoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0023/1655919060000.mp4";
	bool direction = false;

	SetupPlaybackTests f(videoPath, 0, direction, true);
	auto sinkQ = f.sink->getQue();

	// last frame
	f.mp4Reader->step();
	auto frames = sinkQ->try_pop();
	auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame != nullptr);
	BOOST_TEST(frame->timestamp == 1655919060000 + 21136);

	// nothing further on disk
	f.mp4Reader->randomSeek(1655895162000, false);
	f.mp4Reader->step();
	frames = sinkQ->pop();
	BOOST_TEST(frames.begin()->second->isEOS());
	auto eosFrame = dynamic_cast<EoSFrame*>(frames.begin()->second.get());
	auto type = eosFrame->getEoSFrameType();
	BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_SEEK_EOS);

	// last frame of the new (prev) video open 
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1655919060000 + 21120);
}

void printCache(std::map<std::string, std::pair<uint64_t, uint64_t> >& snap)
{
	LOG_INFO << "============printing cache==============";
	for (auto it = snap.begin(); it != snap.end(); ++it)
	{
		LOG_INFO << it->first << ": <" << it->second.first << "> <" << it->second.second << ">";
	}
	LOG_INFO << "============printing cache FIN==============";
}

BOOST_AUTO_TEST_CASE(seek_dir_change_trig_fresh_parse)
{
	std::string videoPath = "data/Mp4_videos/mp4_seek_tests/20220523/0001/1655926320000.mp4";
	bool direction = true;

	SetupPlaybackTests f(videoPath, 0, direction, true);
	auto sinkQ = f.sink->getQue();

	// last frame // first
	f.mp4Reader->step();
	auto frames = sinkQ->try_pop();
	auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame != nullptr);
	//BOOST_TEST(frame->timestamp == 1655919060000 + 21136);

	// fourth/last
	f.mp4Reader->randomSeek(1655926320000 + 5, false);
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame != nullptr);
	BOOST_TEST(frame->timestamp == 1655926320016);

	auto snap = f.mp4Reader->getCacheSnapShot();
	printCache(snap);

	// bwd seek -- first
	f.mp4Reader->play(1, false);
	f.mp4Reader->randomSeek(1655895162221 + 2, false);
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame != nullptr);
	BOOST_TEST(frame->timestamp == 1655895162221);

	snap = f.mp4Reader->getCacheSnapShot();
	printCache(snap);

	// change direction - fwd - seek into --second
	f.mp4Reader->play(1, true);
	f.mp4Reader->randomSeek(1655895288956, false); // use play
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame != nullptr);
	BOOST_TEST(frame->timestamp == 1655895288956);

	snap = f.mp4Reader->getCacheSnapShot();
	printCache(snap);

	// change direction - bwd - seek into --second
	f.mp4Reader->play(1, false);
	f.mp4Reader->randomSeek(1655895288956 + 10, false); // use play
	f.mp4Reader->step();
	frames = sinkQ->try_pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame != nullptr);
	BOOST_TEST(frame->timestamp == 1655895288956);
}

BOOST_AUTO_TEST_CASE(step_only_parse_disabled_video_cov_with_reinitInterval)
{
	/*
		video coverage test i.e. [st -> end (eof) | direction change | end -> st (eof)]
		parse disabled
		using only step
		with reinitInterval
	*/
	std::string videoPath = "data/Mp4_videos/mp4_seek_tests/apra.mp4";
	SetupPlaybackTests f(videoPath, 10, true, false);

	/* forward playback verification */
	f.mp4Reader->step();
	auto sinkQ = f.sink->getQue();
	auto frames = sinkQ->try_pop();
	auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(frame->timestamp == 1673855454254);

	while (1)
	{
		f.mp4Reader->step();
		frames = sinkQ->try_pop();
		frame = frames.begin()->second;
		if (frame->isEOS())
		{
			auto eosFrame = dynamic_cast<EoSFrame*>(frame.get());
			BOOST_TEST(eosFrame->getEoSFrameType() == EoSFrame::EoSFrameType::MP4_PLYB_EOS);
			break;
		}
	}
	LOG_INFO << "Reached EOF !";

	/* backward playback verification */
	LOG_INFO << "changing playback <fwd->bwd>";
	uint64_t lastFrameTS = 0;
	f.mp4Reader->changePlayback(1, false);
	while (1)
	{
		f.mp4Reader->step();
		frames = sinkQ->try_pop();
		frame = frames.begin()->second;
		if (frame->isEOS())
		{
			auto eosFrame = dynamic_cast<EoSFrame*>(frame.get());
			BOOST_TEST(eosFrame->getEoSFrameType() == EoSFrame::EoSFrameType::MP4_PLYB_EOS);
			break;
		}
		else
		{
			lastFrameTS = frame->timestamp;
		}
	}
	LOG_INFO << "Reached EOF !";
	BOOST_TEST(lastFrameTS == 1673855454254);
}

BOOST_AUTO_TEST_CASE(temp, *boost::unit_test::disabled())
{
	std::string videoPath = "data/Mp4_videos/mp4_pipeline_tests_new/20230319/0015/1681898006457.mp4";
	bool direction = true;

	SetupPlaybackTests f(videoPath, 0, direction, true);
	auto sinkQ = f.sink->getQue();

	// last frame // first
	/*f.mp4Reader->randomSeek(1681221492923);
	auto frames = sinkQ->try_pop();
	BOOST_TEST((frames.begin()->second->timestamp == 1681221492923));

	f.mp4Reader->randomSeek(1681221492923);
	frames = sinkQ->try_pop();
	BOOST_TEST((frames.begin()->second->timestamp == 1681221492923));*/

	//f.mp4Reader->randomSeek(1681221599000);
	f.mp4Reader->step();
	auto snap = f.mp4Reader->getCacheSnapShot();
	printCache(snap);

	auto frames = sinkQ->try_pop();
	//BOOST_TEST(frames);
	BOOST_TEST((frames.begin()->second->timestamp == 1681898006457));

	while (1)
	{
		f.mp4Reader->step();
		frames = sinkQ->try_pop();
		if (frames.begin()->second->isEOS())
		{
			LOG_ERROR << "END OF STREAM!!!";
			auto eosFrame = dynamic_cast<EoSFrame*>(frames.begin()->second.get());
			auto type = eosFrame->getEoSFrameType();
			BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_PLYB_EOS);
			break;
		}
		LOG_INFO << "Read <" << frames.begin()->second->timestamp << ">";
		if (frames.begin()->second->timestamp == 1681908047057)
		{
			LOG_ERROR << "FOUND!!!! 1681908047057";
			break;
		}
	}

	while (1)
	{
		f.mp4Reader->step();
		frames = sinkQ->try_pop();
		if (frames.begin()->second->isEOS())
		{
			LOG_ERROR << "END OF STREAM!!!";
			auto eosFrame = dynamic_cast<EoSFrame*>(frames.begin()->second.get());
			auto type = eosFrame->getEoSFrameType();
			BOOST_TEST(type == EoSFrame::EoSFrameType::MP4_PLYB_EOS);
			break;
		}
		if (frames.begin()->second->timestamp == 1681908060024)
		{
			LOG_ERROR << "FOUND!!!! 1681908060024";
			break;
		}
	}
}
BOOST_AUTO_TEST_SUITE_END()