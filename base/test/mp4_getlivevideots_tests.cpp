#include <boost/test/unit_test.hpp>
#include "test_utils.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "PipeLine.h"

#include "FileReaderModule.h"
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "EncodedImageMetadata.h"
#include "FrameContainerQueue.h"

BOOST_AUTO_TEST_SUITE(mp4_getlivevideots_tests)

struct SetupSeekTests
{
	SetupSeekTests(std::string &startingVideoPath, int width, int height, int reInitInterval, bool parseFS, bool readLoop, bool getLiveTS = true)
	{
		LoggerProps loggerProps;
		loggerProps.logLevel = boost::log::trivial::severity_level::info;
		Logger::initLogger(loggerProps);

		auto mp4ReaderProps = Mp4ReaderSourceProps(startingVideoPath, parseFS, reInitInterval, true, readLoop, getLiveTS);
		mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(width, height));
		auto encodedImagePin = mp4Reader->addOutPutPin(encodedImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_3_0"));
		auto mp4MetadataPin = mp4Reader->addOutPutPin(mp4Metadata);


		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE);

		auto sinkProps = ExternalSinkProps();
		sinkProps.logHealth = true;
		sinkProps.logHealthFrequency = 1000;
		sink = boost::shared_ptr<ExternalSink>(new ExternalSink(sinkProps));
		mp4Reader->setNext(sink, mImagePin);

		auto p = boost::shared_ptr<PipeLine>(new PipeLine("mp4reader"));
		p->appendModule(mp4Reader);

		BOOST_TEST(mp4Reader->init());
		BOOST_TEST(sink->init());
	}

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
		bool process(frame_container &frames)
		{
			//LOG_ERROR << "ExternalSinkProcess <>";
			for (const auto &pair : frames)
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
};

uint64_t getCurrentTS()
{
	std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now();
	auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());
	uint64_t currentTime = dur.count();
	return currentTime;
}

BOOST_AUTO_TEST_CASE(seek_read_loop)
{
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/apra.mp4";
	int width = 22, height = 30;
	bool parseFS = false;
	bool readLoop = true;
	SetupSeekTests s(startingVideoPath, width, height, 0, parseFS, readLoop);

	auto currentTS = getCurrentTS();

	/* process one frame */
	s.mp4Reader->step();
	auto frames = s.sink->pop();

	auto imgFrame = frames.begin()->second;
	// first frame in the video is 1673855454254
	BOOST_TEST(imgFrame->timestamp >= currentTS);

	currentTS = getCurrentTS();
	uint64_t skipTS = 1673855454000;
	bool ret = s.mp4Reader->randomSeek(skipTS);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp >= currentTS);

	// seek to last frame
	currentTS = getCurrentTS();
	skipTS = 1673855456243;
	ret = s.mp4Reader->randomSeek(skipTS);
	BOOST_TEST(ret == true);
	s.mp4Reader->step();
	frames = s.sink->pop();

	imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp >= currentTS);

	currentTS = getCurrentTS();
	// read loop should not give EOS - it should give first frame again
	s.mp4Reader->step();
	frames = s.sink->pop();
	
	imgFrame = frames.begin()->second;
	// first frame in the video is 1673855454254
	BOOST_TEST(imgFrame->timestamp >= currentTS);
}

BOOST_AUTO_TEST_SUITE_END()