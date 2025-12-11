#include <vector>
#include <string>
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "Logger.h"

#include "Mp4ReaderSource.h"
#include "Mp4WriterSink.h"
#include "StatSink.h"
#include "EncodedImageMetadata.h"
#include "Mp4VideoMetadata.h"
#include "FileWriterModule.h"
#include "H264Metadata.h"
#include "PipeLine.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(mp4_dts_strategy)

void read_write(std::string videoPath, std::string outPath, 
	bool recordedTSBasedDTS, bool parseFS = true, int chunkTime=1)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	bool readLoop = false;
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, readLoop, false);
	mp4ReaderProps.logHealth = true;
	mp4ReaderProps.logHealthFrequency = 300;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::H264_DATA);
    auto mp4WriterSinkProps = Mp4WriterSinkProps(chunkTime, 1, 30, outPath, recordedTSBasedDTS);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 300;
	auto mp4WriterSink = boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));
	mp4Reader->setNext(mp4WriterSink, mImagePin);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}
	p->run_all_threaded();

	std::this_thread::sleep_for(std::chrono::seconds(15));

	p->stop();
	p->term();

	p->wait_for_all();

	p.reset();
}

BOOST_AUTO_TEST_CASE(read_mul_write_one_as_recorded)
{
	// two videos (9sec, 9 sec) with 17hr time gap
	// writes a 17hr 8mins 27sec  sec video , in which first and last 9 secs is playable
	std::string videoPath = "data/Mp4_videos/h264_videos_dts_test/20221010/0012/1668001826042.mp4";
	std::string outPath = "data/testOutput/mp4_videos/outFrames/file_as_rec.mp4";
	bool parseFS = true;

	// write a fixed rate video with no gaps
	bool recordedTSBasedDTS = true;
	read_write(videoPath, outPath, recordedTSBasedDTS, parseFS, UINT32_MAX);

	Test_Utils::deleteFolder(outPath);
}

BOOST_AUTO_TEST_CASE(read_mul_write_one_fixed_rate)
{  
	//  two videos (9sec, 9 sec) with 17hr time gap
	// writes a 15 secs video playable video
    std::string videoPath = "data/Mp4_videos/h264_videos_dts_test/20221010/0012/1668001826042.mp4";
    std::string outPath = "data/testOutput/mp4_videos/outFrames/file_fixed_rate.mp4";
    bool parseFS = true;

	// write both videos as recorded i.e. including the gap
	bool recordedTSBasedDTS = false;
    read_write(videoPath, outPath, recordedTSBasedDTS, parseFS, UINT32_MAX);

	Test_Utils::deleteFolder(outPath);
}

struct SetupSeekTest {
	SetupSeekTest(std::string videoPath, bool parseFS)
	{
		LoggerProps loggerProps;
		loggerProps.logLevel = boost::log::trivial::severity_level::info;
		Logger::setLogLevel(boost::log::trivial::severity_level::info);
		Logger::initLogger(loggerProps);

		bool readLoop = false;
		auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, readLoop, false);
		mp4ReaderProps.logHealth = true;
		mp4ReaderProps.logHealthFrequency = 300;
		mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
		mp4Reader->addOutPutPin(h264ImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::H264_DATA);

		auto sinkProps = ExternalSinkProps();;
		sink = boost::shared_ptr<ExternalSink>(new ExternalSink(sinkProps));
		mp4Reader->setNext(sink, mImagePin);
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

		~ExternalSink()
		{
		}

	protected:
		bool process(frame_container &frames)
		{
			//LOG_ERROR << "ExternalSinkProcess <>";
			for (const auto &pair : frames)
			{
				LOG_TRACE << pair.first << "," << pair.second;
			}

			auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
			if (frame)
				LOG_INFO << "Timestamp <" << frame->timestamp << ">";

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
	boost::shared_ptr<Mp4ReaderSource> mp4Reader;
	boost::shared_ptr<ExternalSink> sink;
};

BOOST_AUTO_TEST_CASE(eof_seek_step)
{
	/* tests the issue - unable to read the file after randomSeek@same file once it reaches EOF */
	// reads the output video of read_mul_write_one_fixed_rate test above
    std::string videoPath = "data/Mp4_videos/file_fixed_rate.mp4";
    bool parseFS = false;

	SetupSeekTest f(videoPath, parseFS);
	BOOST_TEST(f.mp4Reader->init());
	BOOST_TEST(f.sink->init());

	f.mp4Reader->randomSeek(1668001826042);
	f.mp4Reader->step();

	f.mp4Reader->step();
	auto frames = f.sink->pop();
	auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::H264_DATA);
	BOOST_TEST(frame->timestamp == 1668001826042);

	f.mp4Reader->randomSeek(1668001829042);
	f.mp4Reader->step();

	f.mp4Reader->step();
	frames = f.sink->pop();
	frame = Module::getFrameByType(frames, FrameMetadata::FrameType::H264_DATA);
	BOOST_TEST(frame->timestamp == 1668001826075);
}

BOOST_AUTO_TEST_SUITE_END()