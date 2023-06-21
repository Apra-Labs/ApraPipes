#include <boost/test/unit_test.hpp>

#include "Logger.h"
#include "PipeLine.h"
#include "FileReaderModule.h"
#include "Mp4ReaderSource.h"
#include "StatSink.h"
#include "FrameMetadata.h"
#include "EncodedImageMetadata.h"
#include "H264Metadata.h"
#include "Mp4VideoMetadata.h"
#include "Mp4WriterSink.h"
#include "ExternalSinkModule.h"
#include "test_utils.h"
#include "Mp4ErrorFrame.h"

BOOST_AUTO_TEST_SUITE(Mp4ReaderSource_tests)

class MetadataSinkProps : public ModuleProps
{
public:
	MetadataSinkProps(int _uniqMetadata) : ModuleProps()
	{
		uniqMetadata = _uniqMetadata;
	}
	int uniqMetadata;
};

class MetadataSink : public Module
{
public:
	MetadataSink(MetadataSinkProps props = MetadataSinkProps(0)) : Module(SINK, "MetadataSink", props), mProps(props)
	{
	}
	virtual ~MetadataSink()
	{
	}
protected:
	bool process(frame_container& frames) {
		auto frame = getFrameByType(frames, FrameMetadata::FrameType::MP4_VIDEO_METADATA);
		if (!isFrameEmpty(frame))
		{
			if (frame->fIndex < 100)
			{
				metadata.assign(reinterpret_cast<char*>(frame->data()), 11);

				LOG_INFO << "Metadata\n frame_numer <" << frame->fIndex + 1 << "><" << metadata << ">";
				if (!mProps.uniqMetadata)
				{
					auto frameIndex = frame->fIndex;
					auto index = frameIndex % 5;
					auto myString = "ApraPipes_" + std::to_string(index);
					BOOST_TEST(myString == metadata);
				}
				else
				{
					int num = (frame->fIndex + 1) % mProps.uniqMetadata;
					num = num ? num : mProps.uniqMetadata;
					std::string shouldBeMeta = "frame_" + std::to_string(num);

					BOOST_TEST(shouldBeMeta.size() == metadata.size());
					BOOST_TEST(shouldBeMeta.data() == metadata.data());
				}
			}
		}

		return true;
	}
	bool validateInputPins() { return true; }
	bool validateInputOutputPins() { return true; }
	std::string metadata;
	MetadataSinkProps mProps;
};

struct setupMp4ReaderTest
{

	setupMp4ReaderTest(std::string videoPath, framemetadata_sp inputMetadata, FrameMetadata::FrameType frameType, bool parseFS, int uniqMetadata = 0)
	{
		LoggerProps loggerProps;
		loggerProps.logLevel = boost::log::trivial::severity_level::info;
		Logger::setLogLevel(boost::log::trivial::severity_level::info);
		Logger::initLogger(loggerProps);

		auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
		mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

		mp4Reader->addOutPutPin(inputMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

		sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());

		mp4Reader->setNext(sink, mImagePin);

		auto metaSinkProps = MetadataSinkProps(uniqMetadata);
		metaSinkProps.logHealth = true;
		metaSinkProps.logHealthFrequency = 10;
		metaSink = boost::shared_ptr<MetadataSink>(new MetadataSink(metaSinkProps));
		mp4Reader->setNext(metaSink);

		mp4Reader->init();
		sink->init();

	}

	~setupMp4ReaderTest()
	{
		mp4Reader->term();
		metaSink->term();
		sink->term();
	}

	boost::shared_ptr<PipeLine> p = nullptr;
	boost::shared_ptr<Mp4ReaderSource> mp4Reader;
	boost::shared_ptr<ExternalSinkModule> sink;
	boost::shared_ptr<MetadataSink> metaSink;
};

// todo - basic read test with 4 saveOrCompare for jpeg and h264 (done)
// - read metadata as well. 
BOOST_AUTO_TEST_CASE(mp4v_to_jpg_frames_metadata)
{
	std::string videoPath = "./data/Mp4_videos/jpg_video_metada/20230513/0019/1686666193885.mp4";
	std::string outPath = "data/mp4Reader_saveOrCompare/jpeg/frame_000";
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = false;
	setupMp4ReaderTest s(videoPath, encodedImageMetadata, frameType, parseFS);

	for (int i = 0; i < 180; i++)
	{
		s.mp4Reader->step();
		s.metaSink->step();
		auto frames = s.sink->pop();
		auto outputFrame = frames.begin()->second;
		std::string fileName;
		if (i % 10 == 0)
		{
			if (i < 10)
			{
				fileName = outPath + "00" + to_string(i) + ".jpg";
			}
			else if (i >= 10 && i < 100)
			{
				fileName = outPath + "0" + to_string(i) + ".jpg";
			}
			else
			{
				fileName = outPath + to_string(i) + ".jpg";
			}
			Test_Utils::saveOrCompare(fileName.c_str(), (const uint8_t*)outputFrame->data(), outputFrame->size(), 0);
		}
	}

}

BOOST_AUTO_TEST_CASE(mp4v_to_h264_frames_metadata)
{
	std::string videoPath = "./data/mp4_videoS/h264_video_metadata/20230514/0011/1686723796848.mp4";
	std::string outPath = "data/mp4Reader_saveOrCompare/h264/frame_000";
	bool parseFS = false;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	setupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS);

	for (int i = 0; i < 180; i++)
	{
		s.mp4Reader->step();
		s.metaSink->step();
		auto frames = s.sink->pop();
		auto outputFrame = frames.begin()->second;
		std::string fileName;
		if (i % 10 == 0)
		{
			if (i < 10)
			{
				fileName = outPath + "00" + to_string(i) + ".h264";
			}
			else if (i >= 10 && i < 100)
			{
				fileName = outPath + "0" + to_string(i) + ".h264";
			}
			else
			{
				fileName = outPath + to_string(i) + ".h264";
			}
			Test_Utils::saveOrCompare(fileName.c_str(), (const uint8_t*)outputFrame->data(), outputFrame->size(), 0);
		}
	}
}

// todo - read timestamp from file
BOOST_AUTO_TEST_CASE(read_timeStamp_from_custom_fileName)
{
	/* file structure parsing test */
	std::string videoPath = "./data/Mp4_videos/h264_video/apraH264.mp4";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.h264");
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = false;
	setupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS);

	s.mp4Reader->step();
	auto frames = s.sink->pop();
	auto frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1673420640350);
}

// todo - one get set prop - fix the flow (done)
BOOST_AUTO_TEST_CASE(getSetProps)
{
	std::string videoPath = "./data/Mp4_videos/jpg_video/20220928/0013/1666943213667.mp4";
	std::string outPath = "./data/testOutput/outFrames/";
	bool parseFS = true;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	boost::filesystem::path dir(outPath);

	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	mp4Reader->addOutPutPin(encodedImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());

	std::vector<std::string> encodedImagePin;
	encodedImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE);
	mp4Reader->setNext(sink, encodedImagePin);

	mp4Reader->init();
	sink->init();
	frame_container frames;
	for (int i = 0; i < 182; i++)
	{
		mp4Reader->step();
		frames = sink->pop();
	}

	auto propsChange = mp4Reader->getProps();
	propsChange.readLoop = true;
	mp4Reader->setProps(propsChange);

	//last frame of the open video
	mp4Reader->step();
	frames = sink->pop();
	auto frame = frames.begin()->second;
	BOOST_TEST(frame->size() == 4345);
}

BOOST_AUTO_TEST_CASE(parse_root_dir_and_find_the_video)
{
	std::string videoPath = "./data/Mp4_videos/jpg_video";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = false;
	setupMp4ReaderTest s(videoPath, encodedImageMetadata, frameType, parseFS);
	
	BOOST_TEST(s.mp4Reader->step());
	auto frames = s.sink->pop();
}

//Note: We still have to implement the feature to read and write video and simultaneously
BOOST_AUTO_TEST_CASE(mp4reader_waits_when_no_video_and_reads_whenever_video_is_written, *boost::unit_test::disabled())
{

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto fileReaderProps = FileReaderModuleProps("./data/re3_filtered", 0, -1);
	fileReaderProps.fps = 24;
	fileReaderProps.readLoop = false;

	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	fileReader->addOutputPin(encodedImageMetadata);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 10, 24, "./data/testOutput/mp4_videos/");
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 100;
	auto mp4WriterSink = boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));
	fileReader->setNext(mp4WriterSink);

	boost::filesystem::path dir("./data/testOutput/outFrames/" );

	auto mp4ReaderProps = Mp4ReaderSourceProps("./data/testOutput/mp4_videos/", false, 0, true, false, false, 10);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto imageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	mp4Reader->addOutPutPin(imageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto sink = boost::shared_ptr<Module>(new ExternalSinkModule());
	std::vector<std::string> encodedImagePin;
	encodedImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE);
	mp4Reader->setNext(sink, encodedImagePin);

	StatSinkProps statSinkProps;
	statSinkProps.logHealth = true;
	statSinkProps.logHealthFrequency = 10;
	auto statSink = boost::shared_ptr<Module>(new StatSink(statSinkProps));
	mp4Reader->setNext(statSink);

	auto metaSinkProps = MetadataSinkProps(0);
	metaSinkProps.logHealth = true;
	metaSinkProps.logHealthFrequency = 10;
	auto metaSink = boost::shared_ptr<Module>(new MetadataSink(metaSinkProps));
	mp4Reader->setNext(metaSink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}
	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(15));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();

}

// todo (done)
BOOST_AUTO_TEST_CASE(check_exposed_params)
{
	std::string startingVideoPath = "data/mp4_video/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	bool parseFS = true;

	auto mp4ReaderProps = Mp4ReaderSourceProps(startingVideoPath, parseFS, 0, true, false, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));

	mp4Reader->addOutPutPin(encodedImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	mp4Reader->setNext(sink);

	mp4Reader->init();
	sink->init();
	/* process one frame */
	mp4Reader->step();
	auto frames = sink->pop();
	
	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655895162221);

	BOOST_TEST(mp4Reader->getOpenVideoFPS() == 60, boost::test_tools::tolerance(0.99));
	BOOST_TEST(mp4Reader->getOpenVideoDurationInSecs() == 3);
	BOOST_TEST(mp4Reader->getOpenVideoFrameCount() == 181);
}

// todo (done)
BOOST_AUTO_TEST_CASE(max_buffer_size_change_props)
{
	std::string startingVideoPath = "data/mp4_video/mp4_seek_tests/20220522/0016/1655895162221.mp4";
	int width = 22, height = 30;
	bool parseFS = true;

	auto mp4ReaderProps = Mp4ReaderSourceProps(startingVideoPath, parseFS, 0, true, false, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));

	auto pinId = mp4Reader->addOutPutPin(encodedImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());

	mp4Reader->setNext(sink);

	mp4Reader->init();
	sink->init();
	/* process one frame */
	mp4Reader->step();
	auto frames = sink->pop();
	auto imgFrame = frames.begin()->second;
	BOOST_TEST(imgFrame->timestamp == 1655895162221);

	// change prop
	auto mp4Props = mp4Reader->getProps();
	mp4Props.biggerFrameSize = 100;
	mp4Reader->setProps(mp4Props);

	// process next frame
	mp4Reader->step();
	frames = sink->pop();
	auto frame = frames.begin()->second;
	BOOST_TEST(frame->isMp4ErrorFrame());
	auto errorFrame = dynamic_cast<Mp4ErrorFrame*>(frame.get());
	auto type = errorFrame->errorCode;
	BOOST_TEST(type == MP4_BUFFER_TOO_SMALL);

	// change prop
	mp4Props = mp4Reader->getProps();
	mp4Props.biggerFrameSize = 300;
	mp4Reader->setProps(mp4Props);
	mp4Reader->step();

	// process next frame (size is > 100 i.e. 284 Bytes)
	mp4Reader->step();
	frames = sink->pop();
	BOOST_TEST((frames.find(pinId) != frames.end()));

	/* TODO
		- extend this test to verify ts of next frame as well
		- currently, that will fail because setProps() makes it go to next video instead

	imgFrame = frames[rawImgPinId];
	BOOST_TEST(imgFrame->timestamp == NEXT_TS);
	*/
}

BOOST_AUTO_TEST_SUITE_END()
