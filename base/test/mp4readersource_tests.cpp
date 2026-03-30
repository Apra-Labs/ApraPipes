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
#include "ImageDecoderCV.h"
#include "ImageViewerModule.h"

BOOST_AUTO_TEST_SUITE(mp4readersource_tests)

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

struct SetupMp4ReaderTest
{

	SetupMp4ReaderTest(std::string videoPath, framemetadata_sp inputMetadata, FrameMetadata::FrameType frameType, bool parseFS, bool isMetadata, int uniqMetadata = 0)
	{
		isVideoMetada = isMetadata;
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

		if (isMetadata)
		{
			auto metaSinkProps = MetadataSinkProps(uniqMetadata);
			metaSinkProps.logHealth = true;
			metaSinkProps.logHealthFrequency = 10;
			metaSink = boost::shared_ptr<MetadataSink>(new MetadataSink(metaSinkProps));
			mp4Reader->setNext(metaSink);
		}

		BOOST_TEST(mp4Reader->init());
		BOOST_TEST(sink->init());

	}

	~SetupMp4ReaderTest()
	{
		mp4Reader->term();
		if(isVideoMetada)
		metaSink->term();
		sink->term();
	}

	bool isVideoMetada;
	boost::shared_ptr<PipeLine> p = nullptr;
	boost::shared_ptr<Mp4ReaderSource> mp4Reader;
	boost::shared_ptr<ExternalSinkModule> sink;
	boost::shared_ptr<MetadataSink> metaSink;
};

BOOST_AUTO_TEST_CASE(mp4v_to_jpg_frames_metadata)
{
	std::string videoPath = "./data/Mp4_videos/jpg_video_metadata/20230513/0019/1686666193885.mp4";
	std::string outPath = "data/mp4Reader_saveOrCompare/jpeg/frame_000";
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = false;
	SetupMp4ReaderTest s(videoPath, encodedImageMetadata, frameType, parseFS, true);

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
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	std::string outPath = "data/mp4Reader_saveOrCompare/h264/frame_000";
	bool parseFS = false;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, true);

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

BOOST_AUTO_TEST_CASE(read_timeStamp_from_custom_fileName)
{
	/* file structure parsing test */
	std::string videoPath = "./data/Mp4_videos/h264_video/apraH264.mp4";
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = false;
	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);

	s.mp4Reader->step();
	auto frames = s.sink->pop();
	auto frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1673420640350);
}

BOOST_AUTO_TEST_CASE(getSetProps_change_root_folder)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	bool parseFS = true;
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));

	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);

	frame_container frames;
	// go till the second last frame
	for (int i = 0; i < 230; i++)
	{
		s.mp4Reader->step();
		frames = s.sink->pop();
	}

	// read the  last frame of the open video
	s.mp4Reader->step();
	frames = s.sink->pop();
	auto lastFrame = frames.begin()->second;
	BOOST_TEST(lastFrame->timestamp == 1686723806278);

	//change the video file path , Now read first frame new video of changed root dir instead of last frame of open video 
	auto propsChange = s.mp4Reader->getProps();
	propsChange.videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/20230111/0012/1673420640350.mp4";
	s.mp4Reader->setProps(propsChange);
	s.mp4Reader->step();
	frames = s.sink->pop();
	auto frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1673420640350);
}

BOOST_AUTO_TEST_CASE(getSetProps_change_root_folder_to_custom_file_name)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	bool parseFS = true;
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));

	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);
	frame_container frames;
	// go till the second last frame
	for (int i = 0; i < 230; i++)
	{
		s.mp4Reader->step();
		frames = s.sink->pop();
	}

	// read the  last frame of the open video
	s.mp4Reader->step();
	frames = s.sink->pop();
	auto lastFrame = frames.begin()->second;
	BOOST_TEST(lastFrame->timestamp == 1686723806278);

	//change the video file path , Now read first frame new video of changed root dir instead of last frame of open video 
	auto propsChange = s.mp4Reader->getProps();
	// To read custom file name parseFS needs to be disabled
	propsChange.parseFS = false;
	propsChange.videoPath = "./data/Mp4_videos/h264_video/apraH264.mp4";
	s.mp4Reader->setProps(propsChange);
	s.mp4Reader->step();
	frames = s.sink->pop();
	auto frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1673420640350);
}

BOOST_AUTO_TEST_CASE(getSetProps_NotParseFs_to_parseFS)
{
	std::string videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/apraH264.mp4";
	bool parseFS = false;
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));

	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);
	frame_container frames;

	s.mp4Reader->step();
	frames = s.sink->pop();
	auto frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1673420640350);

	// go till the second last frame
	for (int i = 0; i < 50; i++)
	{
		s.mp4Reader->step();
		frames = s.sink->pop();
	}

	//change the video file path , Now read first frame new video of changed root dir instead of last frame of open video 
	auto propsChange = s.mp4Reader->getProps();
	// To read custom file name parseFS needs to be disabled
	propsChange.parseFS = true;
	propsChange.videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/20230501/0012/1685604318680.mp4";
	s.mp4Reader->setProps(propsChange);
	s.mp4Reader->step();
	frames = s.sink->pop();
	frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1685604318680);
}

BOOST_AUTO_TEST_CASE(getSetProps_custom_file_name_to_root_dir)
{
	std::string videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/apraH264.mp4";
	bool parseFS = false;
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));

	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);
	frame_container frames;

	s.mp4Reader->step();
	frames = s.sink->pop();
	auto frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1673420640350);

	for (int i = 0; i < 50; i++)
	{
		s.mp4Reader->step();
		frames = s.sink->pop();
	}

	//change the video file path , Now read first frame new video of changed root dir instead of last frame of open video 
	auto propsChange = s.mp4Reader->getProps();
	// To read custom file name parseFS needs to be disabled
	propsChange.parseFS = true;
	propsChange.videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/";
	s.mp4Reader->setProps(propsChange);
	s.mp4Reader->step();
	frames = s.sink->pop();
	frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1673420640350);
}

BOOST_AUTO_TEST_CASE(getSetProps_root_dir_to_custom_file_name)
{
	std::string videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/";
	bool parseFS = true;
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));

	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);
	frame_container frames;

	s.mp4Reader->step();
	frames = s.sink->pop();
	auto frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1673420640350);

	for (int i = 0; i < 50; i++)
	{
		s.mp4Reader->step();
		frames = s.sink->pop();
	}

	//change the video file path , Now read first frame new video of changed root dir instead of last frame of open video 
	auto propsChange = s.mp4Reader->getProps();
	// To read custom file name parseFS needs to be disabled
	propsChange.parseFS = false;
	propsChange.videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/apraH264.mp4";
	s.mp4Reader->setProps(propsChange);
	s.mp4Reader->step();
	frames = s.sink->pop();
	frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1673420640350);
}

BOOST_AUTO_TEST_CASE(getSetProps_filename_to_root_dir)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	bool parseFS = true;
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));

	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);
	frame_container frames;

	s.mp4Reader->step();
	frames = s.sink->pop();
	auto frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1686723796848);

	//change the video file path , Now read first frame new video of changed root dir instead of last frame of open video 
	auto propsChange = s.mp4Reader->getProps();
	// To read custom file name parseFS needs to be disabled
	propsChange.parseFS = true;
	propsChange.videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/";
	s.mp4Reader->setProps(propsChange);
	s.mp4Reader->step();
	frames = s.sink->pop();
	frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1673420640350);
}

BOOST_AUTO_TEST_CASE(getSetProps_root_dir_to_filename)
{
	std::string videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/";
	bool parseFS = true;
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));

	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);
	frame_container frames;

	s.mp4Reader->step();
	frames = s.sink->pop();
	auto frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1673420640350);

	//change the video file path , Now read first frame new video of changed root dir instead of last frame of open video 
	auto propsChange = s.mp4Reader->getProps();
	// To read custom file name parseFS needs to be disabled
	propsChange.parseFS = true;
	propsChange.videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	s.mp4Reader->setProps(propsChange);
	s.mp4Reader->step();
	frames = s.sink->pop();
	frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1686723796848);
}

BOOST_AUTO_TEST_CASE(getSetProps_change_root_folder_fail)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";	
	bool parseFS = true;
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));

	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);
	frame_container frames;
	// go till the second last frame
	for (int i = 0; i < 229; i++)
	{
		s.mp4Reader->step();
		frames = s.sink->pop();
	}

	// read the  secoond last frame of the open video
	s.mp4Reader->step();
	frames = s.sink->pop();
	auto lastFrame = frames.begin()->second;
	BOOST_TEST(lastFrame->timestamp == 1686723806237);

	//change the video file path , Now read first frame new video of changed root dir instead of last frame of open video 
	auto propsChange = s.mp4Reader->getProps();
	propsChange.parseFS = false;
	//this path dosen't exist on disk - so cannoical path call will fail it, hence continue reading the open video
	propsChange.videoPath = "./data/Mp4_videos/videos/apraH264.mp4";
	s.mp4Reader->setProps(propsChange);
	s.mp4Reader->step();
	frames = s.sink->pop();
	auto frame = frames.begin()->second;
	// read the last frame of the open video
	BOOST_TEST(frame->timestamp == 1686723806278);
}

BOOST_AUTO_TEST_CASE(parse_root_dir_and_find_the_video)
{
	std::string videoPath = "./data/Mp4_videos/jpg_video";
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = false;
	SetupMp4ReaderTest s(videoPath, encodedImageMetadata, frameType, parseFS, false);
	
	BOOST_TEST(s.mp4Reader->step());
	auto frames = s.sink->pop();
}

BOOST_AUTO_TEST_CASE(check_exposed_params)
{
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
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


BOOST_AUTO_TEST_CASE(max_buffer_size_change_props)
{
	std::string startingVideoPath = "data/Mp4_videos/mp4_seek_tests/20220522/0016/1655895162221.mp4";
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
}


BOOST_AUTO_TEST_CASE(playrate_1x_all_frames_delivered)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	bool parseFS = false;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);

	double baseFPS = s.mp4Reader->getOpenVideoFPS();
	int totalFrames = s.mp4Reader->getOpenVideoFrameCount();

	// At 1x, all frames should be delivered, FPS = baseFPS
	auto props = s.mp4Reader->getProps();
	BOOST_TEST(props.fps == baseFPS, boost::test_tools::tolerance(0.01));

	// Read 10 consecutive frames and verify timestamps are monotonically increasing
	uint64_t prevTS = 0;
	for (int i = 0; i < 10; i++)
	{
		s.mp4Reader->step();
		auto frames = s.sink->pop();
		if (frames.empty()) continue;
		auto frame = frames.begin()->second;
		BOOST_TEST(frame->timestamp > prevTS);
		prevTS = frame->timestamp;
	}
}

BOOST_AUTO_TEST_CASE(playrate_2x_fps_doubled)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	bool parseFS = false;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);

	double baseFPS = s.mp4Reader->getOpenVideoFPS();

	s.mp4Reader->changePlaybackSpeed(2.0f, true);
	s.mp4Reader->step(); // process command
	s.mp4Reader->step(); // process props change

	auto props = s.mp4Reader->getProps();
	BOOST_TEST(props.playbackSpeed == 2.0f);
	// At 2x: FPS should be baseFPS * 2, all frames still delivered
	BOOST_TEST(props.fps == baseFPS * 2.0f, boost::test_tools::tolerance(0.01));

	// Frames should still be monotonically increasing (no frame skipping at 2x)
	uint64_t prevTS = 0;
	for (int i = 0; i < 10; i++)
	{
		s.mp4Reader->step();
		auto frames = s.sink->pop();
		if (frames.empty()) continue;
		auto frame = frames.begin()->second;
		BOOST_TEST(frame->timestamp > prevTS);
		prevTS = frame->timestamp;
	}
}

BOOST_AUTO_TEST_CASE(playrate_4x_iframe_skipping)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	bool parseFS = false;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);

	double baseFPS = s.mp4Reader->getOpenVideoFPS();

	s.mp4Reader->changePlaybackSpeed(4.0f, true);
	s.mp4Reader->step(); // process command
	s.mp4Reader->step(); // process props change

	auto props = s.mp4Reader->getProps();
	BOOST_TEST(props.playbackSpeed == 4.0f);

	// At 4x: I-frame skipping mode, FPS = (baseFPS * 4) / GOP
	// Should NOT equal baseFPS * 4 (GOP division applies)
	BOOST_TEST(props.fps != baseFPS * 4.0f);
	// FPS should be less than baseFPS * 4 (GOP > 1)
	BOOST_TEST(props.fps < baseFPS * 4.0f);

	// At 4x, only I-frames are delivered, so timestamp gaps should be > 1 frame interval
	double frameIntervalMs = 1000.0 / baseFPS;
	uint64_t prevTS = 0;
	int frameCount = 0;
	for (int i = 0; i < 5; i++)
	{
		s.mp4Reader->step();
		auto frames = s.sink->pop();
		if (frames.empty()) continue;
		auto frame = frames.begin()->second;
		if (prevTS > 0)
		{
			uint64_t gap = frame->timestamp - prevTS;
			// Gap should be larger than a single frame interval (I-frame skipping)
			BOOST_TEST(gap > static_cast<uint64_t>(frameIntervalMs));
		}
		prevTS = frame->timestamp;
		frameCount++;
	}
	BOOST_TEST(frameCount > 0);
}

BOOST_AUTO_TEST_CASE(playrate_switch_1x_to_4x_and_back)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	bool parseFS = false;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);

	double baseFPS = s.mp4Reader->getOpenVideoFPS();

	// Read a few frames at 1x
	for (int i = 0; i < 5; i++)
	{
		s.mp4Reader->step();
		s.sink->pop();
	}

	// Switch to 4x
	s.mp4Reader->changePlaybackSpeed(4.0f, true);
	s.mp4Reader->step();
	s.mp4Reader->step();
	auto props = s.mp4Reader->getProps();
	BOOST_TEST(props.playbackSpeed == 4.0f);
	BOOST_TEST(props.fps < baseFPS * 4.0f); // GOP division applied

	// Read a few frames at 4x
	for (int i = 0; i < 3; i++)
	{
		s.mp4Reader->step();
		s.sink->pop();
	}

	// Switch back to 1x
	s.mp4Reader->changePlaybackSpeed(1.0f, true);
	s.mp4Reader->step();
	s.mp4Reader->step();
	props = s.mp4Reader->getProps();
	BOOST_TEST(props.playbackSpeed == 1.0f);
	BOOST_TEST(props.fps == baseFPS, boost::test_tools::tolerance(0.01));
}

BOOST_AUTO_TEST_CASE(playrate_8x_fps_gop_adjusted)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	bool parseFS = false;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);

	double baseFPS = s.mp4Reader->getOpenVideoFPS();

	s.mp4Reader->changePlaybackSpeed(8.0f, true);
	s.mp4Reader->step();
	s.mp4Reader->step();

	auto props = s.mp4Reader->getProps();
	BOOST_TEST(props.playbackSpeed == 8.0f);
	// At 8x: same I-frame skipping as 4x, GOP division applied
	BOOST_TEST(props.fps < baseFPS * 8.0f);
	BOOST_TEST(props.fps > 0.0f);
}

BOOST_AUTO_TEST_CASE(playrate_backward_1x)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	bool parseFS = false;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	SetupMp4ReaderTest s(videoPath, h264ImageMetadata, frameType, parseFS, false);

	// Read a few frames forward first to get away from the start
	for (int i = 0; i < 20; i++)
	{
		s.mp4Reader->step();
		s.sink->pop();
	}

	// Switch to backward playback at 1x
	s.mp4Reader->changePlayback(1.0f, false);
	s.mp4Reader->step(); // process command

	// Read frames and verify timestamps are monotonically decreasing
	uint64_t prevTS = UINT64_MAX;
	int validFrames = 0;
	for (int i = 0; i < 5; i++)
	{
		s.mp4Reader->step();
		auto frames = s.sink->pop();
		if (frames.empty()) continue;
		auto frame = frames.begin()->second;
		if (prevTS != UINT64_MAX)
		{
			BOOST_TEST(frame->timestamp < prevTS);
		}
		prevTS = frame->timestamp;
		validFrames++;
	}
	BOOST_TEST(validFrames > 0);
}

BOOST_AUTO_TEST_CASE(playrate_initial_speed_from_props)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	bool parseFS = false;

	// Initialize with 2x speed in props
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
	mp4ReaderProps.playbackSpeed = 2.0f;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	mp4Reader->setNext(sink);

	BOOST_TEST(mp4Reader->init());
	BOOST_TEST(sink->init());

	double baseFPS = mp4Reader->getOpenVideoFPS();
	auto props = mp4Reader->getProps();

	// FPS should already reflect 2x speed from init
	BOOST_TEST(props.playbackSpeed == 2.0f);
	BOOST_TEST(props.fps == baseFPS * 2.0f, boost::test_tools::tolerance(0.01));

	mp4Reader->term();
	sink->term();
}

BOOST_AUTO_TEST_SUITE_END()
