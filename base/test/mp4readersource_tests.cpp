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
#include "H264Decoder.h"
#include "ImageViewerModule.h"
#include "ColorConversionXForm.h"
#include "..\include\FileWriterModule.h"
#include <conio.h>

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
	boost::shared_ptr<H264Decoder> Decoder;
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
	std::string outPath = "data/testOutput/outFrames";
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
	std::string outPath = "./data/testOutput/outFrames/";
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

BOOST_AUTO_TEST_CASE(getSetProps_change_root_folder_with_custom_file_name)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	std::string outPath = "./data/testOutput/outFrames/";
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

BOOST_AUTO_TEST_CASE(NotParseFs_to_parseFS)
{
	std::string videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/apraH264.mp4";
	std::string outPath = "./data/testOutput/outFrames/";
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
	propsChange.videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/20230501/0012/1685604318680.mp4";
	s.mp4Reader->setProps(propsChange);
	s.mp4Reader->step();
	frames = s.sink->pop();
	frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1685604318680);
}

BOOST_AUTO_TEST_CASE(custom_file_name_to_root_dir)
{
	std::string videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/apraH264.mp4";
	std::string outPath = "./data/testOutput/outFrames/";
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
	propsChange.videoPath = "./data/Mp4_videos/mp4_seeks_tests_h264/";
	s.mp4Reader->setProps(propsChange);
	s.mp4Reader->step();
	frames = s.sink->pop();
	frame = frames.begin()->second;
	BOOST_TEST(frame->timestamp == 1673420640350);
}

BOOST_AUTO_TEST_CASE(getSetProps_change_root_folder_fail)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
	std::string outPath = "./data/testOutput/outFrames/";
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
	bool parseFS = true;
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

BOOST_AUTO_TEST_CASE(mp4v_to_h264_frames_reverseplay)
{
	std::string videoPath = "data/Mp4_videos/reverseplay_h264/20220910/0012/1691758391768.mp4";
	std::string outPath = "data/testOutput/outFrames";
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = false;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);


	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
	mp4ReaderProps.fps = 24;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	mp4Reader->setNext(Decoder, mImagePin);

	auto colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)));
	Decoder->setNext(colorchange);

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("MotionVectorsOverlay")));

	colorchange->setNext(sink);
	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	Test_Utils::sleep_for_seconds(20);

	mp4Reader->changePlayback(1, false);

	Test_Utils::sleep_for_seconds(150);

	mp4Reader->changePlayback(1, true);

	Test_Utils::sleep_for_seconds(20);

	mp4Reader->changePlayback(1, false);

	Test_Utils::sleep_for_seconds(200);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(mp4v_to_h264_frames_reverseplay_fwd)
{
	std::string videoPath = "data/Mp4_videos/reverseplay_h264/20220910/0012/1691758391768.mp4";
	std::string outPath = "data/testOutput/outFrames";
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);


	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, false, false, false);
	mp4ReaderProps.fps = 25;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	mp4Reader->setNext(Decoder, mImagePin);

	auto colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)));
	Decoder->setNext(colorchange);

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("MotionVectorsOverlay")));

	colorchange->setNext(sink);
	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	Test_Utils::sleep_for_seconds(50);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(filewriter)
{
	std::string videoPath = "data/Mp4_videos/reverseplay_h264/20220910/0012/1692776889039.mp4";
	std::string outPath = "data/testOutput/outFrames";
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);


	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
	mp4ReaderProps.fps = 24;
	mp4ReaderProps.readLoop = false;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/3840x2160/frame_????.h264")));
	mp4Reader->setNext(fileWriter, mImagePin);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	Test_Utils::sleep_for_seconds(60);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(mp4v_to_h264_frames_reverseplay_fileReader)
{
	std::string videoPath = "data/Mp4_videos/reverseplay_h264/20220910/0012/1691758391768.mp4";
	std::string outPath = "data/testOutput/outFrames";
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);


	auto fileReaderProps = FileReaderModuleProps("./data/rtspCamFrames_new/");
	fileReaderProps.fps = 24;
	fileReaderProps.readLoop = false;

	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto encodedImageMetadata = framemetadata_sp(new H264Metadata(1920, 1080));
	fileReader->addOutputPin(encodedImageMetadata);

	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	fileReader->setNext(Decoder);

	auto colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)));
	Decoder->setNext(colorchange);

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("MotionVectorsOverlay")));

	colorchange->setNext(sink);
	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	Test_Utils::sleep_for_seconds(50);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(keyboard)
{
	std::string videoPath = "data/Mp4_videos/reverseplay_h264/20220910/0012/1691502958947.mp4";
	std::string outPath = "data/testOutput/outFrames";
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);


	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
	mp4ReaderProps.fps = 25;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	mp4Reader->setNext(Decoder, mImagePin);

	auto colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)));
	Decoder->setNext(colorchange);

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("MotionVectorsOverlay")));

	colorchange->setNext(sink);
	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	mp4Reader->init();
	Decoder->init();
	colorchange->init();
	sink->init();

	for (int i = 0; i <= 4; i++)
	{
		mp4Reader->step();
		Decoder->step();
	}

	char ch;

	//program pauses here until key is pressed

	while (1)
	{
		ch = _getch();
		if (ch == 'd')
		{
			mp4Reader->step();
			Decoder->step();
			colorchange->step();
			sink->step();
		}
		if (ch == 'b')
		{
			mp4Reader->changePlayback(1, false);
		}
		if (ch == 'f')
		{
			mp4Reader->changePlayback(1, true);
		}
		if (ch == 'e')
		{
			mp4Reader->step();
			Decoder->step();
		}
	}
}

BOOST_AUTO_TEST_CASE(handle_direction_change)
{
	std::string videoPath = "data/Mp4_videos/reverseplay_h264/20220910/0012/1691502958947.mp4";
	std::string outPath = "data/testOutput/outFrames";
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);


	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
	mp4ReaderProps.fps = 25;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	mp4Reader->setNext(Decoder, mImagePin);

	auto colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)));
	Decoder->setNext(colorchange);

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("MotionVectorsOverlay")));

	colorchange->setNext(sink);
	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	mp4Reader->init();
	Decoder->init();
	colorchange->init();
	sink->init();

	for (int i = 0; i <= 4; i++)
	{
		mp4Reader->step();
		Decoder->step();
	}

	for (int i = 0; i <= 30; i++)
	{
		mp4Reader->step();
		Decoder->step();
		colorchange->step();
		sink->step();
	}
	mp4Reader->changePlayback(1, false);

	for (int i = 0; i <= 8; i++)
	{
		mp4Reader->step();
		Decoder->step();
		colorchange->step();
		sink->step();
	}

	mp4Reader->changePlayback(1, true);
	for (int i = 0; i <= 13; i++)
	{
		mp4Reader->step();
		Decoder->step();
		colorchange->step();
		sink->step();
	}

	for (int i = 0; i <= 5; i++)
	{
		mp4Reader->step();
		Decoder->step();
	}
}


BOOST_AUTO_TEST_CASE(reverseplay_seek)
{
	std::string videoPath = "data/Mp4_videos/reverseplay_h264/20220910/0012/1691758391768.mp4";
	std::string outPath = "data/testOutput/outFrames";
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);


	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, false, false, false);
	mp4ReaderProps.fps = 24;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	mp4Reader->setNext(Decoder, mImagePin);

	auto colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)));
	Decoder->setNext(colorchange);

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("MotionVectorsOverlay")));

	colorchange->setNext(sink);
	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	Test_Utils::sleep_for_seconds(15);

	mp4Reader->changePlayback(1, true);
	Test_Utils::sleep_for_seconds(14);

	uint64_t skipTS = 1691758409687;
	
	mp4Reader->randomSeek(skipTS, false);
	mp4Reader->changePlayback(1, false);
	

	Test_Utils::sleep_for_seconds(15);

	skipTS = 1691758415687;
	mp4Reader->changePlayback(1, true);
	mp4Reader->randomSeek(skipTS, false);

	Test_Utils::sleep_for_seconds(10);

	skipTS = 1691758391768;
	mp4Reader->randomSeek(skipTS, false);
	mp4Reader->changePlayback(1, false);
	Test_Utils::sleep_for_seconds(20);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(step)
{
	std::string videoPath = "data/Mp4_videos/reverseplay_h264/20220910/0012/1691758391768.mp4";
	std::string outPath = "data/testOutput/outFrames";
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);


	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
	mp4ReaderProps.fps = 25;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	mp4Reader->setNext(Decoder, mImagePin);

	auto colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)));
	Decoder->setNext(colorchange);

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("MotionVectorsOverlay")));

	colorchange->setNext(sink);
	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	mp4Reader->init();
	Decoder->init();
	colorchange->init();
	sink->init();

	for (int i = 0; i <= 4; i++)
	{
		mp4Reader->step();
		Decoder->step();
	}
	mp4Reader->changePlayback(1, false);
	for (int i = 0; i <= 15; i++)
	{
		mp4Reader->step();
		Decoder->step();
	}
	
	mp4Reader->changePlayback(1, true);
	for (int i = 0; i <= 8; i++)
	{
		mp4Reader->step();
		Decoder->step();

	}
	for (int i = 0; i <= 20; i++)
	{
		mp4Reader->step();
		Decoder->step();
		colorchange->step();
		sink->step();
	}

	//mp4Reader->changePlayback(1, true);
	for (int i = 0; i <= 400; i++)
	{
		mp4Reader->step();
		Decoder->step();
		colorchange->step();
		sink->step();
	}

	for (int i = 0; i <= 5; i++)
	{
		mp4Reader->step();
		Decoder->step();
	}
}

BOOST_AUTO_TEST_CASE(ultimate)
{
	std::string videoPath = "data/Mp4_videos/reverseplay_h264/20220910/0012/1691758391768.mp4";
	std::string outPath = "data/testOutput/outFrames";
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);


	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
	mp4ReaderProps.fps = 24;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	mp4Reader->setNext(Decoder, mImagePin);

	auto colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)));
	Decoder->setNext(colorchange);

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("MotionVectorsOverlay")));

	colorchange->setNext(sink);
	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	Test_Utils::sleep_for_seconds(10);

	uint64_t skipTS = 1691758412687;
	mp4Reader->changePlayback(1, false);
	mp4Reader->randomSeek(skipTS, false);


	Test_Utils::sleep_for_seconds(15);

	mp4Reader->changePlayback(1, true);

	Test_Utils::sleep_for_seconds(10);

	skipTS = 1691758391769;
	mp4Reader->changePlayback(1, true);
	mp4Reader->randomSeek(skipTS, false);
	Test_Utils::sleep_for_seconds(10);

	mp4Reader->changePlayback(1, false);
	Test_Utils::sleep_for_seconds(10);

	/*skipTS = 1691758415687;
	mp4Reader->changePlayback(1, true);
	mp4Reader->randomSeek(skipTS, false);*/

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(cars)
{
	std::string videoPath = "data/Mp4_videos/reverseplay_h264/20220910/0012/1691502958947.mp4";
	std::string outPath = "data/testOutput/outFrames";
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);


	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
	mp4ReaderProps.fps = 24;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	mp4Reader->setNext(Decoder, mImagePin);

	auto colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)));
	Decoder->setNext(colorchange);

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("MotionVectorsOverlay")));

	colorchange->setNext(sink);
	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	Test_Utils::sleep_for_seconds(5);

	uint64_t skipTS = 1691502964947;
	mp4Reader->changePlayback(1, false);
	mp4Reader->randomSeek(skipTS, false);


	Test_Utils::sleep_for_seconds(4);

	mp4Reader->changePlayback(1, true);

	Test_Utils::sleep_for_seconds(7);

	mp4Reader->changePlayback(1, false);

	Test_Utils::sleep_for_seconds(7);

	skipTS = 1691502980947;
	mp4Reader->changePlayback(1, true);
	mp4Reader->randomSeek(skipTS, false);
	Test_Utils::sleep_for_seconds(5);

	mp4Reader->changePlayback(1, false);
	Test_Utils::sleep_for_seconds(20);

	/*skipTS = 1691758415687;
	mp4Reader->changePlayback(1, true);
	mp4Reader->randomSeek(skipTS, false);*/

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_SUITE_END()
