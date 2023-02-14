#include <boost/test/unit_test.hpp>

#include "Logger.h"
#include "PipeLine.h"
#include "FileReaderModule.h"
#include "Mp4ReaderSource.h"
#include "FileWriterModule.h"
#include "StatSink.h"
#include "FrameMetadata.h"
#include "EncodedImageMetadata.h"
#include "H264Metadata.h"
#include "Mp4VideoMetadata.h"
#include "Mp4WriterSink.h"

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
				metadata.assign(reinterpret_cast<char*>(frame->data()), frame->size());

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

void read_video_extract_frames(std::string videoPath, std::string outPath, boost::filesystem::path file, framemetadata_sp inputMetadata, FrameMetadata::FrameType frameType, bool parseFS, int uniqMetadata = 0)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	boost::filesystem::path dir(outPath);

	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	mp4Reader->addOutPutPin(inputMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	boost::filesystem::path full_path = dir / file;
	LOG_INFO << full_path;
	auto fileWriterProps = FileWriterModuleProps(full_path.string());
	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(fileWriterProps));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);
	mp4Reader->setNext(fileWriter, mImagePin);

	StatSinkProps statSinkProps;
	statSinkProps.logHealth = true;
	statSinkProps.logHealthFrequency = 10;
	auto statSink = boost::shared_ptr<Module>(new StatSink(statSinkProps));
	mp4Reader->setNext(statSink);

	auto metaSinkProps = MetadataSinkProps(uniqMetadata);
	metaSinkProps.logHealth = true;
	metaSinkProps.logHealthFrequency = 10;
	auto metaSink = boost::shared_ptr<Module>(new MetadataSink(metaSinkProps));
	mp4Reader->setNext(metaSink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

void random_seek_video(std::string skipDir, uint64_t seekStartTS, uint64_t seekEndTS, std::string startingVideoPath, std::string outPath, framemetadata_sp inputMetadata, FrameMetadata::FrameType frameType, boost::filesystem::path file)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	boost::filesystem::path dir(outPath);

	auto mp4ReaderProps = Mp4ReaderSourceProps(startingVideoPath, false,true);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	mp4Reader->addOutPutPin(inputMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	mp4ReaderProps.skipDir = skipDir;

	boost::filesystem::path full_path = dir / file;
	LOG_INFO << full_path;
	auto fileWriterProps = FileWriterModuleProps(full_path.string());
	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(fileWriterProps));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);
	mp4Reader->setNext(fileWriter, mImagePin);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	mp4Reader->setProps(mp4ReaderProps);
	mp4Reader->randomSeek(seekStartTS,seekEndTS);

	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(mp4v_to_rgb_24_jpg)
{
	std::string videoPath = "./data/Mp4_videos/jpg_video/20220928/0013/10.mp4";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = false;
	read_video_extract_frames(videoPath, outPath, file, encodedImageMetadata, frameType, parseFS);
}

BOOST_AUTO_TEST_CASE(mp4v_to_mono_8_jpg)
{
	std::string videoPath = "./data/Mp4_videos/jpg_video/20220928/0013/1666943213667.mp4";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = false;
	read_video_extract_frames(videoPath, outPath, file, encodedImageMetadata, frameType, parseFS);
}

BOOST_AUTO_TEST_CASE(mp4v_read_metadata_jpg)
{
	std::string videoPath = "data/Mp4_videos/jpg_video_metada/20220928/0014/1666949168743.mp4";
	std::string outPath = "./data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = false;
	read_video_extract_frames(videoPath, outPath, file, encodedImageMetadata, frameType, parseFS);
}

BOOST_AUTO_TEST_CASE(fs_parsing_jpg)
{
	/* file structure parsing test */
	std::string videoPath = "data/Mp4_videos/jpg_video/20220928/0013/1666943213667.mp4";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = true;
	read_video_extract_frames(videoPath, outPath, file, encodedImageMetadata, frameType, parseFS, 5);
}

BOOST_AUTO_TEST_CASE(random_seek_jpg)
{
	std::string skipDir = "data/Mp4_videos/jpg_video_metada/";
	std::string startingVideoPath = "data/Mp4_videos/jpg_video_metada/20220928/0014/1666949168743.mp4";
	std::string outPath = "data/testOutput/outFrames";
	uint64_t seekStartTS = 1666949171743;
	uint64_t seekEndTS = 1666949175743;
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	random_seek_video(skipDir, seekStartTS, seekEndTS, startingVideoPath, outPath, encodedImageMetadata, frameType, file);
}

BOOST_AUTO_TEST_CASE(mp4v_to_h264frames_metadata)
{
	std::string videoPath = "./data/Mp4_videos/h264_video_metadata/20221009/0019/1668001826042.mp4";
	std::string outPath = "data/testOutput/outFrames";
	bool parseFS = false;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	boost::filesystem::path file("frame_??????.h264");
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	read_video_extract_frames(videoPath, outPath, file, h264ImageMetadata, frameType, parseFS);
}

BOOST_AUTO_TEST_CASE(mp4v_to_h264frames)
{
	std::string videoPath = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	std::string outPath = "data/testOutput/outFrames";
	bool parseFS = false;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	boost::filesystem::path file("frame_??????.h264");
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	read_video_extract_frames(videoPath, outPath, file, h264ImageMetadata, frameType, parseFS);
}

BOOST_AUTO_TEST_CASE(random_seek_h264)
{
	std::string skipDir = "data/Mp4_videos/h264_video/";
	std::string startingVideoPath = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	std::string outPath = "data/testOutput/outFrames";
	uint64_t seekStartTS = 1668064030062;
	uint64_t seekEndTS = 1668064032062;
	boost::filesystem::path file("frame_??????.h264");
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));

	random_seek_video(skipDir, seekStartTS, seekEndTS, startingVideoPath, outPath, h264ImageMetadata, frameType, file);
}

BOOST_AUTO_TEST_CASE(fs_parsing_h264, *boost::unit_test::disabled())
{
	/* file structure parsing test */
	std::string videoPath = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.h264");
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;
	read_video_extract_frames(videoPath, outPath, file, h264ImageMetadata, frameType, 5, parseFS);
}

BOOST_AUTO_TEST_CASE(read_timeStamp_from_custom_fileName)
{
	/* file structure parsing test */
	std::string videoPath = "./data/Mp4_videos/h264_video/apraH264.mp4";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.h264");
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;
	read_video_extract_frames(videoPath, outPath, file, h264ImageMetadata, frameType, 5, parseFS);
}

BOOST_AUTO_TEST_CASE(getSetProps)
{
	std::string videoPath = "./data/Mp4_videos/jpg_video/20220928/0013/1666943213667.mp4";
	std::string outPath = "./data/testOutput/outFrames/";
	std::string changedVideoPath = "./data/Mp4_videos/jpg_video_metada/20220928/0014/1666949168743.mp4";
	bool parseFS = true;
	int uniqMetadata = 0;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	boost::filesystem::path dir(outPath);

	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	mp4Reader->addOutPutPin(encodedImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	boost::filesystem::path file("frame_??????.jpg");
	boost::filesystem::path full_path = dir / file;
	LOG_INFO << full_path;
	auto fileWriterProps = FileWriterModuleProps(full_path.string());
	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(fileWriterProps));
	std::vector<std::string> encodedImagePin;
	encodedImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE);
	mp4Reader->setNext(fileWriter, encodedImagePin);

	StatSinkProps statSinkProps;
	statSinkProps.logHealth = true;
	statSinkProps.logHealthFrequency = 10;
	auto statSink = boost::shared_ptr<Module>(new StatSink(statSinkProps));
	mp4Reader->setNext(statSink);

	auto metaSinkProps = MetadataSinkProps(uniqMetadata);
	metaSinkProps.logHealth = true;
	metaSinkProps.logHealthFrequency = 10;
	auto metaSink = boost::shared_ptr<Module>(new MetadataSink(metaSinkProps));
	mp4Reader->setNext(metaSink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}
	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	Mp4ReaderSourceProps propsChange(changedVideoPath, true);
	mp4Reader->setProps(propsChange);

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(parse_root_dir_and_find_the_video)
{
	std::string videoPath = "./data/Mp4_videos/jpg_video";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = false;
	read_video_extract_frames(videoPath, outPath, file, encodedImageMetadata, frameType, parseFS);
}

//Note: We still have to implement the feature to read and write video and simultaneously
BOOST_AUTO_TEST_CASE(mp4reader_waits_when_no_video_and_reads_whenever_video_is_written)//
{
	int width = 1280;
	int height = 720;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto fileReaderProps = FileReaderModuleProps("./data/re3_filtered", 0, -1);
	fileReaderProps.fps = 24;
	fileReaderProps.readLoop = false;

	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(width, height));
	fileReader->addOutputPin(encodedImageMetadata);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 10, 24, "./data/testOutput/mp4_videos/");
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 100;
	auto mp4WriterSink = boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));
	fileReader->setNext(mp4WriterSink);

	boost::filesystem::path dir("./data/testOutput/outFrames/" );

	auto mp4ReaderProps = Mp4ReaderSourceProps("./data/testOutput/mp4_videos/", false,10);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto imageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	mp4Reader->addOutPutPin(imageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	boost::filesystem::path file("frame_??????.jpg");
	boost::filesystem::path full_path = dir / file;
	LOG_INFO << full_path;
	auto fileWriterProps = FileWriterModuleProps(full_path.string());
	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(fileWriterProps));
	std::vector<std::string> encodedImagePin;
	encodedImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE);
	mp4Reader->setNext(fileWriter, encodedImagePin);

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
BOOST_AUTO_TEST_SUITE_END()
