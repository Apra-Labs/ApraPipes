#include <boost/test/unit_test.hpp>
 
#include "Logger.h"
#include "AIPExceptions.h"
#include "PipeLine.h"
#include "test_utils.h"
#include "FileReaderModule.h"
#include "CudaCommon.h"
#include "Mp4ReaderSource.h"
#include "FileWriterModule.h"
#include "StatSink.h"
#include "FrameMetadata.h"
#include "EncodedImageMetadata.h"
#include "H264Metadata.h"
#include "Mp4VideoMetadata.h"
 
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
    bool process(frame_container &frames) {
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
 
void read_video_extract_frames(std::string videoPath, std::string outPath, int width, int height, int uniqMetadata = 0, bool parseFS = true)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);
 
    boost::filesystem::path dir(outPath);
 
    auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS);
    auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
    auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0,0));
    mp4Reader->addOutPutPin(encodedImageMetadata);
    auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata());
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
 
    //metaSink
    // #Dec_27_Review - use manual init, step, use save or compare
    // #Dec_27_Review - saveorcompare both jpeg image as well as metadata
 
    boost::shared_ptr<PipeLine> p;
    p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
    p->appendModule(mp4Reader);
 
    if (!p->init())
    {
        throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
    }
    p->run_all_threaded();
 
    boost::this_thread::sleep_for(boost::chrono::seconds(300));
 
    p->stop();
    p->term();
 
    p->wait_for_all();
 
    p.reset();
}
 
void random_seek_video(std::string skipDir, uint64_t skipTS, std::string startingVideoPath, std::string outPath, int width, int height)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);
 
    boost::filesystem::path dir(outPath);
 
    auto mp4ReaderProps = Mp4ReaderSourceProps(startingVideoPath, false);
    auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
    auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
    mp4Reader->addOutPutPin(encodedImageMetadata);
    auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata());
    mp4Reader->addOutPutPin(mp4Metadata);
 
    mp4ReaderProps.skipDir = skipDir;
    
 
    boost::filesystem::path file("frame_??????.jpg");
    boost::filesystem::path full_path = dir / file;
    LOG_INFO << full_path;
    auto fileWriterProps = FileWriterModuleProps(full_path.string());
    auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(fileWriterProps));
    std::vector<std::string> encodedImagePin;
    encodedImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE);
    mp4Reader->setNext(fileWriter, encodedImagePin);

    
 
    boost::shared_ptr<PipeLine> p;
    p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
    p->appendModule(mp4Reader);
 
    if (!p->init())
    {
        throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
    }

    mp4Reader->setProps(mp4ReaderProps);
    mp4Reader->randomSeek(skipTS);

    p->run_all_threaded();
 
    boost::this_thread::sleep_for(boost::chrono::seconds(50000));
 
    p->stop();
    p->term();
 
    p->wait_for_all();
 
    p.reset();
}
 
BOOST_AUTO_TEST_CASE(mp4v_to_rgb_24_jpg)
{
    /* no metadata, rbg, 24bpp, 960x480 */
    std::string videoPath = "./data/Mp4_videos/streamer_mp4v_RGB.mp4";
    std::string outPath = "data/testOutput/outFrames";
    bool parseFS = true;
    read_video_extract_frames(videoPath, outPath, 960, 480, parseFS);
}
 
BOOST_AUTO_TEST_CASE(mp4v_to_mono_8_jpg)
{
    /* no metadata, mono, 8bpp, 448x608 */
    std::string videoPath = "./data/Mp4_videos/jog_video/1666943213667.mp4";
    std::string outPath = "data/testOutput/outFrames";
    bool parseFS = false;
    read_video_extract_frames(videoPath, outPath, 160, 80, parseFS);
}
 
BOOST_AUTO_TEST_CASE(mp4v_read_metadata_jpg)//, *boost::unit_test::disabled())
{
    /* metadata, rgb, 24bpp, 448x608 */
    std::string videoPath = "data/Mp4_videos/jpg_video_metada/20220928/0014/1666949168743.mp4";// poc_mp4v_meta.mp4";
    std::string outPath = "./data/testOutput/outFrames";
    bool parseFS = false;
    read_video_extract_frames(videoPath, outPath, 160, 80, parseFS);
}
 
BOOST_AUTO_TEST_CASE(fs_parsing)
{
    /* file structure parsing test */
    std::string videoPath = "data/Mp4_videos/jpg_video/1666943213667.mp4";
    std::string outPath = "data/testOutput/outFrames";
    bool parseFS = true;
    read_video_extract_frames(videoPath, outPath, 160, 80, 5, parseFS);
}
 
BOOST_AUTO_TEST_CASE(random_seek)
{
    std::string skipDir = "data/Mp4_videos/jpg_video_metada/";
    std::string startingVideoPath = "data/Mp4_videos/jpg_video_metada/20220928/0014/1666949168743.mp4";//"data/mp4_videos/proto_mp4_tests/chunked/final/20220231/0017/1648728769245.mp4";
    uint64_t skipTS = 1666949171743;
    random_seek_video(skipDir, skipTS, startingVideoPath, "data/testOutput/outFrames", 160, 80);// 960, 480);
}
 
BOOST_AUTO_TEST_CASE(random_seek_no_metadata)
{
    std::string skipDir = "data/Mp4_videos/mono_8bpp_2/";
    std::string startingVideoPath = "data/Mp4_videos/mono_8bpp_2/20220308/0018/1649421680651.mp4";
    uint64_t skipTS = 1649421689651;
    random_seek_video(skipDir, skipTS, startingVideoPath, "data/testOutput/outFrames", 448, 608);
}
 
void read_video_decode_frames(std::string startingVideoPath, int width, int height)
{
    auto mp4ReaderProps = Mp4ReaderSourceProps(startingVideoPath, true, 5000000, 50000);
    auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
    auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0,0));
    mp4Reader->addOutputPin(encodedImageMetadata);
    auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata());
    mp4Reader->addOutputPin(mp4Metadata);
 
    boost::shared_ptr<Module> decoder;
    framemetadata_sp rawImageMetadata;
    rawImageMetadata = framemetadata_sp(new RawImageMetadata());
    auto rawImagePin = decoder->addOutputPin(rawImageMetadata);
    std::vector<std::string> encodedImagePin;
    encodedImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE);
    mp4Reader->setNext(decoder, encodedImagePin);
 
    boost::filesystem::path dir("data/testOutput/mp4reader");
    boost::filesystem::path file("frame_??????.raw");
    boost::filesystem::path full_path = dir / file;
    auto fileWriterProps = FileWriterModuleProps(full_path.string());
    auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(fileWriterProps));
    decoder->setNext(fileWriter);
 
    boost::shared_ptr<PipeLine> p;
    p = boost::shared_ptr<PipeLine>(new PipeLine("mp4reader"));
    p->appendModule(mp4Reader);
 
    if (!p->init())
    {
        throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
    }
    p->run_all_threaded();
 
    boost::this_thread::sleep_for(boost::chrono::seconds(50000));
 
    p->stop();
    p->term();
 
    p->wait_for_all();
 
    p.reset();
}
 
BOOST_AUTO_TEST_CASE(read_mp4_decode_jpeg)
{
    read_video_decode_frames("data/mp4_videos/20220329/0002/1651178801859.mp4", 448, 608);
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
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata());
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

	// #Dec_27_Review - use manual init, step, use save or compare
	// #Dec_27_Review - saveorcompare both jpeg image as well as metadata

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}
	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	
	//Mp4ReaderSourceProps propsChange = mp4Reader->getProps();
    Mp4ReaderSourceProps propsChange(changedVideoPath, false);
	//propsChange.parseFS = false;
	//propsChange.videoPath = changedVideoPath;
//	propsChange.skipDir = boost::filesystem::path(changedVideoPath).parent_path().parent_path().parent_path().string();
	mp4Reader->setProps(propsChange);

	/*p->step();
	p->step();
	p->step();
	p->step();*/



	boost::this_thread::sleep_for(boost::chrono::seconds(10));



	p->stop();
	p->term();

	p->wait_for_all();

	p.reset();
}

BOOST_AUTO_TEST_CASE(getSetProps_tiny)
{
	std::string videoPath = "./data/mp4_videos/hetro/20220507/0018/1654605471000.mp4";
	std::string outPath = "C:/Users/developer/Aprapipeszaki/mp4reader/data/testOutput/mp4_videos/outFrames/";
	std::string changedVideoPath = "./data/mp4_videos/hetro1/20220507/0018/1654605480300.mp4";

	bool parseFS = true;
	boost::filesystem::path dir(outPath);

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	mp4Reader->addOutputPin(encodedImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata());
	mp4Reader->addOutputPin(mp4Metadata);

	boost::filesystem::path file("frame_??????.jpg");
	boost::filesystem::path full_path = dir / file;
	auto fileWriterProps = FileWriterModuleProps(full_path.string());
	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(fileWriterProps));
	std::vector<std::string> encodedImagePin;
	encodedImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE);
	mp4Reader->setNext(fileWriter, encodedImagePin);

	StatSinkProps statSinkProps;
	statSinkProps.logHealth = true;
	statSinkProps.logHealthFrequency = 100;
	auto statSink = boost::shared_ptr<Module>(new StatSink(statSinkProps));
	mp4Reader->setNext(statSink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("mp4reader"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}
	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	Mp4ReaderSourceProps propsChange(changedVideoPath,true);// = mp4Reader->getProps();
	//propsChange.biggerFrameSize = 400000;
	mp4Reader->setProps(propsChange);

	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	
	p->stop();
	p->term();

	p->wait_for_all();

	p.reset();
}
BOOST_AUTO_TEST_CASE(mp4v_to_h264)
{
    std::string videoPath = "./data/h264_video/1664523927076.mp4";
    std::string outPath = "./data/testOutput/outFrames/";
    //std::string changedVideoPath = "./data/mp4_videos/hetro1/20220507/0018/1654605480300.mp4";
    bool parseFS = false;
    int uniqMetadata = 0;
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    boost::filesystem::path dir(outPath);

    auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS);
    auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
    auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
    mp4Reader->addOutPutPin(h264ImageMetadata);
    auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata());
    mp4Reader->addOutputPin(mp4Metadata);

    boost::filesystem::path file("frame_??????.h264");
    boost::filesystem::path full_path = dir / file;
    LOG_INFO << full_path;
    auto fileWriterProps = FileWriterModuleProps(full_path.string());
    auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(fileWriterProps));
    std::vector<std::string> encodedImagePin;
    encodedImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::H264_DATA);
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

    // #Dec_27_Review - use manual init, step, use save or compare
    // #Dec_27_Review - saveorcompare both jpeg image as well as metadata

    boost::shared_ptr<PipeLine> p;
    p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
    p->appendModule(mp4Reader);

    if (!p->init())
    {
        throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
    }
    p->run_all_threaded();

    boost::this_thread::sleep_for(boost::chrono::seconds(50));

    p->stop();
    p->term();

    p->wait_for_all();

    p.reset();
}
BOOST_AUTO_TEST_SUITE_END()
