#include <boost/test/unit_test.hpp>
#include "Logger.h"
#include "Frame.h"
#include "PipeLine.h"
#include "test_utils.h"
#include "FileReaderModule.h"
#include "Mp4WriterSink.h"
#include "FramesMuxer.h"
#include "StatSink.h"
#include "EncodedImageMetadata.h"
#include "Mp4VideoMetadata.h"
#include "H264Metadata.h"
#include "H264Utils.h"
#include "ExternalSinkModule.h"

BOOST_AUTO_TEST_SUITE(mp4WriterSink_tests)

void writeH264(bool readLoop, int sleepSeconds, std::string outFolderPath, int chunkTime = 1)
{
	int width = 704;
	int height = 576;

	std::string inFolderPath = "./data/h264_data/";

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
	fileReaderProps.fps = 100;
	fileReaderProps.readLoop = readLoop;

	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(width, height));
	fileReader->addOutputPin(h264ImageMetadata);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(chunkTime, 1, 100, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 100;
	auto mp4WriterSink = boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));
	fileReader->setNext(mp4WriterSink);

	// #Dec_27_Review - do manual init, step and use saveorcompare

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	LOG_ERROR << "processing folder <" << inFolderPath << ">";
	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(sleepSeconds);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}
void write(std::string inFolderPath, std::string outFolderPath, int width, int height, int chunkTime = 1)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
	fileReaderProps.fps = 24;
	fileReaderProps.readLoop = true;

	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(width, height));
	fileReader->addOutputPin(encodedImageMetadata);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(chunkTime, 10, 24, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 100;
	auto mp4WriterSink = boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));
	fileReader->setNext(mp4WriterSink);

	// #Dec_27_Review - do manual init, step and use saveorcompare

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	LOG_ERROR << "processing folder <" << inFolderPath << ">";
	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(120);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

void write_metadata(std::string inFolderPath, std::string outFolderPath, std::string metadataPath, int width, int height, int fps)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
	fileReaderProps.fps = 24;
	fileReaderProps.readLoop = true;

	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(width, height));
	fileReader->addOutputPin(encodedImageMetadata);


	auto fileReaderProps2 = FileReaderModuleProps(metadataPath, 0, -1);
	fileReaderProps2.fps = 24;
	fileReaderProps2.readLoop = true;
	auto metadataReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps2));
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1_0"));
	metadataReader->addOutputPin(mp4Metadata);

	auto readerMuxer = boost::shared_ptr<Module>(new FramesMuxer());
	fileReader->setNext(readerMuxer);
	metadataReader->setNext(readerMuxer);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 10, fileReaderProps.fps, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 100;
	auto mp4WriterSink = boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));
	readerMuxer->setNext(mp4WriterSink);

	// #Dec_27_Review - do manual init, step and use saveorcompare

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);
	p->appendModule(metadataReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	LOG_ERROR << "processing folder <" << inFolderPath << ">";
	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(10);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(jpg_rgb_24_to_mp4v)
{
	int width = 960;
	int height = 480;

	std::string inFolderPath = "./data/streamer_frames";
	std::string outFolderPath = "./data/testOutput/mp4_videos/rgb_24bpp/";

	write(inFolderPath, outFolderPath, width, height);
}

BOOST_AUTO_TEST_CASE(jpg_mono_8_to_mp4v)
{
	int width = 1280;
	int height = 720;

	std::string inFolderPath = "./data/re3_filtered_mono";
	std::string outFolderPath = "./data/testOutput/mp4_videos/mono_8bpp/";

	write(inFolderPath, outFolderPath, width, height);
}

BOOST_AUTO_TEST_CASE(jpg_mono_8_to_mp4v_metadata, *boost::unit_test::disabled())
{
	int width = 1280;
	int height = 720;

	std::string inFolderPath = "./data/re3_filtered_mono";
	std::string outFolderPath = "./data/testOutput/mp4_videos/mono_metadata_video/";
	std::string metadataPath = "./data/metadata/";

	write_metadata(inFolderPath, outFolderPath, metadataPath, width, height, 30);
}

BOOST_AUTO_TEST_CASE(jpeg_metadata, *boost::unit_test::disabled())
{
	/* metadata, RGB, 24bpp, 960x480 */
	int width = 1280;
	int height = 720;
	int fps = 100;

	std::string inFolderPath = "./data/re3_filtered";
	std::string outFolderPath = "./data/testOutput/mp4_videos/rgb_metadata_video";
	std::string metadataPath = "./data/metadata/";

	write_metadata(inFolderPath, outFolderPath, metadataPath, width, height, fps);
}

BOOST_AUTO_TEST_CASE(setgetprops_jpeg)
{
	int width = 1280;
	int height = 720;

	std::string inFolderPath = "./data/re3_filtered_mono";
	std::string outFolderPath = "./data/testOutput/mp4_videos/mono_8bpp/prop1";
	std::string changedOutFolderPath = "./data/testOutput/mp4_videos/mono_8bpp/prop2";

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
	fileReaderProps.fps = 30;
	fileReaderProps.readLoop = true;

	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(width, height));
	fileReader->addOutputPin(encodedImageMetadata);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 1, 30, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 100;
	auto mp4WriterSink = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps));
	fileReader->setNext(mp4WriterSink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	LOG_ERROR << "processing folder <" << inFolderPath << ">";
	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(20);

	Mp4WriterSinkProps propschange = mp4WriterSink->getProps();
	propschange.chunkTime = 2;
	propschange.baseFolder = changedOutFolderPath;
	mp4WriterSink->setProps(propschange);

	Test_Utils::sleep_for_seconds(130);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(h264_to_mp4v, *boost::unit_test::disabled())
{
	std::string outFolderPath = "./data/testOutput/mp4_videos/h264_videos/";
	writeH264(false, 10, outFolderPath);
}

BOOST_AUTO_TEST_CASE(h264_to_mp4v_chunking)
{
	std::string outFolderPath = "./data/testOutput/mp4_videos/h264_videos/";
	writeH264(true, 130, outFolderPath);
}

BOOST_AUTO_TEST_CASE(h264_metadata, *boost::unit_test::disabled())
{
	int width = 704;
	int height = 576;

	std::string inFolderPath = "./data/h264_data/";
	std::string outFolderPath = "./data/testOutput/mp4_videos/h264_metadata/";
	std::string metadataPath = "./data/metadata/";

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
	fileReaderProps.fps = 24;
	fileReaderProps.readLoop = false;

	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto encodedImageMetadata = framemetadata_sp(new H264Metadata(width, height));
	fileReader->addOutputPin(encodedImageMetadata);


	auto fileReaderProps2 = FileReaderModuleProps(metadataPath, 0, -1);
	fileReaderProps2.fps = 24;
	fileReaderProps2.readLoop = true;
	auto metadataReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps2));
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1_0"));
	metadataReader->addOutputPin(mp4Metadata);

	auto readerMuxer = boost::shared_ptr<Module>(new FramesMuxer());
	fileReader->setNext(readerMuxer);
	metadataReader->setNext(readerMuxer);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 1, fileReaderProps.fps, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 100;
	auto mp4WriterSink = boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));
	readerMuxer->setNext(mp4WriterSink);

	// #Dec_27_Review - do manual init, step and use saveorcompare

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);
	p->appendModule(metadataReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	LOG_ERROR << "processing folder <" << inFolderPath << ">";
	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(10);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(parsenalu, *boost::unit_test::disabled())
{
	int width = 640;
	int height = 360;
	
	std::string inFolderPath = "./data/h264_frames/";

	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
	fileReaderProps.fps = 24;
	fileReaderProps.readLoop = false;

	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));

	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(width, height));
	fileReader->addOutputPin(h264ImageMetadata);
	
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	fileReader->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);

	for (int f = 0; f < 31; f++)
	{
		fileReader->step();
		auto frames = sink->pop();
		auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::H264_DATA);
		auto mFrameBuffer = const_buffer(frame->data(), frame->size());
		auto ret = H264Utils::parseNalu(mFrameBuffer);
		const_buffer spsBuff, ppsBuff;
		short typeFound;
		tie(typeFound, spsBuff, ppsBuff) = ret;
		auto spsBuffer = static_cast<const char*>(spsBuff.data());
		auto ppsBuffer = static_cast<const char*>(ppsBuff.data());
		
		std::cout << "frame " << f << std::endl;
		if(f==0)
		{
			BOOST_TEST(typeFound == 5);
			BOOST_TEST(spsBuffer[0] == 0x67);
			BOOST_TEST(ppsBuffer[0] == 0x68);
			//here test for sps and pps using 67 and 68 and sizes
		}
		else if (f == 30)
		{
			BOOST_TEST(typeFound == 5);
			BOOST_TEST(spsBuffer == nullptr);
			BOOST_TEST(ppsBuffer == nullptr);
			BOOST_TEST(spsBuff.size() == 0);
			BOOST_TEST(ppsBuff.size() == 0);
			//here test for missing sps and pps using NULL (data) 0 (size)
		}
		else {
			BOOST_TEST(typeFound == 0);
			BOOST_TEST(spsBuffer == nullptr);
			BOOST_TEST(ppsBuffer == nullptr);
			BOOST_TEST(spsBuff.size() == 0);
			BOOST_TEST(ppsBuff.size() == 0);
			//here test for missing sps and pps using NULL (data) 0 (size)
		}
	}
	
}

BOOST_AUTO_TEST_CASE(setgetprops_h264)
{
	int width = 704;
	int height = 576;

	std::string inFolderPath = "./data/h264_data";
	std::string outFolderPath = "./data/testOutput/mp4_videos/props/prop1";
	std::string changedOutFolderPath = "./data/testOutput/mp4_videos/props/prop2";

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
	fileReaderProps.fps = 30;
	fileReaderProps.readLoop = true;

	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(width, height));
	fileReader->addOutputPin(h264ImageMetadata);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 1, 30, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 100;
	auto mp4WriterSink = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps));
	fileReader->setNext(mp4WriterSink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	LOG_ERROR << "processing folder <" << inFolderPath << ">";
	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(20);
	
	Mp4WriterSinkProps propschange = mp4WriterSink->getProps();
	propschange.chunkTime = 2;
	propschange.baseFolder = changedOutFolderPath;
	mp4WriterSink->setProps(propschange);

	Test_Utils::sleep_for_seconds(130);
	
	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(single_file_given_name_jpeg)
{
	// custom name is only supported while writing to single video file (chunktime = UINT32_MAX).
	int width = 1280;
	int height = 720;

	std::string inFolderPath = "./data/re3_filtered_mono";
	std::string outFolderPath = "./data/testOutput/mp4_videos/mono_8bpp/apra.mp4";

	write(inFolderPath, outFolderPath, width, height, UINT32_MAX);
}

BOOST_AUTO_TEST_CASE(single_file_given_name_h264)
{
	// custom name is only supported while writing to single video file (chunktime = UINT32_MAX).
	std::string outFolderPath = "./data/testOutput/mp4_videos/h264_videos/apraH264.mp4";

	writeH264(true,80,outFolderPath, UINT32_MAX);
}

BOOST_AUTO_TEST_SUITE_END()