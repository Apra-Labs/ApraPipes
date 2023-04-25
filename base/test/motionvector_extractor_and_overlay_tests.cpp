#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "MotionVectorExtractor.h"
#include "OverlayMotionVectors.h"
#include "ExternalSinkModule.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "H264Metadata.h"
#include "FileWriterModule.h"
#include "ImageViewerModule.h"
#include "Logger.h"
#include "RTSPClientSrc.h"

BOOST_AUTO_TEST_SUITE(overlay_motion_vectors_tests)

struct rtsp_client_tests_data {
	rtsp_client_tests_data()
	{
		outFile = string("./data/testOutput/bunny.h264");
		Test_Utils::FileCleaner fc;
		fc.pathsOfFiles.push_back(outFile); //clear any occurance before starting the tests
	}
	string outFile;
	string empty;
};

void motionVectorExtract(MotionVectorExtractorProps::MVExtractMethod MvExtract)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	FileReaderModuleProps fileReaderProps("./data/h264_data/FVDO_Freeway_4cif_???.H264");
	fileReaderProps.fps = 30;
	fileReaderProps.readLoop = false;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new H264Metadata(0, 0));
	fileReader->addOutputPin(metadata);

	auto motionExtractor = boost::shared_ptr<Module>(new MotionVectorExtractor(MotionVectorExtractorProps(MvExtract)));
	fileReader->setNext(motionExtractor);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	motionExtractor->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(motionExtractor->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);

	for (int i = 0; i < 231; i++)
	{
		fileReader->step();
		motionExtractor->step();
		auto frames = sink->pop();
		auto outFrame = frames.begin()->second;
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::MOTION_VECTOR_DATA);
	}
}

void motionVectorExtractAndOverlay(MotionVectorExtractorProps::MVExtractMethod MvExtract, OverlayMotionVectorProps::MVOverlayMethod MvOverlay)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	bool enableOverlay = true;

	FileReaderModuleProps fileReaderProps("./data/h264_data/FVDO_Freeway_4cif_???.H264");
	fileReaderProps.fps = 30;
	fileReaderProps.readLoop = true;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	fileReader->addOutputPin(h264ImageMetadata);

	auto motionExtractor = boost::shared_ptr<MotionVectorExtractor>(new MotionVectorExtractor(MotionVectorExtractorProps(MvExtract,enableOverlay)));
	fileReader->setNext(motionExtractor);

	auto overlayMotionVector = boost::shared_ptr<Module>(new OverlayMotionVector(OverlayMotionVectorProps(MvOverlay)));
	motionExtractor->setNext(overlayMotionVector);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	overlayMotionVector->setNext(sink);


	BOOST_TEST(fileReader->init());
	BOOST_TEST(motionExtractor->init());
	BOOST_TEST(overlayMotionVector->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);

	fileReader->step();
	motionExtractor->step();
	overlayMotionVector->step();
	for (int i = 0; i < 230; i++)
	{
		fileReader->step();
		motionExtractor->step();
		overlayMotionVector->step();
		auto frames = sink->pop();
		auto outFrame = frames.begin()->second;
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	}
}

void motionVectorExtractAndOverlaySetProps(MotionVectorExtractorProps::MVExtractMethod MvExtract, OverlayMotionVectorProps::MVOverlayMethod MvOverlay)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	FileReaderModuleProps fileReaderProps("./data/h264_data/FVDO_Freeway_4cif_???.H264");
	fileReaderProps.fps = 30;
	fileReaderProps.readLoop = true;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	fileReader->addOutputPin(h264ImageMetadata);

	auto motionExtractor = boost::shared_ptr<MotionVectorExtractor>(new MotionVectorExtractor(MotionVectorExtractorProps(MvExtract)));
	fileReader->setNext(motionExtractor);

	auto overlayMotionVector = boost::shared_ptr<Module>(new OverlayMotionVector(OverlayMotionVectorProps(MvOverlay)));
	motionExtractor->setNext(overlayMotionVector);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	overlayMotionVector->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	MotionVectorExtractorProps propsChange(MvExtract,true);
	motionExtractor->setProps(propsChange);

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

void motionVectorExtractAndOverlay_Render(MotionVectorExtractorProps::MVExtractMethod MvExtract, OverlayMotionVectorProps::MVOverlayMethod MvOverlay) //, *boost::unit_test::disabled()) 
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	bool overlayFrames = true;

	FileReaderModuleProps fileReaderProps("./data/h264_data/FVDO_Freeway_4cif_???.H264");
	fileReaderProps.fps = 30;
	fileReaderProps.readLoop = true;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	fileReader->addOutputPin(h264ImageMetadata);

	auto motionExtractor = boost::shared_ptr<MotionVectorExtractor>(new MotionVectorExtractor(MotionVectorExtractorProps(MvExtract,overlayFrames)));
	fileReader->setNext(motionExtractor);

	auto overlayMotionVector = boost::shared_ptr<Module>(new OverlayMotionVector(OverlayMotionVectorProps(MvOverlay)));
	motionExtractor->setNext(overlayMotionVector);

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("MotionVectorsOverlay")));
	overlayMotionVector->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

void rtspCamMotionVectorExtractAndOverlay_Render(MotionVectorExtractorProps::MVExtractMethod MvExtract, OverlayMotionVectorProps::MVOverlayMethod MvOverlay) 
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	rtsp_client_tests_data d;

	bool overlayFrames = false;
	const std::string url = "";
	std::string username = "";
	std::string password = "";
	auto rtspSrc = boost::shared_ptr<Module>(new RTSPClientSrc(RTSPClientSrcProps(url,d.empty,d.empty)));
	auto meta = framemetadata_sp(new H264Metadata());
	rtspSrc->addOutputPin(meta);

	auto motionExtractor = boost::shared_ptr<MotionVectorExtractor>(new MotionVectorExtractor(MotionVectorExtractorProps(MvExtract,overlayFrames)));
	rtspSrc->setNext(motionExtractor);

	auto overlayMotionVector = boost::shared_ptr<Module>(new OverlayMotionVector(OverlayMotionVectorProps(MvOverlay)));
	motionExtractor->setNext(overlayMotionVector);

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("MotionVectorsOverlay")));
	overlayMotionVector->setNext(sink);

	PipeLine p("test");
	p.appendModule(rtspSrc);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(basic_extract_motion_vector_ffmpeg)
{
	motionVectorExtract(MotionVectorExtractorProps::FFMPEG);
}

BOOST_AUTO_TEST_CASE(extract_motion_vectors_and_overlay_ffmpeg)
{
	motionVectorExtractAndOverlay(MotionVectorExtractorProps::FFMPEG, OverlayMotionVectorProps::FFMPEG);
}

BOOST_AUTO_TEST_CASE(extract_motion_vectors_and_overlay_setprops_ffmpeg)
{
	motionVectorExtractAndOverlaySetProps(MotionVectorExtractorProps::FFMPEG, OverlayMotionVectorProps::FFMPEG);
}

BOOST_AUTO_TEST_CASE(extract_motion_vectors_and_overlay_render_ffmpeg, *boost::unit_test::disabled())
{
	motionVectorExtractAndOverlay_Render(MotionVectorExtractorProps::FFMPEG, OverlayMotionVectorProps::FFMPEG);
}

BOOST_AUTO_TEST_CASE(rtspcam_extract_motion_vectors_and_overlay_render_ffmpeg, *boost::unit_test::disabled())
{
	rtspCamMotionVectorExtractAndOverlay_Render(MotionVectorExtractorProps::FFMPEG, OverlayMotionVectorProps::FFMPEG);
}

BOOST_AUTO_TEST_CASE(basic_extract_motion_vector_openh264)
{
	motionVectorExtract(MotionVectorExtractorProps::OPENH264);
}

BOOST_AUTO_TEST_CASE(extract_motion_vectors_and_overlay_openh264)
{
	motionVectorExtractAndOverlay(MotionVectorExtractorProps::OPENH264, OverlayMotionVectorProps::OPENH264);
}

BOOST_AUTO_TEST_CASE(extract_motion_vectors_and_overlay_render_openh264, *boost::unit_test::disabled())
{
	motionVectorExtractAndOverlay_Render(MotionVectorExtractorProps::OPENH264, OverlayMotionVectorProps::OPENH264);
}

BOOST_AUTO_TEST_CASE(rtspcam_extract_motion_vectors_and_overlay_render_openh264, *boost::unit_test::disabled())
{
	rtspCamMotionVectorExtractAndOverlay_Render(MotionVectorExtractorProps::OPENH264, OverlayMotionVectorProps::OPENH264);
}

BOOST_AUTO_TEST_SUITE_END()
