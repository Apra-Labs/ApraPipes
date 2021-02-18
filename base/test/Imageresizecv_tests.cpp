#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "PipeLine.h"
#include "StatSink.h"
#include "test_utils.h"
#include "ImageResizeCV.h"

BOOST_AUTO_TEST_SUITE(Imageresizecv_tests)

BOOST_AUTO_TEST_CASE(mono1_1920x960)
{	

    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1920,960,ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);
	
	auto resize = boost::shared_ptr<ImageResizeCV>(new ImageResizeCV(ImageResizeCVProps(200,200)));
	fileReader->setNext(resize);
	
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	resize->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(resize->init());
	// BOOST_TEST(copy->init());
	BOOST_TEST(sink->init());	
	
	fileReader->step();
	resize->step();
	// copy->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/IMAGERESIZETEST1.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(color_rgb_1280x720)
{	
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1280,720,ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	// auto stream = cudastream_sp(new ApraCudaStream);

	auto resize = boost::shared_ptr<ImageResizeCV>(new ImageResizeCV(ImageResizeCVProps(200,200)));
    fileReader->setNext(resize);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    resize->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(resize->init());
	// BOOST_TEST(copy->init());
	BOOST_TEST(sink->init());	
	
	fileReader->step();
	resize->step();
	// copy->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/IMAGERESIZECVRGB1.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(perf, *boost::unit_test::disabled())
{

	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 3840;
	auto height = 2160;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/4k.yuv")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, 1, CV_8UC1, width, CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	
	auto m2 = boost::shared_ptr<Module>(new ImageResizeCV(ImageResizeCVProps(width >> 1, height >> 1)));
	fileReader->setNext(m2);
	
	
	auto outputPinId = m2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);


	BOOST_TEST(fileReader->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	for (auto i = 0; i < 1; i++)
	{
		fileReader->step();
		m2->step();
		m3->pop();
	}
}
BOOST_AUTO_TEST_CASE(MONO_profile, *boost::unit_test::disabled())
{
	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	
	auto width = 3840;
	auto height = 2160;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/4k.yuv")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, 1, CV_8UC1, width, CV_8U));


	auto rawImagePin = fileReader->addOutputPin(metadata);

	
	auto m2 = boost::shared_ptr<Module>(new ImageResizeCV(ImageResizeCVProps(width >> 1, height >> 1)));
	fileReader->setNext(m2);
	
	
	auto outputPinId = m2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	
	
	StatSinkProps statSinkProps;
	statSinkProps.logHealth = true;
	statSinkProps.logHealthFrequency = 10;
	auto statSink = boost::shared_ptr<Module>(new StatSink(statSinkProps));
	m2->setNext(statSink);

	auto p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);
	p->init();
	p->run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(3000));
	p->stop();
	p->term();
	p->wait_for_all();

}

BOOST_AUTO_TEST_CASE(RGB_profile, *boost::unit_test::disabled())
{
	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	
	auto width = 1280;
	auto height = 720;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, CV_8UC3, width*3, CV_8U, FrameMetadata::HOST));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	
	auto m2 = boost::shared_ptr<Module>(new ImageResizeCV(ImageResizeCVProps(width >> 1, height >> 1)));
	fileReader->setNext(m2);
	
	
	auto outputPinId = m2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	
	
	StatSinkProps statSinkProps;
	statSinkProps.logHealth = true;
	statSinkProps.logHealthFrequency = 10;
	auto statSink = boost::shared_ptr<Module>(new StatSink(statSinkProps));
	m2->setNext(statSink);

	auto p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);
	p->init();
	p->run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(3000));
	p->stop();
	p->term();
	p->wait_for_all();

}

BOOST_AUTO_TEST_CASE(bgra_profile, *boost::unit_test::disabled())
{
	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	
	auto width = 1920;
	auto height = 960;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_1920x960_BGRA.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	
	auto m2 = boost::shared_ptr<Module>(new ImageResizeCV(ImageResizeCVProps(width >> 1, height >> 1)));
	fileReader->setNext(m2);
	
	
	auto outputPinId = m2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	
	
	StatSinkProps statSinkProps;
	statSinkProps.logHealth = true;
	statSinkProps.logHealthFrequency = 10;
	auto statSink = boost::shared_ptr<Module>(new StatSink(statSinkProps));
	m2->setNext(statSink);

	auto p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);
	p->init();
	p->run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(3000));
	p->stop();
	p->term();
	p->wait_for_all();
	
}
BOOST_AUTO_TEST_SUITE_END()
