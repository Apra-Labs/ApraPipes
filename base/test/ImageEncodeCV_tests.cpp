#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"

#include "test_utils.h"
#include "ImageEncoderCV.h"
#include "PipeLine.h"
#include "StatSink.h"
#include "SimpleControlModule.h"

BOOST_AUTO_TEST_SUITE(ImageEncodeCV_tests)

BOOST_AUTO_TEST_CASE(mono1_1920x960)
{	

    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1920,960,ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);
	
	auto encode = boost::shared_ptr<ImageEncoderCV>(new ImageEncoderCV(ImageEncoderCVProps()));
	fileReader->setNext(encode);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encode->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(encode->init());
	// BOOST_TEST(copy->init());
	BOOST_TEST(sink->init());	
	
	fileReader->step();
	encode->step();
	// copy->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/IMAGEencode1.jpg", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}
BOOST_AUTO_TEST_CASE(color_rgb_1280x720)
{	

    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1280,720,ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);
	
	auto encode = boost::shared_ptr<ImageEncoderCV>(new ImageEncoderCV(ImageEncoderCVProps()));
	fileReader->setNext(encode);
				
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encode->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(encode->init());
	// BOOST_TEST(copy->init());
	BOOST_TEST(sink->init());	
	
	fileReader->step();
	encode->step();
	// copy->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/IMAGEencodergb.jpg", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}
BOOST_AUTO_TEST_CASE(color_bgra_1920x960)
{	

    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_1920x960_BGRA.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1920,960,ImageMetadata::ImageType::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);
	
	auto encode = boost::shared_ptr<ImageEncoderCV>(new ImageEncoderCV(ImageEncoderCVProps()));
	fileReader->setNext(encode);
				
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encode->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(encode->init());
	// BOOST_TEST(copy->init());
	BOOST_TEST(sink->init());	
	
	fileReader->step();
	encode->step();
	// copy->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/IMAGEencodergbas.jpg", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(MONO_profile, *boost::unit_test::disabled())
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

	
	auto m2 = boost::shared_ptr<Module>(new ImageEncoderCV(ImageEncoderCVProps()));
	fileReader->setNext(m2);
	
	
	auto outputPinId = m2->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];
	
	
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
	logprops.logLevel = boost::log::trivial::severity_level::error;
	Logger::initLogger(logprops);

	
	auto width = 1280;
	auto height = 720;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, CV_8UC3, width*3, CV_8U, FrameMetadata::HOST));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	ImageEncoderCVProps encoderProps;
	encoderProps.enableHealthCallBack = true;
	encoderProps.healthUpdateIntervalInSec = 10;
	auto m2 = boost::shared_ptr<ImageEncoderCV>(new ImageEncoderCV(encoderProps));
	fileReader->setNext(m2);
	
	
	auto outputPinId = m2->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];

	auto controlProps = SimpleControlModuleProps();
	boost::shared_ptr<SimpleControlModule> mControl = boost::shared_ptr<SimpleControlModule>(new SimpleControlModule(controlProps));
	
	StatSinkProps statSinkProps;
	// statSinkProps.logHealth = true;
	// statSinkProps.logHealthFrequency = 10;
	auto statSink = boost::shared_ptr<Module>(new StatSink(statSinkProps));
	m2->setNext(statSink);

	auto p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);
	p->addControlModule(mControl);
	p->init();
	mControl->init();
	// If you want error callbackand health callback to work with a module, registering it with control is mandatory.
	mControl->enrollModule("Encode", m2);
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

	
	auto m2 = boost::shared_ptr<Module>(new ImageEncoderCV(ImageEncoderCVProps()));
	fileReader->setNext(m2);
	
	
	auto outputPinId = m2->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];
	
	
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
