#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "test_utils.h"
#include "StatSink.h"
#include "PipeLine.h"
#include "FileWriterModule.h"
#include "BrightnessContrastControl.h"

BOOST_AUTO_TEST_SUITE(brightnes_contrast_tests)

BOOST_AUTO_TEST_CASE(mono)
{

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1920, 960, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto brightnessControl = boost::shared_ptr<BrightnessContrastControl>(new BrightnessContrastControl(BrightnessContrastControlProps(0.5, 40)));
	fileReader->setNext(brightnessControl);

	auto m2 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	brightnessControl->setNext(m2);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(brightnessControl->init());
	BOOST_TEST(m2->init());

	fileReader->step();
	brightnessControl->step();
	auto frames = m2->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/brightnessmono.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(bgra)
{
	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	auto width = 1920;
	auto height = 960;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_1920x960_BGRA.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto brightnessControl = boost::shared_ptr<BrightnessContrastControl>(new BrightnessContrastControl(BrightnessContrastControlProps(0.5, 40)));
	fileReader->setNext(brightnessControl);

	auto m2 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	brightnessControl->setNext(m2);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(brightnessControl->init());
	BOOST_TEST(m2->init());

	fileReader->step();
	brightnessControl->step();
	auto frames = m2->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/brightnessbgra.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(getSetProps)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto brightnessControl = boost::shared_ptr<BrightnessContrastControl>(new BrightnessContrastControl(BrightnessContrastControlProps(0.5, 40)));
	fileReader->setNext(brightnessControl);

	auto m2 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	brightnessControl->setNext(m2);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(brightnessControl->init());
	BOOST_TEST(m2->init());

	{
		fileReader->step();
		brightnessControl->step();
		auto frames = m2->pop();
		BOOST_TEST(frames.size() == 1);
		auto outputFrame = frames.cbegin()->second;
		BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
		Test_Utils::saveOrCompare("./data/testOutput/brightnessgetset1.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
	}

	BrightnessContrastControlProps props12(2, 40);

	brightnessControl->setProps(props12);
	brightnessControl->step();

	{
		fileReader->step();
		brightnessControl->step();
		auto frames = m2->pop();
		BOOST_TEST(frames.size() == 1);
		auto outputFrame = frames.cbegin()->second;
		BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
		Test_Utils::saveOrCompare("./data/testOutput/brightnessgetset2.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
	}
}

BOOST_AUTO_TEST_SUITE_END()