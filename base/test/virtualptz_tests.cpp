#include "stdafx.h"
#include <boost/test/unit_test.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include "VirtualPTZ.h"
#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "MetadataHints.h"
#include "test_utils.h"
#include "ImageResizeCV.h"
#include "RotateCV.h"
#include "FramesMuxer.h"
#include "PipeLine.h"
#include "StatSink.h"
#include "ImageOverlayCPU.h"
#include "TextOverlayCPU.h"
#include "OpencvWebcam.h"
#include "StatSink.h"
#include "PipeLine.h"
#include "FileWriterModule.h"
#include "CudaMemCopy.h"
#include "JPEGEncoderNVJPEG.h"
#include "ImageEncoderCV.h"
#include "BrightnessContrastControl.h"
#include "ImageOverlay.h"
#include "WebCamSrc.h"
#ifdef ARM64
BOOST_AUTO_TEST_SUITE(virtual_ptz_tests, *boost::unit_test::disabled())
#else
BOOST_AUTO_TEST_SUITE(virtual_ptz_tests)
#endif

BOOST_AUTO_TEST_CASE(mono)
{

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1920, 960, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto vPtz = boost::shared_ptr<VirtualPTZ>(new VirtualPTZ(VirtualPTZProps(0.09, 0.500, 0.00, 0.00)));
	fileReader->setNext(vPtz);

	auto Vptzprops = vPtz->getProps();

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	vPtz->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(vPtz->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	vPtz->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/vPtztest4.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(rgb)
{

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto vPtz = boost::shared_ptr<VirtualPTZ>(new VirtualPTZ(VirtualPTZProps(0.9, 0.900, 0.90, 0.90)));
	fileReader->setNext(vPtz);

	auto Vptzprops = vPtz->getProps();

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	vPtz->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(vPtz->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	vPtz->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/vPtztest8.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(testGetSetProps)
{

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1920, 960, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto vPtz = boost::shared_ptr<VirtualPTZ>(new VirtualPTZ(VirtualPTZProps(0.300, 0.300, 0.100, 0.100)));
	fileReader->setNext(vPtz);

	auto Vptzprops = vPtz->getProps();

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	vPtz->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(vPtz->init());
	BOOST_TEST(sink->init());

	{
		fileReader->step();
		vPtz->step();
		auto frames = sink->pop();
		BOOST_TEST(frames.size() == 1);
		auto outputFrame = frames.cbegin()->second;
		BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
		Test_Utils::saveOrCompare("./data/testOutput/vPtztest2.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
	}

	VirtualPTZProps props12(0.950, 0.950, 0.90, 0.90);

	vPtz->setProps(props12);
	vPtz->step();

	{
		auto Vptzprops1 = vPtz->getProps();
		fileReader->step();
		vPtz->step();
		auto frames = sink->pop();
		BOOST_TEST(frames.size() == 1);
		auto outputFrame = frames.cbegin()->second;
		BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
		Test_Utils::saveOrCompare("./data/testOutput/vPtztest7.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
	}
}

BOOST_AUTO_TEST_SUITE_END()