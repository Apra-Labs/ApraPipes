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
#include "ImageDecoderCV.h"

BOOST_AUTO_TEST_SUITE(jpegdecodercv_tests)

BOOST_AUTO_TEST_CASE(mono1_1920x960)
{	
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.jpg")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	fileReader->addOutputPin(metadata);

	

	auto decoder = boost::shared_ptr<ImageDecoderCV>(new ImageDecoderCV(ImageDecoderCVProps()));
    auto metadata2 = framemetadata_sp(new RawImageMetadata());
    decoder->addOutputPin(metadata2);

	fileReader->setNext(decoder);

	

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	decoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(decoder->init());
	// BOOST_TEST(copy->init());
	BOOST_TEST(sink->init());	
	
	fileReader->step();
	decoder->step();
	// copy->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/jpegdecodercv_tests_mono_1920x960.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(color_rgb_1920x960)
{	
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.jpg")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	fileReader->addOutputPin(metadata);

	// auto stream = cudastream_sp(new ApraCudaStream);

	auto decoder = boost::shared_ptr<ImageDecoderCV>(new ImageDecoderCV(ImageDecoderCVProps()));
	auto metadata2 = framemetadata_sp(new RawImageMetadata(1920,960,ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    decoder->addOutputPin(metadata2);

    fileReader->setNext(decoder);


	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    


	decoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(decoder->init());
	// BOOST_TEST(copy->init());
	BOOST_TEST(sink->init());	
	
	fileReader->step();
	decoder->step();
	// copy->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/jpegdecodercv_tests_rgb_1920x960.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
