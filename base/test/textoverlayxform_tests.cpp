#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "MetadataHints.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "StatSink.h"
#include "TextOverlayXForm.h"
#include "StatSink.h"
#include "FileWriterModule.h"
#include "ImageEncoderCV.h"
#ifdef ARM64OLD
BOOST_AUTO_TEST_SUITE(text_overlay_tests, *boost::unit_test::disabled())
#else
BOOST_AUTO_TEST_SUITE(text_overlay_tests)
#endif

BOOST_AUTO_TEST_CASE(mono)
{

    std::string text = "Apra Pipes";
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.raw")));
    auto metadata = framemetadata_sp(new RawImageMetadata(1920, 960, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
    fileReader->addOutputPin(metadata);

    auto textOverlay = boost::shared_ptr<TextOverlayXForm>(new TextOverlayXForm(TextOverlayXFormProps(0.5, text, "UpperRight", false, 30, "FFFFFF", "000000")));
    fileReader->setNext(textOverlay);

    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    textOverlay->setNext(sink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(textOverlay->init());
    BOOST_TEST(sink->init());
    {
        fileReader->step();
        textOverlay->step();
        auto frames = sink->pop();
        BOOST_TEST(frames.size() == 1);
        auto outputFrame = frames.cbegin()->second;
        BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
        Test_Utils::saveOrCompare("./data/testOutput/textOverlaymono.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
    }
}

BOOST_AUTO_TEST_CASE(rgb)
{

    std::string text = "Apra Pipes";
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
    auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    fileReader->addOutputPin(metadata);

    auto textOverlay = boost::shared_ptr<TextOverlayXForm>(new TextOverlayXForm(TextOverlayXFormProps(0.8, text, "BottomLeft", false, 30, "FFFFFF", "000000")));
    fileReader->setNext(textOverlay);

    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    textOverlay->setNext(sink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(textOverlay->init());
    BOOST_TEST(sink->init());
    {
        fileReader->step();
        textOverlay->step();
        auto frames = sink->pop();
        BOOST_TEST(frames.size() == 1);
        auto outputFrame = frames.cbegin()->second;
        BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
        Test_Utils::saveOrCompare("./data/testOutput/textOverlayrgb.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
    }
}

BOOST_AUTO_TEST_CASE(bgra)
{

    std::string text = "Apra Pipes";
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_1920x960_BGRA.raw")));
    auto metadata = framemetadata_sp(new RawImageMetadata(1920, 960, ImageMetadata::ImageType::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
    fileReader->addOutputPin(metadata);

    auto textOverlay = boost::shared_ptr<TextOverlayXForm>(new TextOverlayXForm(TextOverlayXFormProps(1.0, text, "BottomRight", false, 30, "FFFFFF", "000000")));
    fileReader->setNext(textOverlay);

    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    textOverlay->setNext(sink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(textOverlay->init());
    BOOST_TEST(sink->init());
    {
        fileReader->step();
        textOverlay->step();
        auto frames = sink->pop();
        BOOST_TEST(frames.size() == 1);
        auto outputFrame = frames.cbegin()->second;
        BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
        Test_Utils::saveOrCompare("./data/testOutput/textOverlayrgba.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
    }
}
BOOST_AUTO_TEST_SUITE_END()