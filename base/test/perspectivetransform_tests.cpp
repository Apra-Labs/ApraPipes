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
#include "PerspectiveTransform.h"
#include "PipeLine.h"
#include "StatSink.h"

BOOST_AUTO_TEST_SUITE(perspectivetransform_tests)

BOOST_AUTO_TEST_CASE(mono_1920x960)
{
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.raw")));
    auto metadata = framemetadata_sp(new RawImageMetadata(1920, 960, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
    fileReader->addOutputPin(metadata);

    // Define source and destination points for perspective transformation
    std::vector<cv::Point2f> srcPoints = {
        {0, 0},
        {1920, 0},
        {1920, 960},
        {0, 960}};
    std::vector<cv::Point2f> dstPoints = {
        {100, 200},  // Top-left corner
        {1820, 50},  // Top-right corner
        {1820, 950}, // Bottom-right corner
        {100, 800}   // Bottom-left corner
    };

    auto perspectiveTransform = boost::shared_ptr<PerspectiveTransform>(new PerspectiveTransform(PerspectiveTransformProps(srcPoints, dstPoints)));
    fileReader->setNext(perspectiveTransform);

    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    perspectiveTransform->setNext(sink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(perspectiveTransform->init());
    BOOST_TEST(sink->init());

    fileReader->step();
    perspectiveTransform->step();
    
    auto frames = sink->pop();
    BOOST_TEST(frames.size() == 1);
    
    auto outputFrame = frames.cbegin()->second;
    BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
    
    auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(outputFrame->getMetadata());
    BOOST_TEST(rawMetadata->getDepth() == CV_8U);

    Test_Utils::saveOrCompare("./data/testOutput/perspectivetransform-mono_1920x960_transform.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(bgr_1080x720)
{
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/BGR_1080x720.raw")));
    auto metadata = framemetadata_sp(new RawImageMetadata(1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    fileReader->addOutputPin(metadata);

    std::vector<cv::Point2f> srcPoints = {
        {0, 0},
        {1080, 0},
        {1080, 720},
        {0, 720}};
    std::vector<cv::Point2f> dstPoints = {
        {50, 100},
        {1030, 50},
        {1030, 670},
        {50, 620}};

    auto perspectiveTransform = boost::shared_ptr<PerspectiveTransform>(new PerspectiveTransform(PerspectiveTransformProps(srcPoints, dstPoints)));
    fileReader->setNext(perspectiveTransform);

    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    perspectiveTransform->setNext(sink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(perspectiveTransform->init());
    BOOST_TEST(sink->init());

    fileReader->step();
    perspectiveTransform->step();
    
    auto frames = sink->pop();
    BOOST_TEST(frames.size() == 1);
    
    auto outputFrame = frames.cbegin()->second;
    BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

    auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(outputFrame->getMetadata());
    BOOST_TEST(rawMetadata->getDepth() == CV_8U);

    Test_Utils::saveOrCompare("./data/testOutput/perspectivetransform-bgr_1080x720_transform.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(rgb_320x180)
{
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/RGB_320x180.raw")));
    auto metadata = framemetadata_sp(new RawImageMetadata(320, 180, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    fileReader->addOutputPin(metadata);

    std::vector<cv::Point2f> srcPoints = {
        {0, 0},
        {320, 0},
        {320, 180},
        {0, 180}};
    std::vector<cv::Point2f> dstPoints = {
        {20, 30},
        {300, 10},
        {310, 170},
        {10, 160}};

    auto perspectiveTransform = boost::shared_ptr<PerspectiveTransform>(new PerspectiveTransform(PerspectiveTransformProps(srcPoints, dstPoints)));
    fileReader->setNext(perspectiveTransform);

    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    perspectiveTransform->setNext(sink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(perspectiveTransform->init());
    BOOST_TEST(sink->init());

    fileReader->step();
    perspectiveTransform->step();

    auto frames = sink->pop();
    BOOST_TEST(frames.size() == 1);

    auto outputFrame = frames.cbegin()->second;
    BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

    auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(outputFrame->getMetadata());
    BOOST_TEST(rawMetadata->getDepth() == CV_8U);

    Test_Utils::saveOrCompare("./data/testOutput/perspectivetransform-rgb_320x180_transform.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()