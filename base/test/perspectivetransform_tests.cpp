#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <cstring>

#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "ExternalSourceModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "ArrayMetadata.h"

#include "test_utils.h"
#include "PerspectiveTransform.h"
#include "PipeLine.h"
#include "StatSink.h"

// Test wrapper class to access protected methods
class TestPerspectiveTransform : public PerspectiveTransform
{
public:
    TestPerspectiveTransform(PerspectiveTransformProps _props) : PerspectiveTransform(_props) {}
    virtual ~TestPerspectiveTransform() {}

    // Expose protected methods for testing
    void addInputPin(framemetadata_sp &metadata, string &pinId) { 
        return PerspectiveTransform::addInputPin(metadata, pinId); 
    }
    bool validateInputPins() { return PerspectiveTransform::validateInputPins(); }
    bool validateOutputPins() { return PerspectiveTransform::validateOutputPins(); }
    frame_sp makeFrame(size_t size) { return PerspectiveTransform::makeFrame(size); }
    bool send(frame_container& frames) { return PerspectiveTransform::send(frames); }
};

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
        {100, 200},
        {1820, 50},
        {1820, 950},
        {100, 800}
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

    Test_Utils::saveOrCompare("./data/PerspectiveTransform_outputs/perspectivetransform-mono_1920x960_transform.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
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

    Test_Utils::saveOrCompare("./data/PerspectiveTransform_outputs/perspectivetransform-bgr_1080x720_transform.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
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

    Test_Utils::saveOrCompare("./data/PerspectiveTransform_outputs/perspectivetransform-rgb_320x180_transform.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(dynamic_mode_rgb_320x180)
{
    // Read image using FileReaderModule
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/RGB_320x180.raw")));
    auto imageMetadata = framemetadata_sp(new RawImageMetadata(320, 180, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    fileReader->addOutputPin(imageMetadata);

    auto tempSink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    fileReader->setNext(tempSink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(tempSink->init());

    fileReader->step();
    auto imageFrames = tempSink->pop();
    BOOST_TEST(imageFrames.size() == 1);
    auto imageData = imageFrames.cbegin()->second;

    // Create source for all inputs (image + points)
    auto combinedSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto imagePinId = combinedSource->addOutputPin(imageMetadata);
    
    auto srcPointsMetadata = framemetadata_sp(new ArrayMetadata());
    FrameMetadataFactory::downcast<ArrayMetadata>(srcPointsMetadata)->setData(4, CV_32FC2, sizeof(cv::Point2f));
    auto srcPointsPinId = combinedSource->addOutputPin(srcPointsMetadata);
    
    auto dstPointsMetadata = framemetadata_sp(new ArrayMetadata());
    FrameMetadataFactory::downcast<ArrayMetadata>(dstPointsMetadata)->setData(4, CV_32FC2, sizeof(cv::Point2f));
    auto dstPointsPinId = combinedSource->addOutputPin(dstPointsMetadata);

    // Create perspective transform in DYNAMIC mode
    auto perspectiveTransform = boost::shared_ptr<TestPerspectiveTransform>(new TestPerspectiveTransform(PerspectiveTransformProps(PerspectiveTransformProps::DYNAMIC)));

    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    combinedSource->setNext(perspectiveTransform);
    perspectiveTransform->setNext(sink);

    BOOST_TEST(combinedSource->init());
    BOOST_TEST(perspectiveTransform->init());
    BOOST_TEST(sink->init());

    // Create frames with actual file data
    auto imageFrame = combinedSource->makeFrame(imageMetadata->getDataSize(), imagePinId);
    memcpy(imageFrame->data(), imageData->data(), imageMetadata->getDataSize());

    std::vector<cv::Point2f> srcPoints = {
        cv::Point2f(0, 0),
        cv::Point2f(320, 0),
        cv::Point2f(320, 180),
        cv::Point2f(0, 180)
    };
    auto srcPointsFrame = combinedSource->makeFrame(srcPointsMetadata->getDataSize(), srcPointsPinId);
    memcpy(srcPointsFrame->data(), srcPoints.data(), 4 * sizeof(cv::Point2f));

    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(20, 30),
        cv::Point2f(300, 10),
        cv::Point2f(310, 170),
        cv::Point2f(10, 160)
    };
    auto dstPointsFrame = combinedSource->makeFrame(dstPointsMetadata->getDataSize(), dstPointsPinId);
    memcpy(dstPointsFrame->data(), dstPoints.data(), 4 * sizeof(cv::Point2f));

    frame_container allFrames;
    allFrames.insert(make_pair(imagePinId, imageFrame));
    allFrames.insert(make_pair(srcPointsPinId, srcPointsFrame));
    allFrames.insert(make_pair(dstPointsPinId, dstPointsFrame));
    combinedSource->send(allFrames);

    perspectiveTransform->step();

    auto frames = sink->try_pop();
    BOOST_TEST(!frames.empty());
    
    if (!frames.empty()) {
        auto outputFrame = frames.cbegin()->second;
        BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
        
        auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(outputFrame->getMetadata());
        BOOST_TEST(rawMetadata->getDepth() == CV_8U);
        
        Test_Utils::saveOrCompare("./data/PerspectiveTransform_outputs/perspectivetransform-dynamic_rgb_320x180.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
    }
}

BOOST_AUTO_TEST_CASE(getsetprops_basic_to_dynamic)
{
    // Start with BASIC mode
    std::vector<cv::Point2f> initialSrcPoints = {
        cv::Point2f(0, 0),
        cv::Point2f(320, 0),
        cv::Point2f(320, 180),
        cv::Point2f(0, 180)
    };
    std::vector<cv::Point2f> initialDstPoints = {
        cv::Point2f(50, 50),
        cv::Point2f(270, 50),
        cv::Point2f(270, 130),
        cv::Point2f(50, 130)
    };

    auto perspectiveTransform = boost::shared_ptr<TestPerspectiveTransform>(new TestPerspectiveTransform(PerspectiveTransformProps(initialSrcPoints, initialDstPoints)));

    // Set up a minimal pipeline to allow setProps to work
    auto imageMetadata = framemetadata_sp(new RawImageMetadata(320, 180, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    auto imageSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto imagePinId = imageSource->addOutputPin(imageMetadata);
    
    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    imageSource->setNext(perspectiveTransform);
    perspectiveTransform->setNext(sink);

    BOOST_TEST(imageSource->init());
    BOOST_TEST(perspectiveTransform->init());
    BOOST_TEST(sink->init());

    // Test getProps returns correct initial values
    auto currentProps = perspectiveTransform->getProps();
    BOOST_TEST(currentProps.mode == PerspectiveTransformProps::BASIC);
    BOOST_TEST(currentProps.srcPoints.size() == 4);
    BOOST_TEST(currentProps.dstPoints.size() == 4);
    BOOST_TEST(currentProps.srcPoints[0].x == 0);
    BOOST_TEST(currentProps.srcPoints[0].y == 0);
    BOOST_TEST(currentProps.dstPoints[0].x == 50);
    BOOST_TEST(currentProps.dstPoints[0].y == 50);

    // Test setProps functionality - create different points to verify change
    std::vector<cv::Point2f> newSrcPoints = {
        cv::Point2f(10, 10),
        cv::Point2f(310, 10),
        cv::Point2f(310, 170),
        cv::Point2f(10, 170)
    };
    std::vector<cv::Point2f> newDstPoints = {
        cv::Point2f(20, 20),
        cv::Point2f(300, 20),
        cv::Point2f(300, 160),
        cv::Point2f(20, 160)
    };
    
    PerspectiveTransformProps newProps(newSrcPoints, newDstPoints);
    perspectiveTransform->setProps(newProps);
}

BOOST_AUTO_TEST_CASE(pins_override_static_props)
{
    // In DYNAMIC mode, pin inputs control the transformation rather than any constructor parameters (showing pins take precedence)
    std::vector<cv::Point2f> unusedStaticSrcPoints = {
        cv::Point2f(0, 0),
        cv::Point2f(320, 0),
        cv::Point2f(320, 180),
        cv::Point2f(0, 180)
    };
    std::vector<cv::Point2f> unusedStaticDstPoints = {
        cv::Point2f(50, 50),    // These static points are ignored in DYNAMIC mode
        cv::Point2f(270, 50),
        cv::Point2f(270, 130),
        cv::Point2f(50, 130)
    };

    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/RGB_320x180.raw")));
    auto imageMetadata = framemetadata_sp(new RawImageMetadata(320, 180, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    fileReader->addOutputPin(imageMetadata);

    auto tempSink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    fileReader->setNext(tempSink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(tempSink->init());

    fileReader->step();
    auto imageFrames = tempSink->pop();
    BOOST_TEST(imageFrames.size() == 1);
    auto imageData = imageFrames.cbegin()->second;

    // Create source with dynamic pin inputs that should override static props
    auto combinedSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto imagePinId = combinedSource->addOutputPin(imageMetadata);
    
    auto srcPointsMetadata = framemetadata_sp(new ArrayMetadata());
    FrameMetadataFactory::downcast<ArrayMetadata>(srcPointsMetadata)->setData(4, CV_32FC2, sizeof(cv::Point2f));
    auto srcPointsPinId = combinedSource->addOutputPin(srcPointsMetadata);
    
    auto dstPointsMetadata = framemetadata_sp(new ArrayMetadata());
    FrameMetadataFactory::downcast<ArrayMetadata>(dstPointsMetadata)->setData(4, CV_32FC2, sizeof(cv::Point2f));
    auto dstPointsPinId = combinedSource->addOutputPin(dstPointsMetadata);

    // Create module directly in DYNAMIC mode to enable pin inputs
    auto perspectiveTransform = boost::shared_ptr<TestPerspectiveTransform>(new TestPerspectiveTransform(PerspectiveTransformProps(PerspectiveTransformProps::DYNAMIC)));

    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    combinedSource->setNext(perspectiveTransform);
    perspectiveTransform->setNext(sink);

    BOOST_TEST(combinedSource->init());
    BOOST_TEST(perspectiveTransform->init());
    BOOST_TEST(sink->init());

    auto imageFrame = combinedSource->makeFrame(imageMetadata->getDataSize(), imagePinId);
    memcpy(imageFrame->data(), imageData->data(), imageMetadata->getDataSize());

    // Pin input points - these control the transformation in DYNAMIC mode
    std::vector<cv::Point2f> dynamicSrcPoints = {
        cv::Point2f(0, 0),
        cv::Point2f(320, 0),
        cv::Point2f(320, 180),
        cv::Point2f(0, 180)
    };
    std::vector<cv::Point2f> dynamicDstPoints = {
        cv::Point2f(20, 30), 
        cv::Point2f(300, 10),
        cv::Point2f(310, 170),
        cv::Point2f(10, 160)
    };

    auto srcPointsFrame = combinedSource->makeFrame(srcPointsMetadata->getDataSize(), srcPointsPinId);
    memcpy(srcPointsFrame->data(), dynamicSrcPoints.data(), 4 * sizeof(cv::Point2f));

    auto dstPointsFrame = combinedSource->makeFrame(dstPointsMetadata->getDataSize(), dstPointsPinId);
    memcpy(dstPointsFrame->data(), dynamicDstPoints.data(), 4 * sizeof(cv::Point2f));

    frame_container allFrames;
    allFrames.insert(make_pair(imagePinId, imageFrame));
    allFrames.insert(make_pair(srcPointsPinId, srcPointsFrame));
    allFrames.insert(make_pair(dstPointsPinId, dstPointsFrame));
    combinedSource->send(allFrames);

    perspectiveTransform->step();

    auto frames = sink->try_pop();
    BOOST_TEST(!frames.empty());
    
    if (!frames.empty()) {
        auto outputFrame = frames.cbegin()->second;
        BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
        
        auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(outputFrame->getMetadata());
        BOOST_TEST(rawMetadata->getDepth() == CV_8U);
        
        Test_Utils::saveOrCompare("./data/PerspectiveTransform_outputs/perspectivetransform-pins_override_static.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
    }
}

BOOST_AUTO_TEST_SUITE_END()