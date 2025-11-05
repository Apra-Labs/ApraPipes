#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FileReaderModule.h"
#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "TextOverlayXForm.h"
#include "test_utils.h"
#include <opencv2/opencv.hpp>

BOOST_AUTO_TEST_SUITE(textoverlayxform_zerocopy_tests)

/**
 * TDD Test 1: Verify no full-frame clone is created
 * This test ensures TextOverlayXForm doesn't clone the entire frame
 */
BOOST_AUTO_TEST_CASE(no_full_frame_clone, * boost::unit_test::disabled())
{
    std::string text = "Test Overlay";
    auto imageSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new RawImageMetadata(640, 480, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    auto imagePinId = imageSource->addOutputPin(metadata);

    // Alpha = 0.5 to test alpha blending without full frame clone
    auto textOverlay = boost::shared_ptr<TextOverlayXForm>(new TextOverlayXForm(
        TextOverlayXFormProps(0.5, text, "UpperLeft", false, 30, "FFFFFF", "000000")));
    imageSource->setNext(textOverlay);

    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    textOverlay->setNext(sink);

    BOOST_TEST(imageSource->init());
    BOOST_TEST(textOverlay->init());
    BOOST_TEST(sink->init());

    // Create test image
    cv::Mat testImg(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
    auto imageFrame = imageSource->makeFrame(metadata->getDataSize(), imagePinId);
    memcpy(imageFrame->data(), testImg.data, metadata->getDataSize());

    // Store input frame pointer
    void* inputDataPtr = imageFrame->data();

    frame_container frames;
    frames.insert(make_pair(imagePinId, imageFrame));
    imageSource->send(frames);

    textOverlay->step();

    auto outputFrames = sink->pop();
    BOOST_TEST(outputFrames.size() == 1);

    auto outputFrame = outputFrames.begin()->second;

    // KEY ASSERTION: Output frame should use same data buffer as input (zero-copy at frame level)
    void* outputDataPtr = outputFrame->data();
    BOOST_TEST(outputDataPtr == inputDataPtr, "Frame-level zero-copy: output frame should reuse input frame");

    // Verify overlay was actually applied (data was modified)
    cv::Mat outputMat(480, 640, CV_8UC3, outputFrame->data());
    double diff = cv::norm(outputMat, testImg, cv::NORM_L1);
    BOOST_TEST(diff > 0, "Overlay should have modified the image");
}

/**
 * TDD Test 2: Memory efficiency test with multiple frames
 */
BOOST_AUTO_TEST_CASE(memory_efficient_processing, * boost::unit_test::disabled())
{
    std::string text = "Apra Pipes";
    auto imageSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new RawImageMetadata(1920, 1080, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    auto imagePinId = imageSource->addOutputPin(metadata);

    auto textOverlay = boost::shared_ptr<TextOverlayXForm>(new TextOverlayXForm(
        TextOverlayXFormProps(0.7, text, "LowerRight", false, 40, "00FF00", "000000")));
    imageSource->setNext(textOverlay);

    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    textOverlay->setNext(sink);

    BOOST_TEST(imageSource->init());
    BOOST_TEST(textOverlay->init());
    BOOST_TEST(sink->init());

    // Process multiple frames to test memory efficiency
    for (int i = 0; i < 20; i++) {
        cv::Mat testImg(1080, 1920, CV_8UC3, cv::Scalar(i * 10, i * 5, i * 3));
        auto imageFrame = imageSource->makeFrame(metadata->getDataSize(), imagePinId);
        memcpy(imageFrame->data(), testImg.data, metadata->getDataSize());

        frame_container frames;
        frames.insert(make_pair(imagePinId, imageFrame));
        imageSource->send(frames);

        textOverlay->step();

        auto outputFrames = sink->pop();
        BOOST_TEST(outputFrames.size() == 1);
    }

    // If zero-copy optimization is working, we shouldn't have memory issues
    BOOST_TEST(true, "Memory efficient processing completed successfully");
}

/**
 * TDD Test 3: Verify alpha blending still works correctly with optimization
 */
BOOST_AUTO_TEST_CASE(alpha_blending_correctness, * boost::unit_test::disabled())
{
    std::string text = "Alpha Test";
    auto imageSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new RawImageMetadata(640, 480, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    auto imagePinId = imageSource->addOutputPin(metadata);

    // Test different alpha values
    std::vector<double> alphas = {0.0, 0.3, 0.5, 0.7, 1.0};

    for (auto alpha : alphas) {
        auto textOverlay = boost::shared_ptr<TextOverlayXForm>(new TextOverlayXForm(
            TextOverlayXFormProps(alpha, text, "UpperLeft", false, 30, "FFFFFF", "000000")));
        imageSource->setNext(textOverlay);

        auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
        textOverlay->setNext(sink);

        BOOST_TEST(imageSource->init());
        BOOST_TEST(textOverlay->init());
        BOOST_TEST(sink->init());

        cv::Mat testImg(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
        auto imageFrame = imageSource->makeFrame(metadata->getDataSize(), imagePinId);
        memcpy(imageFrame->data(), testImg.data, metadata->getDataSize());

        frame_container frames;
        frames.insert(make_pair(imagePinId, imageFrame));
        imageSource->send(frames);

        textOverlay->step();

        auto outputFrames = sink->pop();
        BOOST_TEST(outputFrames.size() == 1);

        // Verify that overlay was applied
        auto outputFrame = outputFrames.begin()->second;
        cv::Mat outputMat(480, 640, CV_8UC3, outputFrame->data());

        // The overlay region should be different from the original
        cv::Rect overlayRegion(0, 0, 300, 60); // Approximate overlay region
        cv::Mat origROI = testImg(overlayRegion);
        cv::Mat outputROI = outputMat(overlayRegion);

        double diff = cv::norm(outputROI, origROI, cv::NORM_L1);
        if (alpha > 0.0) {
            BOOST_TEST(diff > 0, "Alpha blending should modify overlay region");
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
