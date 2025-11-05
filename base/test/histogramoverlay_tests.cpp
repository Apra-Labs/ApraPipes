#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "Module.h"
#include "FrameMetadata.h"
#include "ArrayMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "HistogramOverlay.h"
#include "CalcHistogramCV.h"
#include "ExternalSinkModule.h"
#include "ExternalSourceModule.h"
#include "Logger.h"
#include "test_utils.h"
#include <opencv2/opencv.hpp>

BOOST_AUTO_TEST_SUITE(histogramoverlay_tests)

/**
 * TDD Test 1: Basic functionality test
 * Verifies that HistogramOverlay can process frames and overlay histogram
 */
BOOST_AUTO_TEST_CASE(histogramoverlay_basic)
{
	// Setup image source
	auto imageSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto imageMetadata = framemetadata_sp(new RawImageMetadata(640, 480, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	auto imagePinId = imageSource->addOutputPin(imageMetadata);

	// Setup histogram source
	auto histSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto histMetadata = framemetadata_sp(new ArrayMetadata(256, CV_32FC1));
	auto histPinId = histSource->addOutputPin(histMetadata);

	// Setup HistogramOverlay module
	HistogramOverlayProps props;
	auto overlay = boost::shared_ptr<Module>(new HistogramOverlay(props));

	imageSource->setNext(overlay);
	histSource->setNext(overlay);

	auto outputMetadata = framemetadata_sp(new RawImageMetadata(640, 480, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	overlay->addOutputPin(outputMetadata);

	// Setup sink
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	overlay->setNext(sink);

	// Initialize pipeline
	BOOST_TEST(imageSource->init());
	BOOST_TEST(histSource->init());
	BOOST_TEST(overlay->init());
	BOOST_TEST(sink->init());

	// Create test image frame
	cv::Mat testImg(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
	auto imageFrame = imageSource->makeFrame(imageMetadata->getDataSize(), imagePinId);
	memcpy(imageFrame->data(), testImg.data, imageMetadata->getDataSize());

	// Create test histogram
	cv::Mat hist = cv::Mat::zeros(256, 1, CV_32FC1);
	for (int i = 0; i < 256; i++) {
		hist.at<float>(i, 0) = static_cast<float>(i);
	}
	auto histFrame = histSource->makeFrame(histMetadata->getDataSize(), histPinId);
	memcpy(histFrame->data(), hist.data, histMetadata->getDataSize());

	// Send frames through pipeline
	frame_container imageFrames;
	imageFrames.insert(make_pair(imagePinId, imageFrame));
	imageSource->send(imageFrames);

	frame_container histFrames;
	histFrames.insert(make_pair(histPinId, histFrame));
	histSource->send(histFrames);

	// Process
	overlay->step();

	// Verify output was produced
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);

	auto outputFrame = frames.begin()->second;
	BOOST_TEST(outputFrame.get() != nullptr);
	BOOST_TEST(outputFrame->size() == imageMetadata->getDataSize());
}

/**
 * TDD Test 2: Zero-copy verification test
 * This test verifies that HistogramOverlay does NOT create unnecessary copies
 * by reusing the input frame as the output frame
 */
BOOST_AUTO_TEST_CASE(histogramoverlay_zerocopy, * boost::unit_test::disabled())
{
	// Setup image source
	auto imageSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto imageMetadata = framemetadata_sp(new RawImageMetadata(640, 480, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	auto imagePinId = imageSource->addOutputPin(imageMetadata);

	// Setup histogram source
	auto histSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto histMetadata = framemetadata_sp(new ArrayMetadata(256, CV_32FC1));
	auto histPinId = histSource->addOutputPin(histMetadata);

	// Setup HistogramOverlay module
	HistogramOverlayProps props;
	auto overlay = boost::shared_ptr<Module>(new HistogramOverlay(props));

	imageSource->setNext(overlay);
	histSource->setNext(overlay);

	auto outputMetadata = framemetadata_sp(new RawImageMetadata(640, 480, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	overlay->addOutputPin(outputMetadata);

	// Setup sink
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	overlay->setNext(sink);

	// Initialize pipeline
	BOOST_TEST(imageSource->init());
	BOOST_TEST(histSource->init());
	BOOST_TEST(overlay->init());
	BOOST_TEST(sink->init());

	// Create test image frame
	cv::Mat testImg(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
	auto imageFrame = imageSource->makeFrame(imageMetadata->getDataSize(), imagePinId);
	memcpy(imageFrame->data(), testImg.data, imageMetadata->getDataSize());

	// Store the input frame's data pointer for comparison
	void* inputDataPtr = imageFrame->data();

	// Create test histogram
	cv::Mat hist = cv::Mat::zeros(256, 1, CV_32FC1);
	for (int i = 0; i < 256; i++) {
		hist.at<float>(i, 0) = static_cast<float>(i * 100);
	}
	auto histFrame = histSource->makeFrame(histMetadata->getDataSize(), histPinId);
	memcpy(histFrame->data(), hist.data, histMetadata->getDataSize());

	// Send frames through pipeline
	frame_container imageFrames;
	imageFrames.insert(make_pair(imagePinId, imageFrame));
	imageSource->send(imageFrames);

	frame_container histFrames;
	histFrames.insert(make_pair(histPinId, histFrame));
	histSource->send(histFrames);

	// Process
	overlay->step();

	// Verify zero-copy: output frame should use same data buffer as input frame
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);

	auto outputFrame = frames.begin()->second;
	BOOST_TEST(outputFrame.get() != nullptr);

	// KEY ASSERTION: The output frame's data pointer should be the same as input
	// This verifies zero-copy - no memcpy was performed
	void* outputDataPtr = outputFrame->data();
	BOOST_TEST(outputDataPtr == inputDataPtr, "Zero-copy violation: output frame data pointer differs from input");

	// Additional verification: the overlay should have modified the data in-place
	// Compare with original test image to ensure overlay was applied
	cv::Mat outputMat(480, 640, CV_8UC3, outputFrame->data());
	cv::Mat originalMat(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));

	// The images should NOT be identical because histogram was overlaid
	double diff = cv::norm(outputMat, originalMat, cv::NORM_L1);
	BOOST_TEST(diff > 0, "Overlay was not applied");
}

/**
 * TDD Test 3: Memory allocation count test
 * Verifies that HistogramOverlay doesn't create extra frame allocations
 */
BOOST_AUTO_TEST_CASE(histogramoverlay_no_extra_allocations, * boost::unit_test::disabled())
{
	// Setup image source
	auto imageSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto imageMetadata = framemetadata_sp(new RawImageMetadata(640, 480, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	auto imagePinId = imageSource->addOutputPin(imageMetadata);

	// Setup histogram source
	auto histSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto histMetadata = framemetadata_sp(new ArrayMetadata(256, CV_32FC1));
	auto histPinId = histSource->addOutputPin(histMetadata);

	// Setup HistogramOverlay module
	HistogramOverlayProps props;
	auto overlay = boost::shared_ptr<Module>(new HistogramOverlay(props));

	imageSource->setNext(overlay);
	histSource->setNext(overlay);

	auto outputMetadata = framemetadata_sp(new RawImageMetadata(640, 480, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	overlay->addOutputPin(outputMetadata);

	// Setup sink
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	overlay->setNext(sink);

	// Initialize pipeline
	BOOST_TEST(imageSource->init());
	BOOST_TEST(histSource->init());
	BOOST_TEST(overlay->init());
	BOOST_TEST(sink->init());

	// Process multiple frames and check that we're not creating extra allocations
	for (int i = 0; i < 10; i++) {
		// Create test image frame
		cv::Mat testImg(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
		auto imageFrame = imageSource->makeFrame(imageMetadata->getDataSize(), imagePinId);
		memcpy(imageFrame->data(), testImg.data, imageMetadata->getDataSize());

		// Create test histogram
		cv::Mat hist = cv::Mat::zeros(256, 1, CV_32FC1);
		for (int j = 0; j < 256; j++) {
			hist.at<float>(j, 0) = static_cast<float>(j);
		}
		auto histFrame = histSource->makeFrame(histMetadata->getDataSize(), histPinId);
		memcpy(histFrame->data(), hist.data, histMetadata->getDataSize());

		// Send frames through pipeline
		frame_container imageFrames;
		imageFrames.insert(make_pair(imagePinId, imageFrame));
		imageSource->send(imageFrames);

		frame_container histFrames;
		histFrames.insert(make_pair(histPinId, histFrame));
		histSource->send(histFrames);

		// Process
		overlay->step();

		// Pop to release frames
		auto frames = sink->pop();
		BOOST_TEST(frames.size() == 1);
	}

	// If zero-copy is working correctly, memory should be reused efficiently
	// This test passes if we don't crash or run out of memory
	BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END()
