/**
 * @file test_timelapse.cpp
 * @brief Unit tests for timelapse sample (motion-based video summarization)
 */

#include <boost/test/unit_test.hpp>
#include <boost/shared_ptr.hpp>

// ApraPipes modules
#include "PipeLine.h"
#include "Mp4ReaderSource.h"
#include "H264Decoder.h"
#include "MotionDetectorXform.h"
#include "ValveModule.h"
#include "Mp4VideoMetadata.h"
#include "H264Metadata.h"
#include "FrameMetadata.h"

BOOST_AUTO_TEST_SUITE(timelapse_tests)

/**
 * @brief Test MotionDetectorXform module creation
 */
BOOST_AUTO_TEST_CASE(test_motion_detector_creation)
{
    BOOST_CHECK_NO_THROW({
        MotionDetectorXformProps props;
        props.motionThreshold = 0.05f;  // 5% motion threshold

        auto motionDetector = boost::shared_ptr<MotionDetectorXform>(
            new MotionDetectorXform(props)
        );
    });
}

/**
 * @brief Test motion detector with different threshold values
 */
BOOST_AUTO_TEST_CASE(test_motion_threshold_values)
{
    std::vector<float> thresholds = {0.01f, 0.05f, 0.1f, 0.2f, 0.5f};

    for (float threshold : thresholds) {
        BOOST_CHECK_NO_THROW({
            MotionDetectorXformProps props;
            props.motionThreshold = threshold;

            auto motionDetector = boost::shared_ptr<MotionDetectorXform>(
                new MotionDetectorXform(props)
            );
        });
    }
}

/**
 * @brief Test ValveModule for frame filtering
 */
BOOST_AUTO_TEST_CASE(test_valve_for_timelapse)
{
    // ValveModule is used to filter frames based on motion detection
    BOOST_CHECK_NO_THROW({
        ValveModuleProps props;
        props.allowFrames = 0;  // Start with valve closed
        props.direction = ValveModuleProps::Direction::PUSH;

        auto valve = boost::shared_ptr<ValveModule>(
            new ValveModule(props)
        );
    });
}

/**
 * @brief Test timelapse pipeline structure
 *
 * Pipeline: Mp4Reader → H264Decoder → MotionDetector → Valve → Encoder → Writer
 */
BOOST_AUTO_TEST_CASE(test_timelapse_pipeline_structure)
{
    BOOST_CHECK_NO_THROW({
        // Setup Mp4Reader
        bool parseFS = false;
        auto mp4ReaderProps = Mp4ReaderSourceProps("test.mp4", parseFS, 0, false, false, false);
        mp4ReaderProps.fps = 30;

        auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(
            new Mp4ReaderSource(mp4ReaderProps)
        );

        auto h264ImageMetadata = framemetadata_sp(new H264Metadata(1920, 1080));
        mp4Reader->addOutputPin(h264ImageMetadata);

        auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
        mp4Reader->addOutputPin(mp4Metadata);

        // Setup H264Decoder
        auto decoder = boost::shared_ptr<H264Decoder>(
            new H264Decoder(H264DecoderProps())
        );

        // Setup MotionDetector
        MotionDetectorXformProps motionProps;
        motionProps.motionThreshold = 0.05f;
        auto motionDetector = boost::shared_ptr<MotionDetectorXform>(
            new MotionDetectorXform(motionProps)
        );

        // Setup Valve
        ValveModuleProps valveProps;
        valveProps.allowFrames = 0;
        valveProps.direction = ValveModuleProps::Direction::PUSH;
        auto valve = boost::shared_ptr<ValveModule>(
            new ValveModule(valveProps)
        );

        // Connect modules
        auto frameType = FrameMetadata::FrameType::H264_DATA;
        std::vector<std::string> mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

        mp4Reader->setNext(decoder, mImagePin);
        decoder->setNext(motionDetector);
        motionDetector->setNext(valve);
    });
}

/**
 * @brief Test motion detection sensitivity configurations
 */
BOOST_AUTO_TEST_CASE(test_motion_sensitivity_configs)
{
    // Test low sensitivity (more motion required)
    BOOST_CHECK_NO_THROW({
        MotionDetectorXformProps props;
        props.motionThreshold = 0.2f;  // 20% threshold - less sensitive
        auto detector = boost::shared_ptr<MotionDetectorXform>(
            new MotionDetectorXform(props)
        );
    });

    // Test medium sensitivity
    BOOST_CHECK_NO_THROW({
        MotionDetectorXformProps props;
        props.motionThreshold = 0.05f;  // 5% threshold - medium sensitivity
        auto detector = boost::shared_ptr<MotionDetectorXform>(
            new MotionDetectorXform(props)
        );
    });

    // Test high sensitivity (less motion required)
    BOOST_CHECK_NO_THROW({
        MotionDetectorXform Props props;
        props.motionThreshold = 0.01f;  // 1% threshold - very sensitive
        auto detector = boost::shared_ptr<MotionDetectorXform>(
            new MotionDetectorXform(props)
        );
    });
}

/**
 * @brief Test Mp4Reader configuration for timelapse
 */
BOOST_AUTO_TEST_CASE(test_mp4_reader_timelapse_config)
{
    // Timelapse typically processes entire video without looping
    bool parseFS = false;
    auto mp4ReaderProps = Mp4ReaderSourceProps(
        "test.mp4",
        parseFS,
        0,      // start from beginning
        false,  // no loop
        false,  // no rewind
        false   // forward direction
    );
    mp4ReaderProps.fps = 30;

    BOOST_CHECK_NO_THROW({
        auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(
            new Mp4ReaderSource(mp4ReaderProps)
        );
    });
}

/**
 * @brief Test valve direction for timelapse
 */
BOOST_AUTO_TEST_CASE(test_valve_direction)
{
    // Test PUSH direction (valve controls output)
    BOOST_CHECK_NO_THROW({
        ValveModuleProps props;
        props.direction = ValveModuleProps::Direction::PUSH;
        auto valve = boost::shared_ptr<ValveModule>(new ValveModule(props));
    });

    // Test PULL direction (valve controls input)
    BOOST_CHECK_NO_THROW({
        ValveModuleProps props;
        props.direction = ValveModuleProps::Direction::PULL;
        auto valve = boost::shared_ptr<ValveModule>(new ValveModule(props));
    });
}

BOOST_AUTO_TEST_SUITE_END()
