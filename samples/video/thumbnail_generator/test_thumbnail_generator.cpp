/**
 * @file test_thumbnail_generator.cpp
 * @brief Unit tests for thumbnail_generator sample
 */

#include <boost/test/unit_test.hpp>
#include <boost/shared_ptr.hpp>

// ApraPipes modules
#include "PipeLine.h"
#include "Mp4ReaderSource.h"
#include "H264Decoder.h"
#include "ValveModule.h"
#include "Mp4VideoMetadata.h"
#include "H264Metadata.h"
#include "FrameMetadata.h"

BOOST_AUTO_TEST_SUITE(thumbnail_generator_tests)

/**
 * @brief Test Mp4ReaderSource module creation
 */
BOOST_AUTO_TEST_CASE(test_mp4_reader_creation)
{
    bool parseFS = false;
    std::string testFilePath = "test_video.mp4";  // Placeholder path

    auto mp4ReaderProps = Mp4ReaderSourceProps(
        testFilePath,  // file path
        parseFS,       // parse filesystem
        0,             // start frame
        false,         // read loop
        false,         // rewind on loop
        false          // direction (forward)
    );
    mp4ReaderProps.fps = 30;

    BOOST_CHECK_NO_THROW({
        auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(
            new Mp4ReaderSource(mp4ReaderProps)
        );
    });
}

/**
 * @brief Test ValveModule creation and properties
 *
 * ValveModule is used to control how many frames pass through
 */
BOOST_AUTO_TEST_CASE(test_valve_module)
{
    BOOST_CHECK_NO_THROW({
        ValveModuleProps props;
        props.allowFrames = 1;  // Allow only 1 frame (for thumbnail)

        auto valve = boost::shared_ptr<ValveModule>(
            new ValveModule(props)
        );
    });

    // Test with different frame counts
    BOOST_CHECK_NO_THROW({
        ValveModuleProps props;
        props.allowFrames = 5;
        auto valve = boost::shared_ptr<ValveModule>(new ValveModule(props));
    });

    BOOST_CHECK_NO_THROW({
        ValveModuleProps props;
        props.allowFrames = 10;
        auto valve = boost::shared_ptr<ValveModule>(new ValveModule(props));
    });
}

/**
 * @brief Test H264Decoder module creation
 */
BOOST_AUTO_TEST_CASE(test_h264_decoder)
{
    BOOST_CHECK_NO_THROW({
        auto decoder = boost::shared_ptr<H264Decoder>(
            new H264Decoder(H264DecoderProps())
        );
    });
}

/**
 * @brief Test pipeline structure for thumbnail generation
 *
 * This tests the module connections without requiring actual video file
 */
BOOST_AUTO_TEST_CASE(test_pipeline_structure)
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

        // Setup Valve (allow only 1 frame)
        ValveModuleProps valveProps;
        valveProps.allowFrames = 1;
        auto valve = boost::shared_ptr<ValveModule>(
            new ValveModule(valveProps)
        );

        // Connect modules
        auto frameType = FrameMetadata::FrameType::H264_DATA;
        std::vector<std::string> mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

        mp4Reader->setNext(decoder, mImagePin);
        decoder->setNext(valve);
    });
}

/**
 * @brief Test valve module frame counting logic
 */
BOOST_AUTO_TEST_CASE(test_valve_frame_control)
{
    // Test that valve module can be configured for different frame counts
    for (int frameCount = 1; frameCount <= 10; frameCount++) {
        BOOST_CHECK_NO_THROW({
            ValveModuleProps props;
            props.allowFrames = frameCount;
            auto valve = boost::shared_ptr<ValveModule>(new ValveModule(props));
        });
    }
}

BOOST_AUTO_TEST_SUITE_END()
