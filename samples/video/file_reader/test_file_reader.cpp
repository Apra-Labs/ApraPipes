/**
 * @file test_file_reader.cpp
 * @brief Unit tests for file_reader sample (MP4 playback with seeking)
 */

#include <boost/test/unit_test.hpp>
#include <boost/shared_ptr.hpp>

// ApraPipes modules
#include "PipeLine.h"
#include "Mp4ReaderSource.h"
#include "H264Decoder.h"
#include "ColorConversionXForm.h"
#include "ExternalSinkModule.h"
#include "Mp4VideoMetadata.h"
#include "H264Metadata.h"
#include "FrameMetadata.h"

BOOST_AUTO_TEST_SUITE(file_reader_tests)

/**
 * @brief Test Mp4ReaderSource creation with loop playback
 */
BOOST_AUTO_TEST_CASE(test_mp4_reader_with_loop)
{
    bool parseFS = false;
    std::string testFilePath = "test_video.mp4";

    auto mp4ReaderProps = Mp4ReaderSourceProps(
        testFilePath,  // file path
        parseFS,       // parse filesystem
        0,             // start frame
        true,          // read loop = true
        true,          // rewind on loop = true
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
 * @brief Test color conversion for video playback
 */
BOOST_AUTO_TEST_CASE(test_color_conversion_yuv_to_rgb)
{
    BOOST_CHECK_NO_THROW({
        auto colorConv = boost::shared_ptr<ColorConversion>(
            new ColorConversion(
                ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)
            )
        );
    });
}

/**
 * @brief Test pipeline structure for MP4 playback
 */
BOOST_AUTO_TEST_CASE(test_playback_pipeline_structure)
{
    BOOST_CHECK_NO_THROW({
        // Setup Mp4Reader
        bool parseFS = false;
        auto mp4ReaderProps = Mp4ReaderSourceProps("test.mp4", parseFS, 0, true, true, false);
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

        // Setup ColorConversion
        auto colorConv = boost::shared_ptr<ColorConversion>(
            new ColorConversion(
                ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)
            )
        );

        // Setup ExternalSink for testing
        auto sink = boost::shared_ptr<ExternalSinkModule>(
            new ExternalSinkModule()
        );

        // Connect modules
        auto frameType = FrameMetadata::FrameType::H264_DATA;
        std::vector<std::string> mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

        mp4Reader->setNext(decoder, mImagePin);
        decoder->setNext(colorConv);
        colorConv->setNext(sink);
    });
}

/**
 * @brief Test Mp4ReaderSource with different FPS values
 */
BOOST_AUTO_TEST_CASE(test_different_fps_values)
{
    bool parseFS = false;
    std::vector<int> fpsValues = {10, 15, 24, 30, 60};

    for (int fps : fpsValues) {
        BOOST_CHECK_NO_THROW({
            auto mp4ReaderProps = Mp4ReaderSourceProps("test.mp4", parseFS, 0, true, true, false);
            mp4ReaderProps.fps = fps;

            auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(
                new Mp4ReaderSource(mp4ReaderProps)
            );
        });
    }
}

/**
 * @brief Test seek functionality properties
 */
BOOST_AUTO_TEST_CASE(test_seek_properties)
{
    // Test that Mp4ReaderSource can be created with properties that support seeking
    bool parseFS = false;
    auto mp4ReaderProps = Mp4ReaderSourceProps("test.mp4", parseFS, 0, true, true, false);
    mp4ReaderProps.fps = 30;

    BOOST_CHECK_NO_THROW({
        auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(
            new Mp4ReaderSource(mp4ReaderProps)
        );

        auto h264ImageMetadata = framemetadata_sp(new H264Metadata(1920, 1080));
        mp4Reader->addOutputPin(h264ImageMetadata);

        // Verify module was created successfully
        // Actual seeking would require init() and a valid file
    });
}

/**
 * @brief Test playback direction (forward/backward)
 */
BOOST_AUTO_TEST_CASE(test_playback_direction)
{
    bool parseFS = false;

    // Test forward playback
    BOOST_CHECK_NO_THROW({
        auto mp4ReaderProps = Mp4ReaderSourceProps("test.mp4", parseFS, 0, true, true, false);
        mp4ReaderProps.fps = 30;
        auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(
            new Mp4ReaderSource(mp4ReaderProps)
        );
    });

    // Test backward playback
    BOOST_CHECK_NO_THROW({
        auto mp4ReaderProps = Mp4ReaderSourceProps("test.mp4", parseFS, 0, true, true, true);
        mp4ReaderProps.fps = 30;
        auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(
            new Mp4ReaderSource(mp4ReaderProps)
        );
    });
}

BOOST_AUTO_TEST_SUITE_END()
