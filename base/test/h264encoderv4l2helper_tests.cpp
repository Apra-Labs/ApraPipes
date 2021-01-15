#include <boost/test/unit_test.hpp>

#include "H264EncoderV4L2Helper.h"
#include "test_utils.h"
#include "Logger.h"

BOOST_AUTO_TEST_SUITE(h264encoderv4l2helper_tests)

BOOST_AUTO_TEST_CASE(yuv420_black)
{
    auto width = 1280;
    auto height = 720;
    auto imageSizeY = width * height;
    auto imageSize = (imageSizeY * 3) >> 1;

    auto data = new uint8_t[imageSize];    
    memset(data, 0, imageSizeY);
    memset(data + imageSizeY, 128, imageSizeY >> 1);

    H264EncoderV4L2Helper helper(width, height, 4*1024*1024, 30, [](frame_sp& frame) -> void {
        LOG_INFO << frame->size();
    } );

    for (auto i = 0; i < 100; i++)
    {
        helper.process(data, imageSize);
    }

    boost::this_thread::sleep_for(boost::chrono::seconds(5));

    helper.processEOS();
    delete[] data;
}

BOOST_AUTO_TEST_CASE(memory_cache_free_test)
{
    frame_sp cacheFrame;

    {
        auto width = 1280;
        auto height = 720;
        auto imageSizeY = width * height;
        auto imageSize = (imageSizeY * 3) >> 1;

        auto data = new uint8_t[imageSize];
        memset(data, 0, imageSizeY);
        memset(data + imageSizeY, 128, imageSizeY >> 1);

        H264EncoderV4L2Helper helper(width, height, 4 * 1024 * 1024, 30, [&](frame_sp &frame) -> void {
            cacheFrame = frame;
        });

        for (auto i = 0; i < 10; i++)
        {
            helper.process(data, imageSize);
        }

        boost::this_thread::sleep_for(boost::chrono::seconds(1));

        helper.processEOS();
        delete[] data;
    }

    LOG_ERROR << cacheFrame->data() << "<>" << cacheFrame->size();
    cacheFrame.reset();
}

BOOST_AUTO_TEST_SUITE_END()
