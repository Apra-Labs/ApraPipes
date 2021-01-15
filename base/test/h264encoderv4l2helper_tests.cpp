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
        LOG_ERROR << frame->size();
    } );

    for (auto i = 0; i < 10; i++)
    {
        helper.process(data, imageSize);
    }

    helper.processEOS();
    delete[] data;
}

BOOST_AUTO_TEST_SUITE_END()
