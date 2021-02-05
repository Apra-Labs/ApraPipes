#include <boost/test/unit_test.hpp>

#include "NvArgusCameraHelper.h"
#include "test_utils.h"
#include "Logger.h"

BOOST_AUTO_TEST_SUITE(nvarguscamerahelper_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
    uint32_t width = 1280;
    uint32_t height = 720;

    auto helper = NvArgusCameraHelper::create([](frame_sp &frame) -> void {
        auto ptr = static_cast<int *>(frame->data());
        LOG_ERROR << "Received frame <>" << *ptr;
    });

    BOOST_TEST(helper->start(width, height, 30));

    boost::this_thread::sleep_for(boost::chrono::seconds(5));

    BOOST_TEST(helper->stop());
    helper.reset();
    LOG_ERROR << "FINISHED";
}


BOOST_AUTO_TEST_CASE(cache, *boost::unit_test::disabled())
{
    frame_sp cacheFrame;
    {
        uint32_t width = 1280;
        uint32_t height = 720;

        auto helper = NvArgusCameraHelper::create([&](frame_sp &frame) -> void {
            auto ptr = static_cast<int *>(frame->data());
            LOG_ERROR << "Received frame <>" << *ptr;
            cacheFrame = frame;
        });

        BOOST_TEST(helper->start(width, height, 30));

        boost::this_thread::sleep_for(boost::chrono::seconds(5));

        BOOST_TEST(helper->stop());
        helper.reset();
        LOG_ERROR << "RESET DONE";
    }
    cacheFrame.reset();
    LOG_ERROR << "FINISHED";
}

BOOST_AUTO_TEST_CASE(invalid_sensor_mode)
{
    uint32_t width = 380;
    uint32_t height = 720;

    auto helper = NvArgusCameraHelper::create([](frame_sp &frame) -> void {
        auto ptr = static_cast<int *>(frame->data());
        LOG_ERROR << " Received frame <>" << *ptr;
    });

    try
    {
        helper->start(width,height,30);
        BOOST_TEST(false, "It should go to the catch block");
    }
    catch(...)
    {
        LOG_ERROR << "Testcase passed";
    }
    helper.reset();
    LOG_ERROR << "FINISHED";
}

BOOST_AUTO_TEST_SUITE_END()
