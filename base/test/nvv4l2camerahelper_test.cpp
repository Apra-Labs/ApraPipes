#include <boost/test/unit_test.hpp>
#include "FrameFactory.h"
#include "Module.h"
#include "DMAFDWrapper.h"
#include "NvV4L2CameraHelper.h"
#include "test_utils.h"
#include "Logger.h"

#include <memory>
#include <thread>
#include <chrono>

BOOST_AUTO_TEST_SUITE(nvv4l2camerahelper_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
    uint32_t width = 640;
    uint32_t height = 480;

    auto framemetadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::UYVY, CV_8UC3, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
    framefactory_sp framefactory(new FrameFactory(framemetadata, 10));

    auto helper = std::make_shared<NvV4L2CameraHelper>([](frame_sp &frame) -> void {
        if(!frame.get())
        {
            LOG_ERROR << "RECEIVED NULLPTR FRAME";
            return;
        }
        auto ptr = static_cast<DMAFDWrapper *>(frame->data());
        LOG_INFO << "Received frame <>" << ptr->tempFD;
    },
    [&]() -> frame_sp {
        auto frame = framefactory->create(framemetadata->getDataSize(),framefactory);
        if(!frame.get())
        {
            LOG_ERROR << "SENDING NULLPTR FRAME";         
        }
        else
        {
            LOG_INFO << "SENDING VALID FRAME";
        }
        return frame;
        }
    );

    BOOST_TEST(helper->start(width, height, 10, false));

    std::this_thread::sleep_for(std::chrono::seconds(1));

    BOOST_TEST(helper->stop());
    helper.reset();
    LOG_INFO << "FINISHED";
}


BOOST_AUTO_TEST_CASE(cache, *boost::unit_test::disabled())
{
    frame_sp cacheFrame;
    {
        uint32_t width = 640;
        uint32_t height = 480;

        auto framemetadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::UYVY, CV_8UC3, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
        framefactory_sp framefactory(new FrameFactory(framemetadata,10));

        auto helper = std::make_shared<NvV4L2CameraHelper>([&](frame_sp &frame) -> void {
            auto ptr = static_cast<DMAFDWrapper *>(frame->data());
            LOG_INFO << "Received frame <>" << ptr->tempFD;
            cacheFrame = frame;
        },
        [&]() -> frame_sp {return framefactory->create(framemetadata->getDataSize(),framefactory);}
        );

        BOOST_TEST(helper->start(width, height, 10,false));

        std::this_thread::sleep_for(std::chrono::seconds(5));

        BOOST_TEST(helper->stop());
        helper.reset();
        LOG_INFO << "RESET DONE";
    }
    cacheFrame.reset();
    LOG_INFO << "FINISHED";
}


BOOST_AUTO_TEST_SUITE_END()