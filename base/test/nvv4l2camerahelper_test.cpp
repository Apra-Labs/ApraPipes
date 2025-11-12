#include <boost/test/unit_test.hpp>
#include "FrameFactory.h"
#include "Module.h"
#include "DMAFDWrapper.h"
#include "NvV4L2CameraHelper.h"
#include "DMAAllocator.h"
#include "test_utils.h"
#include "Logger.h"

#include <memory>

BOOST_AUTO_TEST_SUITE(nvv4l2camerahelper_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
    	LoggerProps logProps;
    logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::severity_level::trace);
    uint32_t width = 640;
    uint32_t height = 480;

    auto framemetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
    DMAAllocator::setMetadata(framemetadata, width, height, ImageMetadata::ImageType::YUYV);
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

    helper->start(width, height, 10, false);
    boost::this_thread::sleep_for(boost::chrono::seconds(10));

    BOOST_TEST(helper->stop());
    helper.reset();
    LOG_INFO << "FINISHED";
}


BOOST_AUTO_TEST_CASE(cache, *boost::unit_test::disabled())
{
    frame_sp cacheFrame;
    {
        uint32_t width = 3280;
        uint32_t height = 2464;

        auto framemetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
        DMAAllocator::setMetadata(framemetadata, width, height, ImageMetadata::ImageType::UYVY);
        framefactory_sp framefactory(new FrameFactory(framemetadata,10));

        auto helper = std::make_shared<NvV4L2CameraHelper>([&](frame_sp &frame) -> void {
            auto ptr = static_cast<DMAFDWrapper *>(frame->data());
            LOG_INFO << "Received frame <>" << ptr->tempFD;
            cacheFrame = frame;
        },
        [&]() -> frame_sp {return framefactory->create(framemetadata->getDataSize(),framefactory);}
        );

        BOOST_TEST(helper->start(width, height, 10,false));

        boost::this_thread::sleep_for(boost::chrono::seconds(5));

        BOOST_TEST(helper->stop());
        helper.reset();
        LOG_INFO << "RESET DONE";
    }
    cacheFrame.reset();
    LOG_INFO << "FINISHED";
}


BOOST_AUTO_TEST_SUITE_END()