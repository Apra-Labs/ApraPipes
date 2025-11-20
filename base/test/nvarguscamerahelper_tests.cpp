#include <boost/test/unit_test.hpp>
#include <memory>
#include <thread>
#include <chrono>

#include "Frame.h"
#include "FrameFactory.h"
#include "RawImagePlanarMetadata.h"
#include "RawImageMetadata.h"
#include "NvArgusCameraHelper.h"
#include "DMAFDWrapper.h"
#include "test_utils.h"
#include "Logger.h"

BOOST_AUTO_TEST_SUITE(nvarguscamerahelper_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
    auto argValue = Test_Utils::getArgValue("n", "1");
    auto n_cams = atoi(argValue.c_str());

    uint32_t width = 800;
    uint32_t height = 800;
    size_t size = width * height * 3 >> 1;

    std::shared_ptr<NvArgusCameraHelper> helper[3];
    int i = 0;
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::NV12, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
    auto frameFactory = framefactory_sp(new FrameFactory(metadata, 10));

    helper[i] = NvArgusCameraHelper::create(
        10, [&](frame_sp &frame) -> void
        {
            auto ptr = static_cast<DMAFDWrapper *>(frame->data());
            helper[i]->queueFrameToCamera();
        },
        [&]() -> frame_sp
        { return frameFactory->create(size, frameFactory); });

    BOOST_TEST(helper[i]->start(width, height, 60, i));

    if (n_cams >= 2)
    {
        i = 1;
        auto metadata1 = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::NV12, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
        auto frameFactory1 = framefactory_sp(new FrameFactory(metadata1, 10));

        helper[i] = NvArgusCameraHelper::create(
            10, [&](frame_sp &frame) -> void
            {
                auto ptr = static_cast<DMAFDWrapper *>(frame->data());
                helper[i]->queueFrameToCamera();
            },
            [&]() -> frame_sp
            { return frameFactory1->create(size, frameFactory1); });
    }
    if (n_cams >= 3)
    {
        i = 2;
        auto metadata2 = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::NV12, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
        auto frameFactory2 = framefactory_sp(new FrameFactory(metadata2, 10));

        helper[i] = NvArgusCameraHelper::create(
            10, [&](frame_sp &frame) -> void
            {
                auto ptr = static_cast<DMAFDWrapper *>(frame->data());
                helper[i]->queueFrameToCamera();
            },
            [&]() -> frame_sp
            { return frameFactory2->create(size, frameFactory2); });
    }

    std::this_thread::sleep_for(std::chrono::seconds(50));

    for (int i = 0; i < n_cams; i++)
    {
        BOOST_TEST(helper[i]->stop());
        helper[i].reset();
    }
}

BOOST_AUTO_TEST_CASE(invalid_sensor_mode, *boost::unit_test::disabled())
{
    uint32_t width = 380;
    uint32_t height = 720;
    size_t size = width * height * 3 >> 1;

    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::NV12, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
    auto frameFactory = framefactory_sp(new FrameFactory(metadata, 10));

    std::shared_ptr<NvArgusCameraHelper> helper = NvArgusCameraHelper::create(
        10, [&](frame_sp &frame) -> void {
        auto ptr = static_cast<DMAFDWrapper *>(frame->data());
        LOG_DEBUG << " Received frame <>" << ptr->getFd();
        helper->queueFrameToCamera(); }, [&]() -> frame_sp { return frameFactory->create(size, frameFactory); });

    try
    {
        helper->start(width, height, 30, 0);
        BOOST_TEST(false, "It should go to the catch block");
    }
    catch(...)
    {
        LOG_INFO << "Testcase passed";
    }
    helper.reset();
    LOG_INFO << "FINISHED";
}

BOOST_AUTO_TEST_SUITE_END()
