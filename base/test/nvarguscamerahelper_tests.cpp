#include <boost/test/unit_test.hpp>

#include "Frame.h"
#include "FrameFactory.h"
#include "RawImagePlanarMetadata.h"
#include "NvArgusCameraHelper.h"
#include "DMAFDWrapper.h"
#include "test_utils.h"
#include "Logger.h"

#include <fstream>

BOOST_AUTO_TEST_SUITE(nvarguscamerahelper_tests)

BOOST_AUTO_TEST_CASE(memory_alloc_test, *boost::unit_test::disabled())
{
    uint32_t width = 1280;
    uint32_t height = 720;
    size_t size = width * height * 3 >> 1;

    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::NV12, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
    auto frameFactory = framefactory_sp(new FrameFactory(metadata, 10));
    frameFactory = framefactory_sp(new FrameFactory(metadata, 10));
    auto frame = frameFactory->create(size, frameFactory);
    BOOST_TEST(frame.get() != nullptr);
    if (frame.get())
    {
        BOOST_TEST(frame->size() == size);
    }
}

BOOST_AUTO_TEST_CASE(save_disk_test, *boost::unit_test::disabled())
{
    uint32_t width = 1280;
    uint32_t height = 720;
    size_t size = width * height * 3 >> 1;

    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::NV12, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
    auto frameFactory = framefactory_sp(new FrameFactory(metadata, 10));
    frameFactory = framefactory_sp(new FrameFactory(metadata, 10));

    std::vector<frame_sp> frames;
    for (auto i = 0; i < 100; i++)
    {
        if (i % 10 == 0)
        {
            frames.clear();
        }
        auto frame = frameFactory->create(size, frameFactory);
        if (!frame.get())
        {
            continue;
        }

        std::ofstream file("data/testOutput/hola.bin", std::ios::out | std::ios::binary);
        if (file.is_open())
        {

            auto ptr = static_cast<DMAFDWrapper *>(frame->data());
            file.write((const char *)ptr->getHostPtr(), frame->size());
            LOG_ERROR << i << "<>" << file.bad() << "<>" << file.eof() << "<>" << file.fail();

            file.close();
        }
        frames.push_back(frame);
    }
}

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
    uint32_t width = 1280;
    uint32_t height = 720;
    size_t size = width * height * 3 >> 1;

    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::NV12, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
    auto frameFactory = framefactory_sp(new FrameFactory(metadata, 10));

    std::shared_ptr<NvArgusCameraHelper> helper = NvArgusCameraHelper::create(
        10, [&](frame_sp &frame) -> void {
        auto ptr = static_cast<DMAFDWrapper *>(frame->data());
        LOG_ERROR << "Received frame <>" << ptr->getFd();
        helper->queueFrameToCamera(); }, [&]() -> frame_sp { return frameFactory->create(size, frameFactory); });

    BOOST_TEST(helper->start(width, height, 30));

    boost::this_thread::sleep_for(boost::chrono::seconds(5));

    BOOST_TEST(helper->stop());
    helper.reset();
    LOG_ERROR << "FINISHED";
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
        LOG_ERROR << " Received frame <>" << ptr->getFd();
        helper->queueFrameToCamera(); }, [&]() -> frame_sp { return frameFactory->create(size, frameFactory); });

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
