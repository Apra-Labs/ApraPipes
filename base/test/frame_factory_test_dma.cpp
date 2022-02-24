#include "FrameFactory.h"
#include "Frame.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "DMAFDWrapper.h"

#include <fstream>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(frame_factory_test_dma)

BOOST_AUTO_TEST_CASE(frame_factory_test_dmabuf)
{
	framemetadata_sp metadata(new RawImageMetadata(640,480,ImageMetadata::RGBA,CV_8UC4,0,CV_8U,FrameMetadata::MemType::DMABUF));
	boost::shared_ptr<FrameFactory> fact(new FrameFactory(metadata));
	auto f1 = fact->create(1228800, fact);//uses 1 chunk size of metadata is 921600
	auto f2 = fact->create(1228800, fact);//uses 1 chunk
	auto f3 = fact->create(1228800, fact);//uses 1 chunks
}

BOOST_AUTO_TEST_CASE(memory_alloc_test)
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

BOOST_AUTO_TEST_CASE(save_yuv420)
{
    uint32_t width = 1280;
    uint32_t height = 720;
    size_t sizeY = width*height;
    size_t sizeUV = sizeY >> 2;
    size_t size = width * height * 3 >> 1;

    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
    auto frameFactory = framefactory_sp(new FrameFactory(metadata, 10));
    frameFactory = framefactory_sp(new FrameFactory(metadata, 10));

    std::vector<frame_sp> frames;
    for (auto i = 0; i < 100; i++)
    {
        if(i%10 == 0)
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
            file.write((const char *)ptr->getHostPtrY(), sizeY);
            LOG_DEBUG << i << "<>" << file.bad() << "<>" << file.eof() << "<>" << file.fail();

            BOOST_TEST(!file.bad());
            BOOST_TEST(!file.eof());
            BOOST_TEST(!file.fail());

            file.write((const char *)ptr->getHostPtrU(), sizeUV);
            LOG_DEBUG << i << "<>" << file.bad() << "<>" << file.eof() << "<>" << file.fail();

            BOOST_TEST(!file.bad());
            BOOST_TEST(!file.eof());
            BOOST_TEST(!file.fail());

            file.write((const char *)ptr->getHostPtrV(), sizeUV);
            LOG_DEBUG << i << "<>" << file.bad() << "<>" << file.eof() << "<>" << file.fail();

            BOOST_TEST(!file.bad());
            BOOST_TEST(!file.eof());
            BOOST_TEST(!file.fail());

            file.close();
        }

        frames.push_back(frame);
    }
}

BOOST_AUTO_TEST_CASE(save_nv12)
{
    uint32_t width = 1280;
    uint32_t height = 720;
    size_t sizeY = width*height;
    size_t sizeUV = sizeY >> 1;
    size_t size = width * height * 3 >> 1;

    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::NV12, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
    auto frameFactory = framefactory_sp(new FrameFactory(metadata, 10));
    frameFactory = framefactory_sp(new FrameFactory(metadata, 10));

    std::vector<frame_sp> frames;
    for (auto i = 0; i < 100; i++)
    {
        if(i%10 == 0)
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
            file.write((const char *)ptr->getHostPtrY(), sizeY);
            LOG_DEBUG << i << "<>" << file.bad() << "<>" << file.eof() << "<>" << file.fail();

            BOOST_TEST(!file.bad());
            BOOST_TEST(!file.eof());
            BOOST_TEST(!file.fail());

            file.write((const char *)ptr->getHostPtrUV(), sizeUV);
            LOG_DEBUG << i << "<>" << file.bad() << "<>" << file.eof() << "<>" << file.fail();

            BOOST_TEST(!file.bad());
            BOOST_TEST(!file.eof());
            BOOST_TEST(!file.fail());

            file.close();
        }

        frames.push_back(frame);
    }
}

BOOST_AUTO_TEST_CASE(save_rgba)
{
    uint32_t width = 1280;
    uint32_t height = 720;
    size_t size = width * height * 4;

    auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::RGBA, CV_8UC4, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF, true));
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
            file.write((const char *)ptr->getHostPtr(), size);
            LOG_DEBUG << i << "<>" << file.bad() << "<>" << file.eof() << "<>" << file.fail();

            BOOST_TEST(!file.bad());
            BOOST_TEST(!file.eof());
            BOOST_TEST(!file.fail());

            file.close();
        }
        frames.push_back(frame);
    }
}

BOOST_AUTO_TEST_SUITE_END()