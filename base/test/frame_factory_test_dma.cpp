#include "FrameFactory.h"
#include "Frame.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "DMAFDWrapper.h"
#include "DMAAllocator.h"
#include "ApraEGLDisplay.h"
#include "Logger.h"

#include <fstream>

#include <boost/test/unit_test.hpp>

// Helper macro to skip DMA tests when EGL display is not available (headless CI)
#define SKIP_IF_NO_EGL_DISPLAY() \
    if (!ApraEGLDisplay::isAvailable()) { \
        LOG_WARNING << "Skipping test - EGL display not available (headless mode)"; \
        return; \
    }

BOOST_AUTO_TEST_SUITE(frame_factory_test_dma)

BOOST_AUTO_TEST_CASE(frame_factory_test_dmabuf)
{
	SKIP_IF_NO_EGL_DISPLAY();
	framemetadata_sp metadata(new RawImageMetadata(640,480,ImageMetadata::RGBA,CV_8UC4,0,CV_8U,FrameMetadata::MemType::DMABUF));
	boost::shared_ptr<FrameFactory> fact(new FrameFactory(metadata));
	auto f1 = fact->create(1228800, fact);//uses 1 chunk size of metadata is 921600
	auto f2 = fact->create(1228800, fact);//uses 1 chunk
	auto f3 = fact->create(1228800, fact);//uses 1 chunks
}

BOOST_AUTO_TEST_CASE(memory_alloc_test)
{
    SKIP_IF_NO_EGL_DISPLAY();
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
    SKIP_IF_NO_EGL_DISPLAY();
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
    SKIP_IF_NO_EGL_DISPLAY();
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
    SKIP_IF_NO_EGL_DISPLAY();
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

BOOST_AUTO_TEST_CASE(setMetadata_rawimage)
{
    SKIP_IF_NO_EGL_DISPLAY();
    LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

    uint32_t width = 1280;
    uint32_t height = 720;
    size_t size = width * height * 4;
    size_t pitch[4] = {0,0,0,0};
    auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::RGBA, CV_8UC4, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF, true));
    DMAAllocator::setMetadata(metadata,1280,720,ImageMetadata::ImageType::RGBA,pitch);
    size_t mPitch[1] = { pitch[0] };
    LOG_INFO << "mPitch: " << mPitch[0];
}

BOOST_AUTO_TEST_CASE(setMetadata_rawplanarimage)
{
    SKIP_IF_NO_EGL_DISPLAY();
    LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

    uint32_t width = 1280;
    uint32_t height = 720;
    size_t size = width * height * 4;
    size_t pitch[4] = {0,0,0,0};
    size_t offset[4] = {0,0,0,0};
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
    DMAAllocator::setMetadata(metadata,1280,720,ImageMetadata::ImageType::YUV420,pitch,offset);
    size_t mPitch[4];
    size_t mOffset[4];
    for (int i = 0; i < 4; i++)
    {
      mOffset[i] = offset[i];
      mPitch[i]  = pitch[i];
    }
    LOG_INFO << "mPitch values: ";
    for (int i = 0; i < 4; i++)
    {
      LOG_INFO << mPitch[i] << " ";
    }
    
    LOG_INFO << "mOffset values: ";
    for (int i = 0; i < 4; i++)
    {
      LOG_INFO << mOffset[i] << " ";
    }
}

BOOST_AUTO_TEST_CASE(memcopy_read_write)
{
    SKIP_IF_NO_EGL_DISPLAY();
    uint32_t width = 1280;
    uint32_t height = 720;
    size_t size = width * height * 4;

    auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::RGBA, CV_8UC4, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF, true));
    auto frameFactory = framefactory_sp(new FrameFactory(metadata, 10));
    auto frame = frameFactory->create(size, frameFactory);
    void *iptr = (static_cast<DMAFDWrapper *>(frame->data()))->getHostPtr();
    memset(iptr,200,size);
    unsigned char *bytePtr = static_cast<unsigned char *>(iptr);
    BOOST_TEST(bytePtr[0] == 200);
}

BOOST_AUTO_TEST_SUITE_END()