#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "MemTypeConversion.h"
#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "JPEGEncoderL4TM.h"
#include "FileReaderModule.h"
#include "StatSink.h"
#include "PipeLine.h"
#include "test_utils.h"
#include <fstream>
#include "NvV4L2Camera.h"
#include "DMAFDToHostCopy.h"
#include "FileWriterModule.h"
#include "DMAFDWrapper.h"
#include "DMAAllocator.h"
#include "nvbufsurface.h"
#include "DMAUtils.h"
#include "NvArgusCamera.h"
BOOST_AUTO_TEST_SUITE(dmautils_tests)
BOOST_AUTO_TEST_CASE(getCudaPtrForFD_valid_fd)
{
     LoggerProps logProps;
    logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::severity_level::trace);
    int height=1024;
    int width=1024;
    NvBufSurfaceColorFormat format = NVBUF_COLOR_FORMAT_YUV420;
     auto eglDisplay = ApraEGLDisplay::getEGLDisplay();
    auto dmaWrapper = DMAFDWrapper::create(0, width, height, format, NVBUF_LAYOUT_PITCH, eglDisplay);
    int fd = dmaWrapper->getFd();
    BOOST_REQUIRE(fd > 0);
    EGLImageKHR eglImage = EGL_NO_IMAGE_KHR;
    CUgraphicsResource pResource;
    CUeglFrame eglFrame;
    uint8_t* cudaPtr = DMAUtils::getCudaPtrForFD(fd, eglImage, &pResource, eglFrame, eglDisplay);
    BOOST_TEST(cudaPtr != nullptr);
    BOOST_TEST(eglImage != EGL_NO_IMAGE_KHR);
    cudaError_t result = cudaMemset(cudaPtr, 0xFF, 1024);
    BOOST_TEST(result == cudaSuccess);
    NvBufSurface *surf = nullptr;
    BOOST_REQUIRE(NvBufSurfaceFromFd(fd, (void**)&surf) == 0);
    BOOST_CHECK_NO_THROW(DMAUtils::freeCudaPtr(eglImage, &pResource, surf, eglDisplay));
}
BOOST_AUTO_TEST_CASE(getCudaPtrForFD_different_formats)
{
    LoggerProps logProps;
    logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::severity_level::trace);
    struct FormatTest {
        NvBufSurfaceColorFormat format;
        std::string name;
    };
    std::vector<FormatTest> formats = {
        {NVBUF_COLOR_FORMAT_YUV420, "YUV420"},
        {NVBUF_COLOR_FORMAT_RGBA, "RGBA"}
    };
    auto eglDisplay = ApraEGLDisplay::getEGLDisplay();
    for (const auto& fmt : formats) {
        LOG_INFO << "Testing getCudaPtrForFD with format: " << fmt.name;
        auto dmaWrapper = DMAFDWrapper::create(0, 256, 256, fmt.format, NVBUF_LAYOUT_PITCH, eglDisplay);
        if (!dmaWrapper) {
            LOG_WARNING << "Failed to create DMA buffer for format " << fmt.name << " - skipping";
            continue;
        }
        int fd = dmaWrapper->getFd();
        EGLImageKHR eglImage = EGL_NO_IMAGE_KHR;
        CUgraphicsResource pResource;
        CUeglFrame eglFrame;
        uint8_t* cudaPtr = DMAUtils::getCudaPtrForFD(fd, eglImage, &pResource, eglFrame, eglDisplay);
        if (cudaPtr) {
            size_t testSize = 1024;
            cudaError_t result = cudaMemset(cudaPtr, 0x55, testSize);
            BOOST_TEST(result == cudaSuccess);
            if (result != cudaSuccess) {
                LOG_ERROR << "CUDA operations failed for format " << fmt.name;
            }
            NvBufSurface *surf = nullptr;
            if (NvBufSurfaceFromFd(fd, (void**)&surf) == 0) {
                DMAUtils::freeCudaPtr(eglImage, &pResource, surf, eglDisplay);
            }
        } else {
            LOG_WARNING << "Failed to get CUDA pointer for format " << fmt.name;
        }
    }
}
BOOST_AUTO_TEST_SUITE_END()