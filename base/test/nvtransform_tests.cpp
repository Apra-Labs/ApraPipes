#include <boost/test/unit_test.hpp>
#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "EglRenderer.h"
#include "EGL/egl.h"
#include "cudaEGL.h"
#include "NvTransform.h"
#include "FileWriterModule.h"
#include "FileReaderModule.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include "npp.h"
#include "CCKernel.h"
#include "RawImageMetadata.h"
#include "DMAFDWrapper.h"
#include "DMAUtils.h"
#include <fstream>
#include <chrono>
#include "nvbufsurftransform.h"
#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "ApraData.h"
#include "FrameFactory.h"

using sys_clock = std::chrono::system_clock;

class NvTransformTest : public NvTransform {
public:
    using NvTransform::NvTransform;
    using NvTransform::addInputPin;
    using NvTransform::processSOS;
    using NvTransform::processEOS;

    bool processFrame(frame_container &frames) {
        return NvTransform::process(frames);
    }
};

frame_sp makeYUV420Frame(const std::string& path, uint32_t width, uint32_t height)
{
    size_t sizeY = width * height;
    size_t sizeUV = sizeY >> 2;
    size_t size = sizeY + 2 * sizeUV;
    size_t step[4] = { 640, 320, 320, 0 };

    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(
        width, height, ImageMetadata::YUV420, step, CV_8U, FrameMetadata::MemType::DMABUF
    ));

    auto frameFactory = framefactory_sp(new FrameFactory(metadata, 1));
    auto frame = frameFactory->create(size, frameFactory);
    if (!frame || !frame->data())
        throw std::runtime_error("Failed to create frame or DMA buffer");

    auto dma = static_cast<DMAFDWrapper*>(frame->data());
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open YUV file: " + path);

    const int y_w = static_cast<int>(width);
    const int y_h = static_cast<int>(height);
    const int u_w = y_w >> 1;
    const int u_h = y_h >> 1;
    const int v_w = u_w;
    const int v_h = u_h;

    NvBufSurface* surf = dma->getNvBufSurface();
    const size_t y_pitch = surf->surfaceList[0].planeParams.pitch[0];
    const size_t u_pitch = surf->surfaceList[0].planeParams.pitch[1];
    const size_t v_pitch = surf->surfaceList[0].planeParams.pitch[2];

    std::vector<uint8_t> rowY(y_w);
    std::vector<uint8_t> rowU(u_w);
    std::vector<uint8_t> rowV(v_w);

    // Y plane
    uint8_t* dstY = static_cast<uint8_t*>(dma->getHostPtrY());
    for (int r = 0; r < y_h; ++r) {
        file.read(reinterpret_cast<char*>(rowY.data()), y_w);
        if (file.gcount() != y_w) throw std::runtime_error("Failed to read Y row");
        memcpy(dstY + r * y_pitch, rowY.data(), y_w);
    }

    // U plane
    uint8_t* dstU = static_cast<uint8_t*>(dma->getHostPtrU());
    for (int r = 0; r < u_h; ++r) {
        file.read(reinterpret_cast<char*>(rowU.data()), u_w);
        if (file.gcount() != u_w) throw std::runtime_error("Failed to read U row");
        memcpy(dstU + r * u_pitch, rowU.data(), u_w);
    }

    // V plane
    uint8_t* dstV = static_cast<uint8_t*>(dma->getHostPtrV());
    for (int r = 0; r < v_h; ++r) {
        file.read(reinterpret_cast<char*>(rowV.data()), v_w);
        if (file.gcount() != v_w) throw std::runtime_error("Failed to read V row");
        memcpy(dstV + r * v_pitch, rowV.data(), v_w);
    }

    return frame;
}

BOOST_AUTO_TEST_SUITE(nv_transform_tests, *boost::unit_test::disabled())

BOOST_AUTO_TEST_CASE(test)
{
    EGLDisplay eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if(eglDisplay == EGL_NO_DISPLAY) throw AIPException(AIP_FATAL, "eglGetDisplay failed"); 
    if (!eglInitialize(eglDisplay, NULL, NULL)) throw AIPException(AIP_FATAL, "eglInitialize failed"); 

    DMAFDWrapper* dmafdWrapper = DMAFDWrapper::create(0,1024,1024,NVBUF_COLOR_FORMAT_ABGR,NVBUF_LAYOUT_PITCH,eglDisplay);
    auto mapped = dmafdWrapper->getHostPtr();
    memset(mapped,255,1024*1024*4);

    auto rgbSize = 10;
    for(auto i = 0; i < rgbSize; i++)
        std::cout << (int)*(static_cast<uint8_t*>(mapped) + i) << " ";
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(yuv_dma_crop)
{
    constexpr int src_width = 3840;
    constexpr int src_height = 2160;
    constexpr int dst_width = 640;
    constexpr int dst_height = 840;

    auto input_frame = makeYUV420Frame("/home/developer/ApraPipes/data/4k.yuv", src_width, src_height);
    BOOST_REQUIRE(input_frame != nullptr);

    NvTransformProps props(ImageMetadata::YUV420, dst_width, dst_height, 0, 0);
    auto nv_transform = std::make_shared<NvTransformTest>(props);

    std::string inputPinId = "input";
    framemetadata_sp metadata = input_frame->getMetadata();
    nv_transform->addInputPin(metadata, inputPinId);

    std::string outputPinId = "output";
    frame_container frames;
    frames[inputPinId] = input_frame;

    BOOST_REQUIRE(nv_transform->init());
    nv_transform->processSOS(input_frame); 
    nv_transform->processFrame(frames);
    nv_transform->processEOS(outputPinId);
    nv_transform->term();

    frame_sp out_frame;
    for (const auto &kv : frames) {
        if (kv.first != inputPinId) {
            out_frame = kv.second;
            break;
        }
    }

    BOOST_REQUIRE(out_frame != nullptr);
    auto out_dma = static_cast<DMAFDWrapper*>(out_frame->data());
    auto out_md = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(out_frame->getMetadata());

    const int y_w = out_md->getWidth(0);
    const int y_h = out_md->getHeight(0);
    const size_t y_pitch = out_md->getStep(0);
    const int u_w = out_md->getWidth(1);
    const int u_h = out_md->getHeight(1);
    const size_t u_pitch = out_md->getStep(1);
    const int v_w = out_md->getWidth(2);
    const int v_h = out_md->getHeight(2);
    const size_t v_pitch = out_md->getStep(2);

    std::ofstream f_out("/home/developer/ApraPipes/data/4k_cropped.yuv", std::ios::binary);

    // Write Y plane
    const uint8_t* srcY = static_cast<const uint8_t*>(out_dma->getHostPtrY());
    for (int r = 0; r < y_h; ++r)
        f_out.write(reinterpret_cast<const char*>(srcY + r * y_pitch), y_w);

    // Write U plane
    const uint8_t* srcU = static_cast<const uint8_t*>(out_dma->getHostPtrU());
    for (int r = 0; r < u_h; ++r)
        f_out.write(reinterpret_cast<const char*>(srcU + r * u_pitch), u_w);

    // Write V plane
    const uint8_t* srcV = static_cast<const uint8_t*>(out_dma->getHostPtrV());
    for (int r = 0; r < v_h; ++r)
        f_out.write(reinterpret_cast<const char*>(srcV + r * v_pitch), v_w);
}

BOOST_AUTO_TEST_SUITE_END()
