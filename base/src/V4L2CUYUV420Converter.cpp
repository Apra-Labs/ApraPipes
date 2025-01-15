#include "V4L2CUYUV420Converter.h"
#include "DMAFDWrapper.h"
#include "ApraEGLDisplay.h"
#include "Frame.h"
#include "AIPExceptions.h"

#ifdef JP512_TBD
    #include "nvbuf_utils.h"
#endif
#include <cstring>

V4L2CUYUV420Converter::V4L2CUYUV420Converter(uint32_t srcWidth, uint32_t srcHeight, struct v4l2_format &format) : mFormat(format)
{
    mWidthY = srcWidth;
    mWidthUV = mWidthY >> 1;

    mHeightY = srcHeight;
    mHeightUV = mHeightY >> 1;

    mBytesUsedY = mWidthY * mHeightY;
    mBytesUsedUV = mBytesUsedY >> 2;
}

V4L2CUYUV420Converter::~V4L2CUYUV420Converter()
{
}

void V4L2CUYUV420Converter::process(frame_sp& frame, AV4L2Buffer *buffer)
{
    auto data = static_cast<uint8_t*>(frame->data());

    uint32_t i;
    auto numPlanes = buffer->getNumPlanes();
    for (i = 0; i < numPlanes; i++)
    {
        buffer->v4l2_buf.m.planes[i].bytesused = mBytesUsedY;
        auto v4l2Data = buffer->planesInfo[i].data;
        auto height = mHeightY;
        auto width = mWidthY;
        auto bytesperline = mFormat.fmt.pix_mp.plane_fmt[i].bytesperline;

        if (i != 0)
        {
            height = mHeightUV;
            width = mWidthUV;
            buffer->v4l2_buf.m.planes[i].bytesused = mBytesUsedUV;
        }
        for (uint32_t j = 0; j < height; j++)
        {
            memcpy(v4l2Data, data, width);
            data = data + width;
            v4l2Data = v4l2Data + bytesperline;
        }
    }

    for (i = 0; i < numPlanes; i++)
    {

#ifdef JP512_TBD
        if (NvBufferMemSyncForDevice(buffer->planesInfo[i].fd, i, (void **)(&buffer->planesInfo[i].data)) < 0)
#endif        
        {
            LOG_FATAL << "NvBufferMemSyncForDevice failed<>" << i;
        }
    }
}

V4L2CUDMABufYUV420Converter::V4L2CUDMABufYUV420Converter(uint32_t srcWidth, uint32_t srcHeight, struct v4l2_format &format): V4L2CUYUV420Converter(srcWidth, srcHeight, format)
{

}

V4L2CUDMABufYUV420Converter::~V4L2CUDMABufYUV420Converter()
{
    mCache.clear();
}

void V4L2CUDMABufYUV420Converter::process(frame_sp& frame, AV4L2Buffer *buffer)
{
    auto ptr = static_cast<DMAFDWrapper *>(frame->data());    
    buffer->v4l2_buf.m.planes[0].m.fd = ptr->getFd();
    buffer->v4l2_buf.m.planes[0].bytesused = 1;

    std::lock_guard<std::mutex> lock(mCacheMutex);
    mCache.push_back(frame);
}

void V4L2CUDMABufYUV420Converter::releaseFrame()
{
    std::lock_guard<std::mutex> lock(mCacheMutex);
    mCache.pop_front();
}

V4L2CURGBToYUV420Converter::V4L2CURGBToYUV420Converter(uint32_t srcWidth, uint32_t srcHeight, uint32_t srcStep, struct v4l2_format &format) : V4L2CUYUV420Converter(srcWidth, srcHeight, format)
{
    eglDisplay = ApraEGLDisplay::getEGLDisplay();
    cudaFree(0);
    oSizeROI = {static_cast<int>(srcWidth), static_cast<int>(srcHeight)};
    nsrcStep = static_cast<int>(srcStep);

    for (auto i = 0; i < 3; i++)
    {
        dstPitch[i] = static_cast<int>(mFormat.fmt.pix_mp.plane_fmt[i].bytesperline);
    }
}

V4L2CURGBToYUV420Converter::~V4L2CURGBToYUV420Converter()
{
    
}

void V4L2CURGBToYUV420Converter::process(frame_sp& frame, AV4L2Buffer *buffer)
{
#ifdef JP512_TBD    
    eglImage = NvEGLImageFromFd(eglDisplay, buffer->planesInfo[0].fd);
    status = cuGraphicsEGLRegisterImage(&pResource, eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsEGLRegisterImage failed: " << status << " cuda process stop";
        return;
    }

    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsSubResourceGetMappedArray failed status<" << status << ">";
        return;
    }

    for (auto i = 0; i < 3; i++)
    {
        dst[i] = static_cast<Npp8u *>(eglFrame.frame.pPitch[i]);
    }

    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuCtxSynchronize failed status<" << status << ">";
        return;
    }

    auto data = static_cast<uint8_t*>(frame->data());
    auto res = nppiRGBToYUV420_8u_C3P3R(static_cast<const Npp8u *>(data), nsrcStep, dst, dstPitch, oSizeROI);
    if (res != NPP_SUCCESS)
    {
        LOG_ERROR << "nppiRGBToYUV420_8u_C3P3R failed";
    }

    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuCtxSynchronize failed after cc status<" << status << ">";
    }

    status = cuGraphicsUnregisterResource(pResource);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsEGLUnRegisterResource failed: " << status;
    }

    NvDestroyEGLImage(eglDisplay, eglImage);

    for (auto i = 0; i < 3; i++)
    {
        buffer->v4l2_buf.m.planes[i].bytesused = mBytesUsedY;
        if (i != 0)
        {
            buffer->v4l2_buf.m.planes[i].bytesused = mBytesUsedUV;
        }
    }
#endif    
}