#include "V4L2CUYUV420Converter.h"
#include <cstring>
#include "nvbuf_utils.h"
#include "AIPExceptions.h"

V4L2CUYUV420Converter::V4L2CUYUV420Converter(uint32_t srcWidth, uint32_t srcHeight, struct v4l2_format& format): mFormat(format)
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

void V4L2CUYUV420Converter::process(uint8_t* data, size_t size, AV4L2Buffer* buffer)
{
    uint32_t i;
    auto numPlanes =buffer->getNumPlanes();
    for( i = 0; i < numPlanes; i++)
    {    
        buffer->v4l2_buf.m.planes[i].bytesused = mBytesUsedY;
        auto v4l2Data = buffer->planesInfo[i].data;
        auto height = mHeightY;
        auto width = mWidthY;
        auto bytesperline = mFormat.fmt.pix_mp.plane_fmt[i].bytesperline;

        if(i != 0)
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

    for(i = 0; i < numPlanes; i++)
    {
        if( NvBufferMemSyncForDevice (buffer->planesInfo[i].fd, i, (void**)(&buffer->planesInfo[i].data) ) < 0)
        {
            LOG_FATAL << "NvBufferMemSyncForDevice failed<>" << i;
        }
    }
}

V4L2CURGBToYUV420Converter::V4L2CURGBToYUV420Converter(uint32_t srcWidth, uint32_t srcHeight, struct v4l2_format& format): V4L2CUYUV420Converter(srcWidth, srcHeight, format)
{
    initEGLDisplay();
}

V4L2CURGBToYUV420Converter::~V4L2CURGBToYUV420Converter()
{
    termEGLDisplay();
}

void V4L2CURGBToYUV420Converter::initEGLDisplay()
{
    eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (eglDisplay == EGL_NO_DISPLAY)
    {
        throw AIPException(AIP_FATAL, "Could not get EGL display connection");
    }

    /**
     * Initialize egl, egl maps DMA mFD of encoder output plane
     * for CUDA to process (render a black rectangle).
     */
    if (!eglInitialize(eglDisplay, NULL, NULL))
    {
        throw AIPException(AIP_FATAL, "init EGL display failed");
    }
}

void V4L2CURGBToYUV420Converter::termEGLDisplay()
{
     if (!eglTerminate(eglDisplay))
    {
        LOG_ERROR << "ERROR eglTerminate failed";
        return;
    }

    if (!eglReleaseThread())
    {
        LOG_ERROR << "ERROR eglReleaseThread failed";
    }
}