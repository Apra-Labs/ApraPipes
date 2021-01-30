#pragma once

#include "AV4L2Buffer.h"

#include "EGL/egl.h"
#include "cudaEGL.h"
#include "npp.h"

#include "Frame.h"
#include <deque>
#include <mutex>

class V4L2CUYUV420Converter
{
public:
    V4L2CUYUV420Converter(uint32_t srcWidth, uint32_t srcHeight, struct v4l2_format &format);
    virtual ~V4L2CUYUV420Converter();

    // YUV420 data - stride is expected to match width
    virtual void process(frame_sp& frame, AV4L2Buffer *buffer);

    virtual void releaseFrame() {}

protected:
    uint32_t mBytesUsedY;
    uint32_t mBytesUsedUV;
    struct v4l2_format mFormat;
    uint32_t mHeightY;
    uint32_t mHeightUV;

    uint32_t mWidthY;
    uint32_t mWidthUV;
};

class V4L2CUDMABufYUV420Converter: public V4L2CUYUV420Converter
{
public:
    V4L2CUDMABufYUV420Converter(uint32_t srcWidth, uint32_t srcHeight, struct v4l2_format &format);
    ~V4L2CUDMABufYUV420Converter();

    // YUV420 data - stride is expected to match width
    void process(frame_sp& frame, AV4L2Buffer *buffer);
    void releaseFrame();

private:
   std::deque<frame_sp> mCache;
   std::mutex mCacheMutex;
};

class V4L2CURGBToYUV420Converter : public V4L2CUYUV420Converter
{
public:
    V4L2CURGBToYUV420Converter(uint32_t srcWidth, uint32_t srcHeight, uint32_t srcStep, struct v4l2_format &format);
    ~V4L2CURGBToYUV420Converter();

    void process(frame_sp& frame, AV4L2Buffer *buffer);

private:
    void initEGLDisplay();
    void termEGLDisplay();

private:
    EGLDisplay eglDisplay;
    CUresult status;
    CUeglFrame eglFrame;
    CUgraphicsResource pResource;
    EGLImageKHR eglImage;

    Npp8u *dst[3];
    NppiSize oSizeROI;
    int dstPitch[3];
    int nsrcStep;
};