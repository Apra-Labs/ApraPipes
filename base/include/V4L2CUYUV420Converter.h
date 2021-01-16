#pragma once

#include "AV4L2Buffer.h"

#include "EGL/egl.h"
#include "cudaEGL.h"
#include "npp.h"

class V4L2CUYUV420Converter
{
public:
    V4L2CUYUV420Converter(uint32_t srcWidth, uint32_t srcHeight, struct v4l2_format& format);
    virtual ~V4L2CUYUV420Converter();

    // YUV420 data - stride is expected to match width
    virtual void process(uint8_t *data, size_t size, AV4L2Buffer *buffer);

protected:    
    uint32_t mBytesUsedY;
    uint32_t mBytesUsedUV;

private:
    struct v4l2_format mFormat;    
    uint32_t mHeightY;
    uint32_t mHeightUV;

    uint32_t mWidthY;
    uint32_t mWidthUV;
};

class V4L2CURGBToYUV420Converter: public V4L2CUYUV420Converter
{
public:
    V4L2CURGBToYUV420Converter(uint32_t srcWidth, uint32_t srcHeight, uint32_t srcStep, struct v4l2_format& format);
    ~V4L2CURGBToYUV420Converter();
    
    void process(uint8_t *data, size_t size, AV4L2Buffer *buffer);

private:
    void initEGLDisplay();
    void termEGLDisplay();

private:    
    EGLDisplay eglDisplay;
    CUresult status;
    CUeglFrame eglFrames[3];
    CUgraphicsResource pResources[3];
    EGLImageKHR eglImages[3];

    Npp8u *dst[3];    
    NppiSize oSizeROI;
    int dstPitch[3];
    int nsrcStep;
};