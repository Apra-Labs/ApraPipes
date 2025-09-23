#pragma once
#include "ApraEGLDisplay.h" // this is added to address the following issue: https://github.com/opencv/opencv/issues/7113
#include "nvbufsurface.h"
#include "EGL/egl.h"
#include "cudaEGL.h"

struct NvBufSurface;

class DMAFDWrapper
{
public:
    /* Always use this static method to create DMAFDWrapper */
    static DMAFDWrapper *create(int index, int width, int height,
        NvBufSurfaceColorFormat  colorFormat,
        NvBufSurfaceLayout  layout, EGLDisplay eglDisplay);

    virtual ~DMAFDWrapper();

    /* Return DMA buffer handle */
    int getFd() const { return m_fd; }
    NvBufSurface* getNvBufSurface() const { return m_surf; }
    EGLImageKHR getEGLImage() const { return eglImage; }
    EGLDisplay getEGLDisplay() const { return eglDisplay; }
    void* getHostPtr();
    void* getHostPtrY();
    void* getHostPtrU();
    void* getHostPtrV();
    void* getHostPtrUV();
    void* getCudaPtr() const;
    void refreshHostPointers();
    
    int getIndex() const;

    const void* getClientData() const;
    void setClientData(const void* clientData);


public:
    int tempFD;

private:
    DMAFDWrapper(int index, EGLDisplay EGLDisplay);

private:
    int m_fd;
    NvBufSurface* m_surf;
    EGLImageKHR eglImage;
    CUgraphicsResource  pResource;
    CUeglFrame eglFrame;
    EGLDisplay eglDisplay;

    void* hostPtr; // Y, InterleavedUYVY, RGBA
    void* hostPtrU; // U and UV (NV12)
    void* hostPtrV;

    uint8_t* cudaPtr;
    const int index;

    const void* clientData;
};
