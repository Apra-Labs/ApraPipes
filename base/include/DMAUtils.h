#pragma once

#include "EGL/egl.h"
#include "cudaEGL.h"
#include <nvbufsurface.h> 

class Frame;
class DMAUtils {
public:
		static uint8_t* getCudaPtrForFD(int fd, EGLImageKHR eglImage, CUgraphicsResource *pResource, CUeglFrame eglFrame, EGLDisplay eglDisplay);
        static uint8_t* getCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, CUeglFrame *pEglFrame);
        static void freeCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, NvBufSurface *surf, EGLDisplay eglDisplay);
};
