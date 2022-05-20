#pragma once

#include "nvbuf_utils.h"
#include "EGL/egl.h"
#include "cudaEGL.h"


class Frame;
class DMAUtils {
public:
		static uint8_t* getCudaPtrForFD(int fd, EGLImageKHR eglImage, CUgraphicsResource *pResource, CUeglFrame eglFrame, EGLDisplay eglDisplay);
        static uint8_t* getCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, CUeglFrame eglFrame, EGLDisplay eglDisplay);
        static void freeCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, EGLDisplay eglDisplay);
        static bool getCudaPtrForAllChannels(EGLImageKHR eglImage, CUgraphicsResource *pResource, CUeglFrame eglFrame, EGLDisplay eglDisplay, uint8_t** aPtr);
};