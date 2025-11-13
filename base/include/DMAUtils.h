#pragma once

#include "EGL/egl.h"
#include "cudaEGL.h"
#include <nvbufsurface.h> // Include the new header for NvBufSurface

class Frame;
class DMAUtils {
public:
    /**
     * @brief Registers an EGLImage as a CUDA resource and maps it to a CUeglFrame.
     * @param eglImage The EGL image handle.
     * @param pResource Pointer to the CUgraphicsResource handle.
     * @param pEglFrame Pointer to the CUeglFrame structure to fill.
     * @return A pointer to the CUDA mapped buffer, or nullptr on failure.
     */
    static uint8_t* getCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, CUeglFrame *pEglFrame);

    /**
     * @brief Unregisters a CUDA resource, unmaps the EGLImage, and destroys the NvBufSurface.
     * @param eglImage The EGL image handle (no longer needed, but kept for legacy reference).
     * @param pResource Pointer to the CUgraphicsResource handle.
     * @param surf Pointer to the NvBufSurface.
     * @param eglDisplay The EGL display handle.
     */
    static void freeCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, NvBufSurface *surf, EGLDisplay eglDisplay);
};
