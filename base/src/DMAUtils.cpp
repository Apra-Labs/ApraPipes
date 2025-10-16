#include "DMAUtils.h"
#include "Logger.h"
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

// Forward declaration of NvBufSurface to avoid header dependencies if possible
//struct _NvBufSurface;
//typedef struct _NvBufSurface NvBufSurface;

// This function is the equivalent of the old getCudaPtr.
// It directly uses the EGLImageKHR, which is created externally now.

uint8_t* DMAUtils::getCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, CUeglFrame *pEglFrame)
{
    // Register the EGL image as a CUDA graphics resource
    auto status = cuGraphicsEGLRegisterImage(pResource, eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsEGLRegisterImage failed: " << status << " cuda process stop";
        return NULL;
    }

    // Get the mapped CUeglFrame from the resource
    status = cuGraphicsResourceGetMappedEglFrame(pEglFrame, *pResource, 0, 0);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsResourceGetMappedEglFrame failed status<" << status << ">";
        return NULL;
    }

    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuCtxSynchronize failed status<" << status << ">";
        return NULL;
    }

    return static_cast<uint8_t *>(pEglFrame->frame.pPitch[0]);
}

// Updated freeCudaPtr function to handle cleanup for JetPack 6
void DMAUtils::freeCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, NvBufSurface *surf, EGLDisplay eglDisplay)
{
    auto status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuCtxSynchronize failed after cc status<" << status << ">";
        // Continue cleanup even if sync fails
    }

    status = cuGraphicsUnregisterResource(*pResource);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsUnregisterResource failed: " << status;
        // Continue cleanup even if unregister fails
    }

    // Unmap the EGLImage
    auto res_unmap_egl = NvBufSurfaceUnMapEglImage(surf, 0);
    if (res_unmap_egl)
    {
        LOG_ERROR << "NvBufSurfaceUnMapEglImage Error: " << res_unmap_egl;
    }

    // Destroy the NvBufSurface
    auto res_destroy = NvBufSurfaceDestroy(surf);
    if (res_destroy)
    {
        LOG_ERROR << "NvBufSurfaceDestroy Error: " << res_destroy;
    }
}