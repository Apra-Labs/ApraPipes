#include "DMAUtils.h"
#include "Logger.h"
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>


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


void DMAUtils::freeCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, NvBufSurface *surf, EGLDisplay eglDisplay)
{
    auto status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuCtxSynchronize failed<" << status << ">";

    }

    status = cuGraphicsUnregisterResource(*pResource);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsUnregisterResource failed: " << status;
    }

    auto res_unmap_egl = NvBufSurfaceUnMapEglImage(surf, 0);
    if (res_unmap_egl)
    {
        LOG_ERROR << "NvBufSurfaceUnMapEglImage Error: " << res_unmap_egl;
    }
    auto res_destroy = NvBufSurfaceDestroy(surf);
    if (res_destroy)
    {
        LOG_ERROR << "NvBufSurfaceDestroy Error: " << res_destroy;
    }
}
uint8_t* DMAUtils::getCudaPtrForFD(int fd, EGLImageKHR &eglImage, CUgraphicsResource *pResource, CUeglFrame eglFrame, EGLDisplay eglDisplay){
    
    NvBufSurface *surf = nullptr;
    if (NvBufSurfaceFromFd(fd, (void**)&surf) != 0) {
        LOG_ERROR << "Failed to create NvBufSurface from file descriptor";
        return nullptr;
    }
    if (NvBufSurfaceMapEglImage(surf, 0) != 0) {
        LOG_ERROR << "NvBufSurfaceMapEglImage Error";
        NvBufSurfaceDestroy(surf);
        return nullptr;
    }
    eglImage = surf->surfaceList[0].mappedAddr.eglImage;
    if (eglImage == EGL_NO_IMAGE_KHR) {
        LOG_ERROR << "Failed to get EGL image from NvBufSurface";
        NvBufSurfaceUnMapEglImage(surf, 0);
        NvBufSurfaceDestroy(surf);
        return nullptr;
    }
    return getCudaPtr(eglImage, pResource, &eglFrame);
}