#include "DMAUtils.h"
#include "Logger.h"

uint8_t* DMAUtils::getCudaPtrForFD(int fd, EGLImageKHR eglImage, CUgraphicsResource *pResource, CUeglFrame eglFrame, EGLDisplay eglDisplay){
    eglImage = NvEGLImageFromFd(eglDisplay, fd);
    if (eglImage == NULL)
    {
        LOG_ERROR << "DID not find eglImage for File Descriptor";
        return nullptr;
    }
    return getCudaPtr(eglImage, pResource, eglFrame, eglDisplay);
}
uint8_t* DMAUtils::getCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, CUeglFrame eglFrame, EGLDisplay eglDisplay)
{
    auto status = cuGraphicsEGLRegisterImage(pResource, eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsEGLRegisterImage failed: " << status << " cuda process stop";
        return NULL;
    }

    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, *pResource, 0, 0);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsSubResourceGetMappedArray failed status<" << status << ">";
        return NULL;
    }

    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuCtxSynchronize failed status<" << status << ">";
        return NULL;
    }

    return static_cast<uint8_t *>(eglFrame.frame.pPitch[0]);
}

void DMAUtils::freeCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, EGLDisplay eglDisplay)
{
    auto status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuCtxSynchronize failed after cc status<" << status << ">";
        return;
    }

    status = cuGraphicsUnregisterResource(*pResource);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsEGLUnRegisterResource failed: " << status;
        return;
    }

    auto res = NvDestroyEGLImage(eglDisplay, eglImage);
    if (res)
    {
        LOG_ERROR << "NvDestroyEGLImage Error<>" << res;
    }
}