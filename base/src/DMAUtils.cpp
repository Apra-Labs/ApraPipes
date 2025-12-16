#include "DMAUtils.h"
#include "Logger.h"
#include "CudaDriverLoader.h"

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
    auto& loader = CudaDriverLoader::getInstance();
    if (!loader.isAvailable()) {
        LOG_ERROR << "CUDA driver not available for DMA operations";
        return nullptr;
    }

    auto status = loader.cuGraphicsEGLRegisterImage(pResource, eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsEGLRegisterImage failed: " << status << " cuda process stop";
        return NULL;
    }

    status = loader.cuGraphicsResourceGetMappedEglFrame(&eglFrame, *pResource, 0, 0);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsSubResourceGetMappedArray failed status<" << status << ">";
        return NULL;
    }

    status = loader.cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuCtxSynchronize failed status<" << status << ">";
        return NULL;
    }

    return static_cast<uint8_t *>(eglFrame.frame.pPitch[0]);
}

void DMAUtils::freeCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, EGLDisplay eglDisplay)
{
    auto& loader = CudaDriverLoader::getInstance();
    if (!loader.isAvailable()) {
        LOG_ERROR << "CUDA driver not available for DMA cleanup";
        return;
    }

    auto status = loader.cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuCtxSynchronize failed after cc status<" << status << ">";
        return;
    }

    status = loader.cuGraphicsUnregisterResource(*pResource);
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
