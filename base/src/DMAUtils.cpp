#include "DMAUtils.h"
#include "Logger.h"

uint8_t *DMAUtils::getCudaPtrForFD(int fd, EGLImageKHR eglImage, CUgraphicsResource *pResource, CUeglFrame eglFrame, EGLDisplay eglDisplay)
{
    eglImage = NvEGLImageFromFd(eglDisplay, fd);
    if (eglImage == NULL)
    {
        LOG_ERROR << "DID not find eglImage for File Descriptor";
        return nullptr;
    }
    return getCudaPtr(eglImage, pResource, eglFrame, eglDisplay);
}

bool DMAUtils::getCudaPtrForAllChannels(EGLImageKHR eglImage, CUgraphicsResource *pResource, CUeglFrame eglFrame, EGLDisplay eglDisplay, uint8_t **aPtr)
{
    // aPtr[4] ={0};
    aPtr[0] = NULL;
    auto status = cuGraphicsEGLRegisterImage(pResource, eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsEGLRegisterImage failed: " << status << " cuda process stop";
        return false;
    }

    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, *pResource, 0, 0);
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuGraphicsSubResourceGetMappedArray failed status<" << status << ">";
        return false;
    }

    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        LOG_ERROR << "cuCtxSynchronize failed status<" << status << ">";
        return false;
    }

    // LOG_ERROR << "Width " << eglFrame.width;
    // LOG_ERROR << "Height " << eglFrame.height;
    // LOG_ERROR << "Depth " << eglFrame.depth;
    // LOG_ERROR << "Pitch " << eglFrame.pitch;
    // LOG_ERROR << "PlaneCount " << eglFrame.planeCount;
    // LOG_ERROR << "NumChannels " << eglFrame.numChannels;
    //cout << "Pitch " << eglFrame.frame.pitch;
    for (int i = 0; i < eglFrame.planeCount; i++)
    {
        
        // LOG_ERROR << "Buffer of frame " << i << " is " << eglFrame.frame.pPitch[i];
        // printf("Buffer %d -> %8x \n", i, eglFrame.frame.pPitch[i]);
        aPtr[i] = static_cast<uint8_t *>(eglFrame.frame.pPitch[i]);
    }
    return true;
}

uint8_t *DMAUtils::getCudaPtr(EGLImageKHR eglImage, CUgraphicsResource *pResource, CUeglFrame eglFrame, EGLDisplay eglDisplay)
{
    uint8_t *ret[4] = {0};
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

    // LOG_ERROR << "Width " << eglFrame.width;
    // LOG_ERROR << "Height " << eglFrame.height;
    // LOG_ERROR << "Depth " << eglFrame.depth;
    // LOG_ERROR << "Pitch " << eglFrame.pitch;
    // LOG_ERROR << "PlaneCount " << eglFrame.planeCount;
    // LOG_ERROR << "NumChannels " << eglFrame.numChannels;
    //cout << "Pitch " << eglFrame.frame.pitch;
    for (int i = 0; i < eglFrame.planeCount; i++)
    {
        // printf("Buffer %d -> %8x \n", i, eglFrame.frame.pPitch[i]);
        ret[i] = static_cast<uint8_t *>(eglFrame.frame.pPitch[i]);
    }
    return ret[0];
    //return static_cast<uint8_t *>(eglFrame.frame.pPitch[0]);
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