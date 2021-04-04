#include "DMAFDWrapper.h"
#include "DMAUtils.h"
#include "nvbuf_utils.h"
#include "Logger.h"
#include "cuda_runtime.h"

DMAFDWrapper *DMAFDWrapper::create(int index, int width, int height,
                                    NvBufferColorFormat colorFormat,
                                    NvBufferLayout layout, EGLDisplay eglDisplay)
{
    DMAFDWrapper *buffer = new DMAFDWrapper(index, eglDisplay);
    if (!buffer)
    {
        return nullptr;
    }

    NvBufferCreateParams inputParams = {0};

    inputParams.width = width;
    inputParams.height = height;
    inputParams.layout = layout;
    inputParams.colorFormat = colorFormat;
    inputParams.payloadType = NvBufferPayload_SurfArray;
    inputParams.nvbuf_tag = NvBufferTag_CAMERA;

    if (NvBufferCreateEx(&buffer->m_fd, &inputParams))
    {
        LOG_ERROR << "Failed NvBufferCreateEx";
        delete buffer;
        return nullptr;
    }

    // Use NvBufferMemMapEx
    NvBufferMemMap(buffer->m_fd, 0, NvBufferMem_Read_Write, &(buffer->hostPtr));	
    if(colorFormat == NvBufferColorFormat_ARGB32){
        cudaFree(0);
        buffer->eglImage = NvEGLImageFromFd(eglDisplay, buffer->m_fd);
        if(buffer->eglImage == EGL_NO_IMAGE_KHR){
            LOG_ERROR << "Failed to create eglImage";
            delete buffer;
            return nullptr;
        }
        buffer->cudaPtr = DMAUtils::getCudaPtr(buffer->eglImage,&buffer->pResource,buffer->eglFrame, eglDisplay);
    }

    return buffer;
}

DMAFDWrapper::DMAFDWrapper(int _index, EGLDisplay _eglDisplay) : eglImage(EGL_NO_IMAGE_KHR), m_fd(-1), index(_index), eglDisplay(_eglDisplay), hostPtr(nullptr), cudaPtr(nullptr)
{
}

DMAFDWrapper::~DMAFDWrapper()
{
    if (eglImage != EGL_NO_IMAGE_KHR)
    {
        DMAUtils::freeCudaPtr(eglImage,&pResource, eglDisplay);
    }

    if (hostPtr)
    {
        auto res = NvBufferMemMap(m_fd, 0, NvBufferMem_Read_Write, &hostPtr);
        if(res)
        {
            LOG_ERROR << "NvBufferMemMap Error<>" << res;
        }
    }

    if (m_fd >= 0)
    {
        NvBufferDestroy(m_fd);
        m_fd = -1;
    }
}