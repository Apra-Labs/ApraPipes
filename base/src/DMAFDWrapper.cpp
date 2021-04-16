#include "DMAFDWrapper.h"
#include "DMAUtils.h"
#include "nvbuf_utils.h"
#include "Logger.h"
#include "AIPExceptions.h"

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

    NvBufferParams fdParams;
    if(NvBufferGetParams(buffer->m_fd, &fdParams))
    {
        delete buffer;
        return nullptr;
    }

    LOG_INFO << "PixelFormat<" << fdParams.pixel_format << "> Planes<" << fdParams.num_planes << "> NvBufferSize<" << fdParams.nv_buffer_size << "> MemSize<" << fdParams.memsize << "> InColorFormat<" << colorFormat << "> Index<" << index << ">";
    for (auto i = 0; i < fdParams.num_planes; i++)
    {
        LOG_INFO << "Width<" << fdParams.width[i] << "> Height<" << fdParams.height[i] << "> Pitch<" << fdParams.pitch[i] << "> Offset<" << fdParams.offset[i] << "> PSize<" << fdParams.psize[i] << "> Layout<" << fdParams.layout[i] << ">";
    }

    // Use NvBufferMemMapEx
    auto res = NvBufferMemMap(buffer->m_fd, 0, NvBufferMem_Read, &(buffer->hostPtr));
    if (res)
    {
        LOG_ERROR << "NvBufferMemMap Error<>" << res;
        delete buffer;
        return nullptr;
    }

    if (colorFormat == NvBufferColorFormat_NV12 ||
        colorFormat == NvBufferColorFormat_YUV420)
    {
        res = NvBufferMemMap(buffer->m_fd, 1, NvBufferMem_Read, &(buffer->hostPtrU));
        if (res)
        {
            LOG_ERROR << "NvBufferMemMap Error<>" << res;
            delete buffer;
            return nullptr;
        }
    }

    if (colorFormat == NvBufferColorFormat_YUV420)
    {
        res = NvBufferMemMap(buffer->m_fd, 2, NvBufferMem_Read, &(buffer->hostPtrV));
        if (res)
        {
            LOG_ERROR << "NvBufferMemMap Error<>" << res;
            delete buffer;
            return nullptr;
        }
    }

    if (colorFormat != NvBufferColorFormat_UYVY)
    {
        buffer->eglImage = NvEGLImageFromFd(eglDisplay, buffer->m_fd);
        if (buffer->eglImage == EGL_NO_IMAGE_KHR)
        {
            LOG_ERROR << "Failed to create eglImage";
            delete buffer;
            return nullptr;
        }

        cudaFree(0);
        buffer->cudaPtr = DMAUtils::getCudaPtr(buffer->eglImage, &buffer->pResource, buffer->eglFrame, eglDisplay);
    }

    return buffer;
}

DMAFDWrapper::DMAFDWrapper(int _index, EGLDisplay _eglDisplay) : eglImage(EGL_NO_IMAGE_KHR),
                                                                 m_fd(-1),
                                                                 index(_index),
                                                                 eglDisplay(_eglDisplay),
                                                                 hostPtr(nullptr),
                                                                 hostPtrU(nullptr),
                                                                 hostPtrV(nullptr),
                                                                 cudaPtr(nullptr)
{
}

DMAFDWrapper::~DMAFDWrapper()
{
    if (eglImage != EGL_NO_IMAGE_KHR)
    {
        cudaFree(0);
        DMAUtils::freeCudaPtr(eglImage, &pResource, eglDisplay);
    }

    if (hostPtr)
    {
        auto res = NvBufferMemUnMap(m_fd, 0, &hostPtr);
        if (res)
        {
            LOG_ERROR << "NvBufferMemUnMap Error<>" << res;
        }
    }

    if (hostPtrU)
    {
        auto res = NvBufferMemUnMap(m_fd, 1, &hostPtrU);
        if (res)
        {
            LOG_ERROR << "NvBufferMemUnMap Error<>" << res;
        }
    }

    if (hostPtrV)
    {
        auto res = NvBufferMemUnMap(m_fd, 2, &hostPtrV);
        if (res)
        {
            LOG_ERROR << "NvBufferMemUnMap Error<>" << res;
        }
    }

    if (m_fd >= 0)
    {
        NvBufferDestroy(m_fd);
        m_fd = -1;
    }
}

void *DMAFDWrapper::getHostPtr()
{
    if (NvBufferMemSyncForCpu(m_fd, 0, &hostPtr))
    {
        throw AIPException(AIP_FATAL, "NvBufferMemSyncForCpu FAILED.");
    }

    return hostPtr;
}

void *DMAFDWrapper::getHostPtrY()
{
    return getHostPtr();
}

void *DMAFDWrapper::getHostPtrU()
{
    if (NvBufferMemSyncForCpu(m_fd, 1, &hostPtrU))
    {
        throw AIPException(AIP_FATAL, "NvBufferMemSyncForCpu FAILED.");
    }

    return hostPtrU;
}

void *DMAFDWrapper::getHostPtrV()
{
    if (NvBufferMemSyncForCpu(m_fd, 2, &hostPtrV))
    {
        throw AIPException(AIP_FATAL, "NvBufferMemSyncForCpu FAILED.");
    }

    return hostPtrV;
}

void *DMAFDWrapper::getHostPtrUV()
{
    return getHostPtrU();
}

void *DMAFDWrapper::getCudaPtr() const
{
    return cudaPtr;
}

int DMAFDWrapper::getIndex() const
{
    return index;
}

const void *DMAFDWrapper::getClientData() const
{
    return clientData;
}

void DMAFDWrapper::setClientData(const void *_clientData)
{
    clientData = _clientData;
}