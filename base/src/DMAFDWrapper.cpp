#include "DMAFDWrapper.h"
#include "DMAUtils.h"
#include "nvbufsurface.h"
#include "Logger.h"
#include "AIPExceptions.h"

#include "cuda_runtime.h"

DMAFDWrapper *DMAFDWrapper::create(int index, int width, int height,
    NvBufSurfaceColorFormat colorFormat,
    NvBufSurfaceLayout layout, EGLDisplay eglDisplay)
{
    DMAFDWrapper *buffer = new DMAFDWrapper(index, eglDisplay);
    if (!buffer)
    {
        return nullptr;
    }

    NvBufSurfaceAllocateParams inputParams = {0};

    inputParams.params.width = width;
    inputParams.params.height = height;
    inputParams.params.layout = layout;
    inputParams.params.colorFormat = colorFormat;
    inputParams.params.memType = NVBUF_MEM_SURFACE_ARRAY;
    inputParams.memtag = NvBufSurfaceTag_VIDEO_DEC;

    if (NvBufSurfaceAllocate(&buffer->m_surf, 1, &inputParams))
    {
        LOG_ERROR << "Failed NvBufSurfaceAllocate";
        delete buffer;
        return nullptr;
    }

    buffer->m_fd = buffer->m_surf->surfaceList[0].bufferDesc;

    // Use NvBufferMemMapEx
    auto res = NvBufSurfaceMap(buffer->m_surf, 0, 0, NVBUF_MAP_READ_WRITE);
    if (res)
    {
        LOG_ERROR << "NvBufSurfaceMap Error<>" << res;
        delete buffer;
        return nullptr;
    }

    // JP5: Set hostPtr to mapped address for plane 0
    buffer->hostPtr = buffer->m_surf->surfaceList[0].mappedAddr.addr[0];

    if (colorFormat == NVBUF_COLOR_FORMAT_NV12 ||
        colorFormat == NVBUF_COLOR_FORMAT_YUV420)
    {
        res = NvBufSurfaceMap(buffer->m_surf, 0, 1, NVBUF_MAP_READ_WRITE);
        if (res)
        {
            LOG_ERROR << "NvBufSurfaceMap Error<>" << res;
            delete buffer;
            return nullptr;
        }

        // JP5: Set hostPtrU to mapped address for plane 1
        // For NV12, UV plane comes after Y plane in memory
        if (colorFormat == NVBUF_COLOR_FORMAT_NV12) {
            uint32_t yPitch = buffer->m_surf->surfaceList[0].planeParams.pitch[0];
            uint32_t yHeight = buffer->m_surf->surfaceList[0].planeParams.height[0];
            buffer->hostPtrU = (uint8_t*)buffer->hostPtr + (yPitch * yHeight);
            //LOG_ERROR << "[DMAFDWrapper] NV12 UV pointer fix:";
            LOG_ERROR << "  Y ptr = " << buffer->hostPtr << " UV ptr = " << buffer->hostPtrU;
            //LOG_ERROR << "  UV offset = " << (yPitch * yHeight) << " bytes";
        } else {
            buffer->hostPtrU = buffer->m_surf->surfaceList[0].mappedAddr.addr[1];
        }
    }

    if (colorFormat == NVBUF_COLOR_FORMAT_YUV420)
    {
        res = NvBufSurfaceMap(buffer->m_surf, 0, 2, NVBUF_MAP_READ_WRITE);
        if (res)
        {
            LOG_ERROR << "NvBufSurfaceMap Error<>" << res;
            delete buffer;
            return nullptr;
        }

        // JP5: Set hostPtrV to mapped address for plane 2
        buffer->hostPtrV = buffer->m_surf->surfaceList[0].mappedAddr.addr[2];
    }

    // if (colorFormat != NvBufferColorFormat_UYVY)
    {
    //     buffer->eglImage = NvEGLImageFromFd(eglDisplay, buffer->m_fd);
    //     if (buffer->eglImage == EGL_NO_IMAGE_KHR)
    //     {
    //         LOG_ERROR << "Failed to create eglImage";
    //         delete buffer;
    //         return nullptr;
    //     }

    //     cudaFree(0);
    //     buffer->cudaPtr = DMAUtils::getCudaPtr(buffer->eglImage, &buffer->pResource, buffer->eglFrame, eglDisplay);
    }

    return buffer;
}

DMAFDWrapper::DMAFDWrapper(int _index, EGLDisplay _eglDisplay) : eglImage(EGL_NO_IMAGE_KHR),
                                                                 m_fd(-1),
                                                                 m_surf(nullptr),
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
        // cudaFree(0);
        // DMAUtils::freeCudaPtr(eglImage, &pResource, eglDisplay);
    }

    if (hostPtr)
    {
        auto res = NvBufSurfaceUnMap(getNvBufSurface(), 0, 0);
        if (res)
        {
            LOG_ERROR << "NvBufSurfaceUnMap Error<>" << res;
        }
    }

    if (hostPtrU)
    {
        auto res = NvBufSurfaceUnMap(getNvBufSurface(), 0, 1);
        if (res)
        {
            LOG_ERROR << "NvBufSurfaceUnMap Error<>" << res;
        }
    }

    if (hostPtrV)
    {
        auto res = NvBufSurfaceUnMap(getNvBufSurface(), 0, 2);
        if (res)
        {
            LOG_ERROR << "NvBufSurfaceUnMap Error<>" << res;
        }
    }

    if (m_surf)
    {
        NvBufSurfaceDestroy(m_surf);
        m_surf = nullptr;
        m_fd = -1;
    }
}

void *DMAFDWrapper::getHostPtr()
{

    return hostPtr;
}

void *DMAFDWrapper::getHostPtrY()
{
    return getHostPtr();
}

void *DMAFDWrapper::getHostPtrU()
{

    return hostPtrU;
}

void *DMAFDWrapper::getHostPtrV()
{

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
void DMAFDWrapper::refreshHostPointers()
{
    if (m_surf) {
        hostPtr = m_surf->surfaceList[0].mappedAddr.addr[0];
        
        // For NV12, calculate UV pointer correctly
        if (m_surf->surfaceList[0].colorFormat == NVBUF_COLOR_FORMAT_NV12) {
            uint32_t yPitch = m_surf->surfaceList[0].planeParams.pitch[0];
            uint32_t yHeight = m_surf->surfaceList[0].planeParams.height[0];
            hostPtrU = (uint8_t*)hostPtr + (yPitch * yHeight);
        } else {
            if (m_surf->surfaceList[0].planeParams.num_planes > 1) {
                hostPtrU = m_surf->surfaceList[0].mappedAddr.addr[1];
            }
        }
        
        if (m_surf->surfaceList[0].planeParams.num_planes > 2) {
            hostPtrV = m_surf->surfaceList[0].mappedAddr.addr[2];
        }
    }
}
