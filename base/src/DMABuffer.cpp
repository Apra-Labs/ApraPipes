#include "DMABuffer.h"
#include "Logger.h"

DMABuffer *DMABuffer::create(const Argus::Size2D<uint32_t> &size,
                                    NvBufferColorFormat colorFormat,
                                    NvBufferLayout layout, EGLDisplay eglDisplay)
{
    DMABuffer *buffer = new DMABuffer(size);
    if (!buffer)
    {
        return nullptr;
    }

    NvBufferCreateParams inputParams = {0};

    inputParams.width = size.width();
    inputParams.height = size.height();
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

    buffer->eglImage = NvEGLImageFromFd(eglDisplay, buffer->m_fd);
    if (buffer->eglImage == EGL_NO_IMAGE_KHR)
    {
        LOG_ERROR << "Failed to create EGLImage";
        delete buffer;
        return nullptr;
    }

    return buffer;
}

DMABuffer::DMABuffer(const Argus::Size2D<uint32_t> &size) : m_buffer(nullptr), eglImage(EGL_NO_IMAGE_KHR), m_fd(-1)
{
}

DMABuffer::~DMABuffer()
{
    if (eglImage != EGL_NO_IMAGE_KHR)
    {
        auto res = NvDestroyEGLImage(NULL, eglImage);
        if(res)
        {
            LOG_ERROR << "NvDestroyEGLImage Error<>" << res;
        }
    }

    if (m_fd >= 0)
    {
        NvBufferDestroy(m_fd);
        m_fd = -1;
    }
}

/* Help function to convert Argus Buffer to DMABuffer */
DMABuffer *DMABuffer::fromArgusBuffer(Argus::Buffer *buffer)
{
    Argus::IBuffer *iBuffer = Argus::interface_cast<Argus::IBuffer>(buffer);
    const DMABuffer *dmabuf = static_cast<const DMABuffer *>(iBuffer->getClientData());

    return const_cast<DMABuffer *>(dmabuf);
}