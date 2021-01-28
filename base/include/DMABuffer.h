#pragma once

#include <Argus/Argus.h>
#include "nvbuf_utils.h"

class DMABuffer
{
public:
    /* Always use this static method to create DMABuffer */
    static DMABuffer *create(const Argus::Size2D<uint32_t> &size,
                             NvBufferColorFormat colorFormat,
                             NvBufferLayout layout, EGLDisplay eglDisplay);

    virtual ~DMABuffer();

    /* Help function to convert Argus Buffer to DMABuffer */
    static DMABuffer *fromArgusBuffer(Argus::Buffer *buffer);

    /* Return DMA buffer handle */
    int getFd() const { return m_fd; }
    EGLImageKHR getEGLImage() const { return eglImage; }

    /* Get and set reference to Argus buffer */
    void setArgusBuffer(Argus::Buffer *buffer) { m_buffer = buffer; }
    Argus::Buffer *getArgusBuffer() const { return m_buffer; }

    int tempFD;

private:
    DMABuffer(const Argus::Size2D<uint32_t> &size);

    Argus::Buffer *m_buffer; /* Reference to Argus::Buffer */
    int m_fd;
    EGLImageKHR eglImage;
};