#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <getopt.h> /* getopt_long() */
#include <fcntl.h>  /* low-level i/o */
#include <string.h>
#include <unistd.h>
#include <linux/videodev2.h>
#include "V4L2CameraSourceHelper.h"
 
#define CLEAR(x) memset(&(x), 0, sizeof(x))
 
V4L2CameraSourceHelper::V4L2CameraSourceHelper(string devPath) : m_debug(false), m_fd(-1), m_buffers(NULL), m_nBuffers(0)
{
    m_devPath = devPath;
   
}
 
V4L2CameraSourceHelper::~V4L2CameraSourceHelper()
{
    if (m_buffers)
    {
        deinitCamera();
    }
}
 
bool V4L2CameraSourceHelper::openCamera()
{
    struct stat st;
 
    if (-1 == stat(m_devPath.c_str(), &st))
    {
        LOG_ERROR << "Cannot identify '%s': %d : %s\n"
                  << m_devPath.c_str() << errno << strerror(errno);
        return false;
    }
 
    if (!S_ISCHR(st.st_mode))
    {
        LOG_ERROR << "%s is no device\n"
                  << m_devPath.c_str();
        return false;
    }
 
    m_fd = open(m_devPath.c_str(), O_RDWR /* required */ | O_NONBLOCK, 0);
 
    if (-1 == m_fd)
    {
        LOG_ERROR << "Cannot open '%s': %d :%s\n"
                  << m_devPath.c_str() << errno << strerror(errno);
        return false;
    }
    return true;
}
 
bool V4L2CameraSourceHelper::initCamera()
{
 
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
 
    if (-1 == xioctl(m_fd, VIDIOC_QUERYCAP, &cap))
    {
        if (m_debug)
        {
            if (EINVAL == errno)
            {
                LOG_ERROR << "%s is no V4L2 device\n"
                          << m_devPath.c_str();
            }
            else
            {
                LOG_ERROR << "error in VIDIOC_QUERYCAP\n";
            }
        }
        return false;
    }
 
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE))
    {
        LOG_ERROR << "%s is no video capture device\n"
                  << m_devPath.c_str();
        return false;
    }
 
    /*
     switch (io) {
     case IO_METHOD_READ:
     if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
     fprintf(stderr, "%s does not support read i/o\n", m_devPath.c_str());
     exit(EXIT_FAILURE);
     }
     break;
 
     case IO_METHOD_MMAP:
     case IO_METHOD_USERPTR:
     if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
     fprintf(stderr, "%s does not support streaming i/o\n", m_devPath.c_str());
     exit(EXIT_FAILURE);
     }
     break;
     }
     */
    if (!(cap.capabilities & V4L2_CAP_STREAMING))
    {
        LOG_ERROR << "%s does not support streaming i/o\n"
                  << m_devPath.c_str();
        return false;
    }
 
    CLEAR(cropcap);
 
    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
 
    if (0 == xioctl(m_fd, VIDIOC_CROPCAP, &cropcap))
    {
        crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        crop.c = cropcap.defrect; /* reset to default */
 
        if (-1 == xioctl(m_fd, VIDIOC_S_CROP, &crop))
        {
            switch (errno)
            {
            case EINVAL:
                break;
            default:
                break;
            }
        }
    }
    else
    {
    }
 
    CLEAR(fmt)
 
    return true;
}
 
bool V4L2CameraSourceHelper::requestBuffers(uint16_t bufferCount)
{
    struct v4l2_requestbuffers req;
 
    CLEAR(req);
 
    req.count = bufferCount;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
 
    if (-1 == xioctl(m_fd, VIDIOC_REQBUFS, &req))
    {
        if (EINVAL == errno)
        {
            LOG_ERROR << "%s does not support memory mapping\n"
                      << m_devPath.c_str();
        }
        else
        {
            LOG_ERROR << "error:: VIDIOC_REQBUFS\n";
        }
        return false;
    }
 
    if (req.count < 2)
    {
        LOG_ERROR << "Insufficient buffer memory on %s\n" << m_devPath.c_str();
        return false;
    }
 
    m_buffers = (buffer *)calloc(req.count, sizeof(*m_buffers));
 
    if (!m_buffers)
    {
        LOG_ERROR << "Out of memory\n";
        return false;
    }
 
    for (m_nBuffers = 0; m_nBuffers < req.count; ++m_nBuffers)
    {
        struct v4l2_buffer buf;
 
        CLEAR(buf);
 
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = m_nBuffers;
 
        if (-1 == xioctl(m_fd, VIDIOC_QUERYBUF, &buf))
        {
            LOG_ERROR" << error:: VIDIOC_QUERYBUF\n";
            return false;
        }
 
        m_buffers[m_nBuffers].length = buf.length;
        m_buffers[m_nBuffers].start = mmap(NULL /* start anywhere */,
                                           buf.length,
                                           PROT_READ | PROT_WRITE /* required */,
                                           MAP_SHARED /* recommended */, m_fd, buf.m.offset);
 
        if (MAP_FAILED == m_buffers[m_nBuffers].start)
        {
            LOG_ERROR << "error:: mmap\n";
            return false;
        }
    }
    return true;
}
 
bool V4L2CameraSourceHelper::startStreaming()
{
    for (int i = 0; i < m_nBuffers; ++i)
    {
        struct v4l2_buffer buf;
 
        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
 
        if (-1 == xioctl(m_fd, VIDIOC_QBUF, &buf))
        {
            LOG_ERROR << "error:: VIDIOC_QBUF\n";
            return false;
        }
    }
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(m_fd, VIDIOC_STREAMON, &type))
    {
        LOG_ERROR << "error:: VIDIOC_STREAMON\n";
        return false;
    }
    return true;
}
 
bool V4L2CameraSourceHelper::readFrame(uint8_t *frameBuffer, uint64_t &frameBufferSize)
{
    struct v4l2_buffer buf;
    fd_set fds;
    struct timeval tv;
    int r;
 
    FD_ZERO(&fds);
    FD_SET(m_fd, &fds);
 
    /* Timeout. */
    tv.tv_sec = 20;
    tv.tv_usec = 0;
    bool isError = true;
 
    for (int i = 0; i < 3; i++)
    {
        r = select(m_fd + 1, &fds, NULL, NULL, &tv);
 
        if (-1 == r)
        {
            if (EINTR == errno)
                continue;
            LOG_ERROR <<"error:: select\n";
            return false;
        }
 
        if (0 == r)
        {
            LOG_ERROR <<"error:: select timeout\n";
            return false;
        }
        isError = false;
        break;
    }
    if (isError)
    {
        LOG_ERROR << "error:: EINTR\n";
        return false;
    }
    if (!dequeBuffer(buf))
    {
        return false;
    }
    frameBufferSize = buf.bytesused;
    memcpy(frameBuffer, m_buffers[buf.index].start, buf.bytesused);
    return enqueBuffer(buf);
}
 
bool V4L2CameraSourceHelper::enqueBuffer(v4l2_buffer &buf)
{
    if (-1 == xioctl(m_fd, VIDIOC_QBUF, &buf))
    {
        LOG_ERROR << "error:: VIDIOC_QBUF\n";
        return false;
    }
    return true;
}
bool V4L2CameraSourceHelper::dequeBuffer(v4l2_buffer &buf)
{
    CLEAR(buf);
 
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
 
    if (-1 == xioctl(m_fd, VIDIOC_DQBUF, &buf))
    {
        switch (errno)
        {
        case EAGAIN:
            LOG_ERROR << "error:: VIDIOC_DQBUF: EAGAIN\n" ;
            return 0;
        case EIO:
 
        default:
            LOG_ERROR <<"error:: VIDIOC_DQBUF\n";
            return false;
        }
    }
    return true;
}
 
bool V4L2CameraSourceHelper::stopStreaming()
{
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(m_fd, VIDIOC_STREAMOFF, &type))
    {
        LOG_ERROR << "error:: VIDIOC_STREAMOFF\n";
        return false;
    }
    return true;
}
 
bool V4L2CameraSourceHelper::deinitCamera()
{
    for (uint16_t i = 0; i < m_nBuffers; ++i)
    {
        if (-1 == munmap(m_buffers[i].start, m_buffers[i].length))
        {
            LOG_ERROR << "error:: munmap\n";
            return false;
        }
    }
    m_buffers = NULL;
    m_nBuffers = 0;
    return true;
}
 
bool V4L2CameraSourceHelper::closeCamera()
{
    if (-1 == close(m_fd))
    {
        LOG_ERROR << "error:: close" ;
        return false;
    }
    m_fd = -1;
    return true;
}
 
int V4L2CameraSourceHelper::xioctl(int fh, int request, void *arg)
{
    int r;
    do
    {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);
 
    return r;
}
 
bool V4L2CameraSourceHelper::setFormatBGGR10(uint32_t width, uint32_t height,
                                             v4l2_format &fmt)
{
 
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_SBGGR10;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    fmt.fmt.pix.bytesperline = fmt.fmt.pix.width * 2;
    fmt.fmt.pix.sizeimage = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
 
    if (-1 == xioctl(m_fd, VIDIOC_S_FMT, &fmt))
    {
        LOG_ERROR << "Could not set format error:: VIDIOC_S_FMT\n";
        return false;
    }
 
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_SBGGR10)
    {
        LOG_ERROR << "Libv4l didn't accept RGB24 format. Can't proceed.\n";
        return false;
        ;
    }
    if ((fmt.fmt.pix.width != width) || (fmt.fmt.pix.height != height))
        LOG_ERROR << "Warning: driver is sending image at %dx%d\n" << fmt.fmt.pix.width <<  fmt.fmt.pix.height;
 
    return true;
}
 
bool V4L2CameraSourceHelper::setFormatRGGB10(uint32_t width, uint32_t height,
                                             v4l2_format &fmt)
{
    unsigned int min;
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_SRGGB10;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    fmt.fmt.pix.bytesperline = fmt.fmt.pix.width * 2;
    fmt.fmt.pix.sizeimage = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
 
    if (-1 == xioctl(m_fd, VIDIOC_S_FMT, &fmt))
    {
        LOG_ERROR << "Could not set format error:: VIDIOC_S_FMT\n";
        return false;
    }
 
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_SRGGB10)
    {
        LOG_ERROR "Libv4l didn't accept RGB24 format. Can't proceed.\n";
        return false;
        ;
    }
    if ((fmt.fmt.pix.width != width) || (fmt.fmt.pix.height != height))
        LOG_ERROR << "Warning: driver is sending image at %dx%d\n"<< fmt.fmt.pix.width << fmt.fmt.pix.height;
    return true;
}
 

