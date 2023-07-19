#include "NvV4L2CameraHelper.h"
#include "DMAFDWrapper.h"
#include "NvEglRenderer.h"
#include "NvUtils.h"
#include "nvbuf_utils.h"
#include "Logger.h"

NvV4L2CameraHelper::NvV4L2CameraHelper(SendFrame sendFrame,std::function<frame_sp()> _makeFrame)
{
    // hardcoded device name and pixfmt which is fine for now 
    mCamDevname = "/dev/video0";
    mCamFD = -1;
    mCamPixFmt = V4L2_PIX_FMT_YUYV;

    mRunning = false;
    mSendFrame = sendFrame;
    mMakeFrame = _makeFrame;
}

NvV4L2CameraHelper::~NvV4L2CameraHelper()
{
    if (mCamFD > 0){
        close(mCamFD);
    }
}

bool NvV4L2CameraHelper::cameraInitialize(bool isMirror)
{
    struct v4l2_format fmt;

    /* Open camera device */
    mCamFD = open(mCamDevname, O_RDWR);
    if (mCamFD == -1)
    {
        LOG_ERROR << "Failed to open camera /dev/video0";
        return false;
    }

    // struct v4l2_control inp;
    // memset(&inp, 0, sizeof(inp));
    // inp.id = V4L2_CID_HFLIP;
    // inp.value = !isMirror;

    // if(ioctl(mCamFD, VIDIOC_S_CTRL, &inp) < 0){
    //     LOG_ERROR << "Flip failed";
    //     return false;
    // }

    /* Set camera output format */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = mCamWidth;
    fmt.fmt.pix.height = mCamHeight;
    fmt.fmt.pix.pixelformat = mCamPixFmt;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(mCamFD, VIDIOC_S_FMT, &fmt) < 0)
    {
        LOG_ERROR << "Failed to set camera ouput format to UYVY";
        return false;
    }

    /* Get the real format in case the desired is not supported */
    memset(&fmt, 0, sizeof fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(mCamFD, VIDIOC_G_FMT, &fmt) < 0)
    {
        LOG_ERROR << "Failed to get camera output format";
        return false;
    }

    if (fmt.fmt.pix.width != mCamWidth ||
        fmt.fmt.pix.height != mCamHeight ||
        fmt.fmt.pix.pixelformat != mCamPixFmt)
    {
        LOG_ERROR << "The desired format is not supported";
        LOG_ERROR << "Supported width is : " << fmt.fmt.pix.width;
        LOG_ERROR << "Supported height is : " << fmt.fmt.pix.height;
        LOG_ERROR << "Supported pixelformat is : " << fmt.fmt.pix.pixelformat;

        return false;
    }

    return true;
}

bool NvV4L2CameraHelper::startStream()
{
    /* Start v4l2 streaming */
    auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(mCamFD, VIDIOC_STREAMON, &type) < 0)
    {
        LOG_ERROR << "Failed to start streaming";
        return false;
    }

    LOG_INFO << "Camera video streaming on ...";
    return true;
}

bool NvV4L2CameraHelper::stopStream()
{
    /* Stop v4l2 streaming */
    auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(mCamFD, VIDIOC_STREAMOFF, &type))
    {
        LOG_ERROR << "Failed to stop streaming";
        return false;
    }

    LOG_INFO << "Camera video streaming off ...";
    return true;
}

void NvV4L2CameraHelper::operator()()
{
    int fds;
    fd_set rset;

    fds = mCamFD;
    FD_ZERO(&rset);
    FD_SET(fds, &rset);
    mRunning = true;

    /* Wait for camera event with timeout = 5000 ms */
    while (select(fds + 1, &rset, NULL, NULL, NULL) > 0 && mRunning)
    {
        if (FD_ISSET(fds, &rset))
        {
            struct v4l2_buffer v4l2_buf;

            /* Dequeue a camera buff */
            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            v4l2_buf.memory = V4L2_MEMORY_DMABUF;
            if (ioctl(mCamFD, VIDIOC_DQBUF, &v4l2_buf) < 0)
            {
                LOG_ERROR << "Failed to dequeue camera buff";
                break;
            }

            // lock
            std::lock_guard<std::mutex> lock(mBufferFDMutex);
            auto frameItr = mBufferFD.find(v4l2_buf.m.fd);  
            if(frameItr == mBufferFD.end())          
            {
                LOG_FATAL << " mBufferFD failed. fd<" << v4l2_buf.m.fd << "> size<" << mBufferFD.size() << ">";
            }
            mSendFrame(frameItr->second);
            mBufferFD.erase(frameItr);
        }
    }
}

bool NvV4L2CameraHelper::queueBufferToCamera()
{
    while(true)
    {
        auto frame = mMakeFrame();
        if(!frame.get()){
            break;
        }
        auto dmaFDWrapper = static_cast<DMAFDWrapper *>(frame->data());

        struct v4l2_buffer buf;

        /* Query camera v4l2 buf length */
        memset(&buf, 0, sizeof buf);
        buf.index = dmaFDWrapper->getIndex();
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_DMABUF;

        if (ioctl(mCamFD, VIDIOC_QUERYBUF, &buf) < 0){
            LOG_ERROR << "Failed to query buff";
            return false;
        }
        buf.m.fd = (unsigned long)dmaFDWrapper->tempFD;

        {
            //lock
            std::lock_guard<std::mutex> lock(mBufferFDMutex);
            mBufferFD.insert(make_pair(buf.m.fd, frame));
        }

        if (ioctl(mCamFD, VIDIOC_QBUF, &buf) < 0){
            LOG_ERROR << "Failed to enqueue buffers";
            return false;
        }
    }
    return true;
}

bool NvV4L2CameraHelper::requestCameraBuff()
{
    /* Request camera v4l2 buffer */
    struct v4l2_requestbuffers rb;
    memset(&rb, 0, sizeof(rb));
    rb.count = mMaxConcurrentFrames;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_DMABUF;
    if (ioctl(mCamFD, VIDIOC_REQBUFS, &rb) < 0)
    {
        LOG_ERROR << "Failed to request v4l2 buffers";
        return false;
    }

    if (rb.count != mMaxConcurrentFrames)
    {
        LOG_ERROR << "V4l2 buffer number is not as desired";
        return false;
    }


    if(!queueBufferToCamera()){
        return false; 
    }

    return true;
}

bool NvV4L2CameraHelper::start(uint32_t width, uint32_t height, uint32_t _maxConcurrentFrames, bool isMirror)
{
    mCamHeight = height;
    mCamWidth = width;
    mMaxConcurrentFrames = _maxConcurrentFrames;
    bool status = false;
    status = cameraInitialize(isMirror);
    if (status == false)
    {
        LOG_ERROR << "Camera Initialization Failed";
        return false;
    }
    status = requestCameraBuff();
    if (status == false)
    {
        LOG_ERROR << "Buffer Preparation Failed";
        return false;
    }
    status = startStream();
    if (status == false)
    {
        LOG_ERROR << "Start Stream Failed";
        return false;
    }
    mThread = std::thread(std::ref(*this));

    return true;
}

bool NvV4L2CameraHelper::stop()
{
    LOG_INFO << "STOP SIGNAL STARTING";

    mRunning = false;

    if (!stopStream())
    {
        LOG_ERROR << "Stop Stream Failed";
        return false;
    }

    mThread.join();

    LOG_INFO << "Coming out of stop helper";

    return true;
}