#include "AV4L2ElementPlane.h"
#include "Logger.h"
#include "AIPExceptions.h"

AV4L2ElementPlane::AV4L2ElementPlane(int mFD, uint32_t type, uint32_t pixelFormat, enum v4l2_memory memType) : mFD(mFD), mType(type),
                                                                                     mPixelFormat(pixelFormat), mCount(10),
                                                                                     mStreamOn(false), mBuffers(nullptr),
                                                                                     mCallback(nullptr), mDQThread(0), mDQThreadRunning(false),
                                                                                     mStopDQThread(false), mFreeCount(mCount), mMemType(memType)
{
    mNumPlanes = 1;
    if (pixelFormat == V4L2_PIX_FMT_YUV420M)
    {
        mNumPlanes = 3;
    }

    pthread_mutex_init(&plane_lock, NULL);
    pthread_cond_init(&plane_cond, NULL);
}

AV4L2ElementPlane::~AV4L2ElementPlane()
{
    pthread_mutex_destroy(&plane_lock);
    pthread_cond_destroy(&plane_cond);
}

void AV4L2ElementPlane::setPlaneFormat(uint32_t width, uint32_t height)
{
    memset(&mFormat, 0, sizeof(struct v4l2_format));

    mFormat.type = mType;
    mFormat.fmt.pix_mp.pixelformat = mPixelFormat;
    mFormat.fmt.pix_mp.width = width;
    mFormat.fmt.pix_mp.height = height;
    mFormat.fmt.pix_mp.num_planes = mNumPlanes;
    mFormat.fmt.pix_mp.plane_fmt[0].sizeimage = 2 * 1024 * 1024; // this line is not required for yuv420 - test this

    auto ret = v4l2_ioctl(mFD, VIDIOC_S_FMT, &mFormat);
    if (ret)
    {
        throw AIPException(AIP_FATAL, "Error in setPlaneFormat VIDIOC_S_FMT");
    }
}

void AV4L2ElementPlane::setupPlane()
{
    reqbufs(mCount);

    if(mMemType == V4L2_MEMORY_DMABUF)
    {
        return;
    }

    for (auto i = 0; i < mCount; i++)
    {
        queryBuffer(i);
        exportBuffer(i);
        mBuffers[i]->map();
    }
}

void AV4L2ElementPlane::reqbufs(uint32_t count)
{
    struct v4l2_requestbuffers v4l2_reqbufs;
    int ret;

    memset(&v4l2_reqbufs, 0, sizeof(struct v4l2_requestbuffers));
    v4l2_reqbufs.count = count;
    v4l2_reqbufs.type = mType;

    v4l2_reqbufs.memory = mMemType;
    ret = v4l2_ioctl(mFD, VIDIOC_REQBUFS, &v4l2_reqbufs);
    if (ret)
    {
        throw AIPException(AIP_FATAL, "Error in VIDIOC_REQBUFS");
    }

    if (count)
    {
        mCount = v4l2_reqbufs.count;
        mFreeCount = mCount;
        mBuffers = new AV4L2Buffer *[mCount];
        for (uint32_t i = 0; i < mCount; i++)
        {
            mBuffers[i] = new AV4L2Buffer(i, mType, mMemType, mNumPlanes);
        }

        mTempBuffer = new AV4L2Buffer(0, mType, mMemType, mNumPlanes);
    }
    else
    {
        if (mBuffers)
        {
            for (auto i = 0; i < mCount; i++)
            {
                delete mBuffers[i];
            }

            delete[] mBuffers;

            mBuffers = nullptr;
        }

        if (mTempBuffer)
        {
            delete mTempBuffer;
        }
    }
}

void AV4L2ElementPlane::queryBuffer(uint32_t i)
{
    auto &v4l2_buf = mBuffers[i]->v4l2_buf;

    auto ret = v4l2_ioctl(mFD, VIDIOC_QUERYBUF, &v4l2_buf);
    if (ret)
    {
        throw AIPException(AIP_FATAL, "Error in QueryBuf");
    }
}

void AV4L2ElementPlane::exportBuffer(uint32_t i)
{
    struct v4l2_exportbuffer expbuf;
    memset(&expbuf, 0, sizeof(expbuf));
    expbuf.type = mType;
    expbuf.index = i;

    int ret;
    for (auto j = 0; j < mNumPlanes; j++)
    {
        expbuf.plane = j;
        ret = v4l2_ioctl(mFD, VIDIOC_EXPBUF, &expbuf);
        if (ret || expbuf.fd == -1)
        {
            throw AIPException(AIP_FATAL, "Error in ExportBuf for Buffer ");
        }

        mBuffers[i]->planesInfo[j].fd = expbuf.fd;
    }
}

void AV4L2ElementPlane::setStreamStatus(bool status)
{
    int ret;

    if (status == mStreamOn)
    {
        LOG_ERROR << "Already in " << (status ? "STREAMON" : "STREAMOFF");
        return;
    }

    pthread_mutex_lock(&plane_lock);
    if (status)
    {
        ret = v4l2_ioctl(mFD, VIDIOC_STREAMON, &mType);
    }
    else
    {
        ret = v4l2_ioctl(mFD, VIDIOC_STREAMOFF, &mType);
    }
    if (ret)
    {
        pthread_mutex_unlock(&plane_lock);
        throw AIPException(AIP_FATAL, "Error in setStreamStatus");
    }

    mStreamOn = status;
    if (!mStreamOn)
    {
        pthread_cond_broadcast(&plane_cond);
    }

    pthread_mutex_unlock(&plane_lock);
}

void AV4L2ElementPlane::setDQThreadCallback(dqThreadCallback callback)
{
    if (mDQThreadRunning)
    {
        LOG_ERROR << "setDQThreadCallback failed. DQThread is running";
        return;
    }

    mCallback = callback;
}

int AV4L2ElementPlane::dqBuffer(AV4L2Buffer **buffer, uint32_t retries)
{
    auto &v4l2_buf = mTempBuffer->v4l2_buf;
    int ret;

    do
    {
        ret = v4l2_ioctl(mFD, VIDIOC_DQBUF, &v4l2_buf);
        if(ret == 0)
        {
            // success
        }
        else if (errno == EAGAIN)
        {
            pthread_mutex_lock(&plane_lock);
            if (v4l2_buf.flags & V4L2_BUF_FLAG_LAST)
            {
                pthread_mutex_unlock(&plane_lock);
                LOG_INFO << "DQing V4L2_BUF_FLAG_LAST";
                break;
            }
            pthread_mutex_unlock(&plane_lock);

            if (retries-- == 0)
            {
                LOG_FATAL << "Error while DQing buffer: Resource temporarily unavailable";
                break;
            }          
        }
        else
        {
            LOG_FATAL << "Error while DQing buffer <>" << errno;
            *buffer = nullptr;
            return ret;
        }
    } while(ret);

    pthread_mutex_lock(&plane_lock);
    auto curBuffer = mBuffers[v4l2_buf.index];
    *buffer = curBuffer;
    auto numPlanes = curBuffer->getNumPlanes();

    for (uint32_t i = 0; i < numPlanes; i++)
    {
        curBuffer->v4l2_buf.m.planes[i].bytesused = v4l2_buf.m.planes[i].bytesused;
    }

    pthread_cond_broadcast(&plane_cond);
    pthread_mutex_unlock(&plane_lock);

    return ret;
}

int AV4L2ElementPlane::qBuffer(uint32_t index)
{
    pthread_mutex_lock(&plane_lock);
    auto buffer = mBuffers[index];

    auto ret = v4l2_ioctl(mFD, VIDIOC_QBUF, &buffer->v4l2_buf);
    if (ret)
    {
        LOG_ERROR << "Error while Qing buffer";
    }
    else
    {
        pthread_cond_broadcast(&plane_cond);
    }
    pthread_mutex_unlock(&plane_lock);

    return ret;
}

void AV4L2ElementPlane::startDQThread()
{
    pthread_mutex_lock(&plane_lock);
    if (mDQThreadRunning)
    {
        LOG_FATAL << "DQ Thread already started";
        pthread_mutex_unlock(&plane_lock);
        return;
    }

    pthread_create(&mDQThread, NULL, dqThread, this);
    mDQThreadRunning = true;
    pthread_mutex_unlock(&plane_lock);
}

void* AV4L2ElementPlane::dqThread(void *data)
{
    auto *plane = static_cast<AV4L2ElementPlane *>(data);
    plane->mStopDQThread = false;

    while (!plane->mStopDQThread)
    {
        AV4L2Buffer *buffer;

        if (plane->dqBuffer(&buffer, -1) < 0)
        {
            LOG_FATAL << "DQ BUFFER ERROR EXITING THREAD " << errno;
            break;
        }

        if (buffer->v4l2_buf.m.planes[0].bytesused == 0)
        {
            LOG_INFO << "received EOS";
            break;
        }

        plane->mCallback(buffer);
    }
    plane->mStopDQThread = false;

    pthread_mutex_lock(&plane->plane_lock);
    plane->mDQThreadRunning = false;
    pthread_cond_broadcast(&plane->plane_cond);
    pthread_mutex_unlock(&plane->plane_lock);

    LOG_INFO << "DQ Thread exiting";
    return nullptr;
}

int AV4L2ElementPlane::waitForDQThread(uint32_t max_wait_ms)
{
    if(!mDQThread)
    {
        LOG_INFO << "Thread already exited";
        return 0;
    }
    struct timespec timeToWait;
    struct timeval now;
    int return_val = 0;
    int ret = 0;

    gettimeofday(&now, NULL);

    timeToWait.tv_nsec = (now.tv_usec + (max_wait_ms % 1000) * 1000L) * 1000L;
    timeToWait.tv_sec = now.tv_sec + max_wait_ms / 1000 +
                        timeToWait.tv_nsec / 1000000000L;
    timeToWait.tv_nsec = timeToWait.tv_nsec % 1000000000L;

    pthread_mutex_lock(&plane_lock);
    while (mDQThreadRunning)
    {
        ret = pthread_cond_timedwait(&plane_cond, &plane_lock, &timeToWait);
        if (ret == ETIMEDOUT)
        {
            return_val = -1;
            break;
        }
    }
    pthread_mutex_unlock(&plane_lock);

    if (ret == 0)
    {
        pthread_join(mDQThread, NULL);
        mDQThread = 0;
        LOG_INFO << "Stopped DQ Thread";
    }
    else
    {
        LOG_ERROR << "Timed out waiting for dqthread";
    }

    return return_val;
}

AV4L2Buffer *AV4L2ElementPlane::getFreeBuffer()
{
    if (mFreeCount)
    {
        mFreeCount--;
        return mBuffers[mFreeCount];
    }

    AV4L2Buffer *buffer = nullptr;
    dqBuffer(&buffer, 10);

    return buffer;
}

void AV4L2ElementPlane::qAllBuffers()
{
    for (auto i = 0; i < mFreeCount; i++)
    {
        if (qBuffer(i))
        {
            throw AIPException(AIP_FATAL, "qAllBuffers failed");
        }
    }

    mFreeCount = 0;
}

void AV4L2ElementPlane::deinitPlane()
{
    setStreamStatus(false);
    waitForDQThread(-1); // waiting second time
    if (mMemType == V4L2_MEMORY_MMAP)
    {
        for (uint32_t i = 0; i < mCount; i++)
        {
            mBuffers[i]->unmap();
        }
    }
    reqbufs(0);
    LOG_INFO << "deinitPlane successful";
}

 void AV4L2ElementPlane::setEOSFlag(AV4L2Buffer* buffer)
 {    
    for (uint32_t i = 0; i < mNumPlanes; i++)
    {
        buffer->v4l2_buf.m.planes[i].bytesused = 0;
    }
 }