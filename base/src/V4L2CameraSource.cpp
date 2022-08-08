#include "V4L2CameraSource.h"
#include "FrameMetadata.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <getopt.h> /* getopt_long() */
#include <fcntl.h> /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <linux/videodev2.h>

V4L2CameraSource::V4L2CameraSource(V4L2CameraSourceProps props)
    : Module(SOURCE, "V4L2CameraSource", props), mProps(props)
{
    // auto outputMetadata = framemetadata_sp(new RawImageMetadata(props.width, props.height, ImageMetadata::UYVY, CV_8UC2, size_t(0), CV_8U, FrameMetadata::MemType::HOST, true));
    auto outputMetadata = framemetadata_sp(new RawImageMetadata(props.width, props.height, ImageMetadata::ImageType::BG10, CV_16UC1, 2 * props.width, CV_16U, FrameMetadata::MemType::HOST));
    // auto outputMetadata = framemetadata_sp(new RawImageMetadata(props.width, props.height, ImageMetadata::ImageType::BG10, CV_16UC1, 2 * props.width, CV_16U, FrameMetadata::MemType::HOST));
    mOutputPinId = addOutputPin(outputMetadata);
}

V4L2CameraSource::~V4L2CameraSource() {}

bool V4L2CameraSource::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        return false;
    }

    return true;
}

bool V4L2CameraSource::init()
{
    if (!Module::init())
    {
        return false;
    }
    mHelper.reset(new V4L2CameraSourceHelper(mProps.cameraName));
    if (!mHelper->openCamera())
    {
        LOG_ERROR << "Failed To Open Camera";
        return false;
    }
    if (!mHelper->initCamera())
    {
        LOG_ERROR << " Failed To INIT Camera";
        return false;
    }
    v4l2_format format;
    mHelper->setFormatBGGR10(mProps.width, mProps.height, format);

    auto bufStatus = mHelper->requestBuffers(3);
    if (!bufStatus)
    {
        LOG_ERROR << "Failed To Fetch Buffers";
    }
    auto streamStatus = mHelper->startStreaming();
    if (!streamStatus)
    {
        LOG_ERROR << "Failed To Start Stream";
    }
    return true;
}

bool V4L2CameraSource::term()
{
    auto ret = mHelper->stopStreaming();
    if (!ret)
    {
        LOG_ERROR << "Failed To Stop Stream";
        return false;
    }
    auto deinit = mHelper->deinitCamera();
    if (!deinit)
    {
        LOG_ERROR << "Failed To Deinit Camera";
        return false;
    }
    auto close = mHelper->closeCamera();
    if (!close)
    {
        LOG_ERROR << "Failed To Close Camera";
        return false;
    }
    auto moduleRet = Module::term();

    return moduleRet;
}

bool V4L2CameraSource::produce()
{
    auto frame = makeFrame(mProps.width * mProps.height * 2);
    uint64_t frameSize = 0;
    while (isFrameEmpty(frame))
    {
        boost::this_thread::sleep_for(boost::chrono::microseconds(5000));
        frame = makeFrame(mProps.width * mProps.height * 2);
    }
    if(isFrameEmpty(frame))
    {
        LOG_ERROR << "<============================ Frame is Empty==============================================================>";
    }

    auto getFrameStatus = mHelper->readFrame(static_cast<uint8_t *>(frame->data()), frameSize);

    if(frameSize == 0) ///comment
    {
        auto getFrameStatus = mHelper->readFrame(static_cast<uint8_t *>(frame->data()), frameSize);   
        LOG_ERROR << "Asking For New Frame ";
    }
    if (!getFrameStatus)
    {
        LOG_ERROR << "Failed to Read Frame ";
    }
    
    frame_container frames;
    frames.insert(make_pair(mOutputPinId, frame));
    send(frames);
    return true;
}
