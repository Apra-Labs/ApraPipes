#include <cmath>
#include "Overlay.h"
#include "Logger.h"
#include "AIPExceptions.h"

#include "H264EncoderV4L2Helper.h"

inline bool checkv4l2(int ret, int iLine, const char *szFile, std::string message, bool raiseException)
{
    if (ret < 0)
    {
        std::string errorMsg = std::string(szFile) + ":" + std::to_string(iLine) + "<" + message + ">";
        if (raiseException)
        {
            throw AIPException(AIP_FATAL, errorMsg.c_str());
        }

        LOG_ERROR << errorMsg;
        return false;
    }

    return true;
}

#define CHECKV4L2(call, message, raiseException) checkv4l2(call, __LINE__, __FILE__, message, raiseException)

std::shared_ptr<H264EncoderV4L2Helper> H264EncoderV4L2Helper::create(enum v4l2_memory memType, uint32_t pixelFormat, uint32_t width, uint32_t height, uint32_t step, uint32_t bitrate, bool enableMotionVectors, int motionVectorThreshold, uint32_t fps, std::string h264FrameOutputPinId, std::string motionVectorFramePinId,  framemetadata_sp h264Metadata, std::function<frame_sp(size_t size, string& pinId)> makeFrame, SendFrameContainer sendFrameContainer)
{
    auto instance = std::make_shared<H264EncoderV4L2Helper>(memType, pixelFormat, width, height, step, bitrate, enableMotionVectors, motionVectorThreshold, fps,h264FrameOutputPinId, motionVectorFramePinId, h264Metadata, makeFrame, sendFrameContainer);
    instance->setSelf(instance);

    return instance;
}

void H264EncoderV4L2Helper::setSelf(std::shared_ptr<H264EncoderV4L2Helper> &self)
{
    mSelf = self;
}

H264EncoderV4L2Helper::H264EncoderV4L2Helper(enum v4l2_memory memType, uint32_t pixelFormat, uint32_t width, uint32_t height, uint32_t step, uint32_t bitrate, bool _enableMotionVectors, int _motionVectorThreshold, uint32_t fps, std::string _h264FrameOutputPinId, std::string _motionVectorFramePinId,  framemetadata_sp _h264Metadata, std::function<frame_sp(size_t size, string& pinId)> _makeFrame, SendFrameContainer sendFrameContainer) : mSendFrameContainer(sendFrameContainer), mFD(-1), mWidth(width), mHeight(height), enableMotionVectors(_enableMotionVectors), motionVectorThreshold(_motionVectorThreshold), h264FrameOutputPinId(_h264FrameOutputPinId), motionVectorFramePinId(_motionVectorFramePinId), h264Metadata(_h264Metadata), makeFrame(_makeFrame)
{
    initV4L2();

    mCapturePlane = std::make_unique<AV4L2ElementPlane>(mFD, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, V4L2_PIX_FMT_H264, V4L2_MEMORY_MMAP);
    mOutputPlane = std::make_unique<AV4L2ElementPlane>(mFD, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, V4L2_PIX_FMT_YUV420M, memType);

    mCapturePlane->setPlaneFormat(width, height);
    mOutputPlane->setPlaneFormat(width, height);

    initEncoderParams(bitrate, fps);

    mOutputPlane->setupPlane();
    mCapturePlane->setupPlane();

    mOutputPlane->setStreamStatus(true);
    mCapturePlane->setStreamStatus(true);

    mCapturePlane->setDQThreadCallback(std::bind(&H264EncoderV4L2Helper::capturePlaneDQCallback, this, std::placeholders::_1));
    mCapturePlane->startDQThread();

    mCapturePlane->qAllBuffers();

    if(memType == V4L2_MEMORY_DMABUF)
    {
        mConverter = std::make_unique<V4L2CUDMABufYUV420Converter>(width, height, mOutputPlane->mFormat);
    }
    else if (pixelFormat == V4L2_PIX_FMT_YUV420M)
    {
        mConverter = std::make_unique<V4L2CUYUV420Converter>(width, height, mOutputPlane->mFormat);
    }
    else if(pixelFormat == V4L2_PIX_FMT_RGB24)
    {
        mConverter = std::make_unique<V4L2CURGBToYUV420Converter>(width, height, step, mOutputPlane->mFormat);
    }
    else
    {
        throw AIPException(AIP_FATAL, "Unimplemented colorspace<>" + std::to_string(pixelFormat));
    }
}

H264EncoderV4L2Helper::~H264EncoderV4L2Helper()
{
    processEOS();
    mOutputPlane->deinitPlane();
    mCapturePlane->deinitPlane();

    mOutputPlane.reset();
    mCapturePlane.reset();

    termV4L2();
}

void H264EncoderV4L2Helper::stop()
{
    mSelf.reset();
}

void H264EncoderV4L2Helper::termV4L2()
{
    if (mFD != -1)
    {
        v4l2_close(mFD);
        LOG_FATAL << "Device closed, fd = " << mFD;
    }
}

void H264EncoderV4L2Helper::initV4L2()
{
    mFD = v4l2_open("/dev/nvhost-msenc", O_RDWR);
    if (mFD == -1)
    {
        throw AIPException(AIP_FATAL, "Could not open device nvhost-msenc");
    }

    struct v4l2_capability caps;
    auto ret = v4l2_ioctl(mFD, VIDIOC_QUERYCAP, &caps);
    if (ret != 0)
    {
        throw AIPException(AIP_FATAL, "Error in VIDIOC_QUERYCAP");
    }

    if (!(caps.capabilities & V4L2_CAP_VIDEO_M2M_MPLANE))
    {
        throw AIPException(AIP_FATAL, "Device does not support V4L2_CAP_VIDEO_M2M_MPLANE");
    }
}


int
H264EncoderV4L2Helper::setExtControlsMV(v4l2_ext_controls &ctl)
{
    int ret;

    ret = v4l2_ioctl(mFD, VIDIOC_S_EXT_CTRLS, &ctl);

    return ret;
}


int
H264EncoderV4L2Helper::enableMotionVectorReporting()
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_ENABLE_METADATA_MV;
    control.value = 1;

    setExtControlsMV(ctrls);
    return 1;
}

void H264EncoderV4L2Helper::initEncoderParams(uint32_t bitrate, uint32_t fps)
{
    setBitrate(bitrate);
    setProfile();
    setLevel();
    setFrameRate(fps, 1);
    if(enableMotionVectors)
    {
        enableMotionVectorReporting();
    }
}

void H264EncoderV4L2Helper::setBitrate(uint32_t bitrate)
{
    struct v4l2_ext_control control;
    memset(&control, 0, sizeof(control));

    control.id = V4L2_CID_MPEG_VIDEO_BITRATE;
    control.value = bitrate;

    CHECKV4L2(setExtControls(control), "Setting encoder bitrate", true);
}

void H264EncoderV4L2Helper::setProfile()
{
    struct v4l2_ext_control control;
    memset(&control, 0, sizeof(control));

    control.id = V4L2_CID_MPEG_VIDEO_H264_PROFILE;
    control.value = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH;

    CHECKV4L2(setExtControls(control), "Setting encoder profile ", true);

    memset(&control, 0, sizeof(control));

    control.id = V4L2_CID_MPEG_VIDEOENC_NUM_BFRAMES;
    control.value = 0;

    CHECKV4L2(setExtControls(control), "Setting V4L2_CID_MPEG_VIDEOENC_NUM_BFRAMES ", true);
}

void H264EncoderV4L2Helper::setLevel()
{
    struct v4l2_ext_control control;
    memset(&control, 0, sizeof(control));

    control.id = V4L2_CID_MPEG_VIDEO_H264_LEVEL;
    control.value = (enum v4l2_mpeg_video_h264_level)V4L2_MPEG_VIDEO_H264_LEVEL_5_0;

    CHECKV4L2(setExtControls(control), "Setting encoder level ", true);
}

void H264EncoderV4L2Helper::setFrameRate(uint32_t framerate_num, uint32_t framerate_den)
{
    struct v4l2_streamparm parms;
    memset(&parms, 0, sizeof(parms));

    parms.parm.output.timeperframe.numerator = framerate_den;
    parms.parm.output.timeperframe.denominator = framerate_num;
    parms.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;

    auto ret = v4l2_ioctl(mFD, VIDIOC_S_PARM, &parms);
    if (ret)
    {
        throw AIPException(AIP_FATAL, "setFrameRate failed");
    }
}

int H264EncoderV4L2Helper::setExtControls(v4l2_ext_control &control)
{
    struct v4l2_ext_controls ctrls;
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    return v4l2_ioctl(mFD, VIDIOC_S_EXT_CTRLS, &ctrls);
}

int
H264EncoderV4L2Helper::getExtControls(v4l2_ext_controls &ctl)
{
    int ret;

    ret = v4l2_ioctl(mFD, VIDIOC_G_EXT_CTRLS, &ctl);

    return ret;
}

int
H264EncoderV4L2Helper::getMotionVectors(uint32_t buffer_index,
        v4l2_ctrl_videoenc_outputbuf_metadata_MV &enc_mv_metadata)
{
    v4l2_ctrl_video_metadata metadata;
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    metadata.buffer_index = buffer_index;
    metadata.VideoEncMetadataMV = &enc_mv_metadata;

    control.id = V4L2_CID_MPEG_VIDEOENC_METADATA_MV;
    control.string = (char *)&metadata;

    getExtControls(ctrls);

    return 1;
}

void H264EncoderV4L2Helper::serializeMotionVectors(v4l2_ctrl_videoenc_outputbuf_metadata_MV enc_mv_metadata, frame_container &frames)
{
    uint32_t numMVs = enc_mv_metadata.bufSize / sizeof(MVInfo);
    MVInfo *pInfo = enc_mv_metadata.pMVInfo;

    std::vector<CircleOverlay> circleOverlays;
    CompositeOverlay compositeOverlay;

    int totalMacroblockInRow = floor(mWidth / 16); // Tells about the total number of macro blocks in each row.
    for (uint32_t i = 0; i < numMVs; i++, pInfo++) // numMVs is the total macroblock in the frame.
    {

        if (abs(pInfo->mv_x) > motionVectorThreshold || abs(pInfo->mv_y) > motionVectorThreshold)
        {
            auto tempY = floor(i / totalMacroblockInRow); // i represents current macroblock , To get the y offset of macroblock the current macroblock is divided by macroblock across width.
            auto y = tempY * 16 + 8; // Here every macroblock is of 16x16 , So multiply it by 16. To get to the centre of the macroblock add it by 8.
            auto tempX = floor(i % totalMacroblockInRow);
            auto x = tempX * 16 + 8;
            CircleOverlay circleOverlay;
            circleOverlay.x1 = x;
            circleOverlay.y1 = y;
            circleOverlay.radius = 1;
            circleOverlays.push_back(circleOverlay);
        }
    }

    for (auto &circleOverlay : circleOverlays)
    {
        compositeOverlay.add(&circleOverlay);
    }

    if (circleOverlays.size())
    {
        DrawingOverlay drawingOverlay;
        drawingOverlay.add(&compositeOverlay);
        auto serializeSize = drawingOverlay.mGetSerializeSize();
        serializeSize += 100;
        auto motionVectorFrame = makeFrame(serializeSize, motionVectorFramePinId);
        drawingOverlay.serialize(motionVectorFrame);
        frames.insert(make_pair(motionVectorFramePinId, motionVectorFrame));
    }
}
void H264EncoderV4L2Helper::capturePlaneDQCallback(AV4L2Buffer *buffer)
{
    auto frame = frame_sp(frame_opool.construct(buffer->planesInfo[0].data, buffer->v4l2_buf.m.planes[0].bytesused), std::bind(&H264EncoderV4L2Helper::reuseCatureBuffer, this, std::placeholders::_1, buffer->getIndex(), mSelf));
    frame->setMetadata(h264Metadata);
    frame_container frames;
    frame->timestamp = incomingTimeStamp.front();
    incomingTimeStamp.pop();
    frames.insert(make_pair(h264FrameOutputPinId, frame));

    if (enableMotionVectors)
    {
        v4l2_ctrl_videoenc_outputbuf_metadata_MV enc_mv_metadata;
        getMotionVectors(buffer->v4l2_buf.index, enc_mv_metadata);
        serializeMotionVectors(enc_mv_metadata, frames);
    }

    mSendFrameContainer(frames);
    mConverter->releaseFrame();
}

void H264EncoderV4L2Helper::reuseCatureBuffer(ExtFrame *pointer, uint32_t index, std::shared_ptr<H264EncoderV4L2Helper> self)
{
    // take care of destruction case
    frame_opool.free(pointer);
    mCapturePlane->qBuffer(index);
}

bool H264EncoderV4L2Helper::process(frame_sp& frame)
{
    incomingTimeStamp.push(frame->timestamp);
    auto buffer = mOutputPlane->getFreeBuffer();
    if (!buffer)
    {
        return true;
    }

    mConverter->process(frame, buffer);
    mOutputPlane->qBuffer(buffer->getIndex());

    return true;
}

bool H264EncoderV4L2Helper::processEOS()
{
    auto buffer = mOutputPlane->getFreeBuffer();
    if (!buffer)
    {
        return true;
    }

    mOutputPlane->setEOSFlag(buffer);
    mOutputPlane->qBuffer(buffer->getIndex());

    mCapturePlane->waitForDQThread(2000); // blocking call - waits for 2 secs for thread to exit

    return true;
}