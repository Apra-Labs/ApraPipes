#pragma once
#include <boost/pool/object_pool.hpp>

#include "ExtFrame.h"
#include "AV4L2ElementPlane.h"
#include "V4L2CUYUV420Converter.h"
#include <v4l2_nv_extensions.h>

class H264EncoderV4L2Helper
{
public:
    typedef std::function<void(frame_container& errorFrame)> SendFrameContainer;    
    static std::shared_ptr<H264EncoderV4L2Helper> create(enum v4l2_memory memType, uint32_t pixelFormat, uint32_t width, uint32_t height, uint32_t step, uint32_t bitrate, bool enableMotionVectors, int motionVectorThreshold, uint32_t fps, std::string h264FrameOutputPinId, std::string motionVectorFramePinId,  framemetadata_sp h264Metadata, std::function<frame_sp(size_t size, string& pinId)> makeFrame, SendFrameContainer sendFrameContainer);

    H264EncoderV4L2Helper(enum v4l2_memory memType, uint32_t pixelFormat, uint32_t width, uint32_t height, uint32_t step, uint32_t bitrate, bool enableMotionVectors, int motionVectorThreshold, uint32_t fps,std::string h264FrameOutputPinId, std::string motionVectorFramePinId,  framemetadata_sp h264Metadata, std::function<frame_sp(size_t size, string& pinId)> makeFrame, SendFrameContainer sendFrameContainer);
    ~H264EncoderV4L2Helper();

    void stop();

    // data is cuda rgb data pointer and should be already synced
    bool process(frame_sp& frame);

private:
    void setSelf(std::shared_ptr<H264EncoderV4L2Helper> &mother);

    void initV4L2();
    void termV4L2();

    void setBitrate(uint32_t bitrate);
    void setProfile();
    void setLevel();
    void setFrameRate(uint32_t framerate_num, uint32_t framerate_den);

    int setExtControlsMV(v4l2_ext_controls &ctl);
    int enableMotionVectorReporting();
    void initEncoderParams(uint32_t bitrate, uint32_t fps);
    int setExtControls(v4l2_ext_control &control);

    void capturePlaneDQCallback(AV4L2Buffer *buffer);
    void reuseCatureBuffer(ExtFrame *pointer, uint32_t index, std::shared_ptr<H264EncoderV4L2Helper> self);

    bool processEOS();
    int getExtControls(v4l2_ext_controls &ctl);
    int getMotionVectors(uint32_t buffer_index,
            v4l2_ctrl_videoenc_outputbuf_metadata_MV &enc_mv_metadata);
    void serializeMotionVectors(v4l2_ctrl_videoenc_outputbuf_metadata_MV enc_mv_metadata, frame_container &frames);
private:
    std::shared_ptr<H264EncoderV4L2Helper> mSelf;
    int mFD;
    std::unique_ptr<AV4L2ElementPlane> mOutputPlane;
    std::unique_ptr<AV4L2ElementPlane> mCapturePlane;
    boost::object_pool<ExtFrame> frame_opool;
    SendFrameContainer mSendFrameContainer;
    int mWidth = 0;
    int mHeight = 0;
    bool enableMotionVectors;
    int motionVectorThreshold;
    std::string h264FrameOutputPinId;
    std::string motionVectorFramePinId;
    framemetadata_sp h264Metadata;
    std::function<frame_sp(size_t size, string& pinId)> makeFrame;
    std::unique_ptr<V4L2CUYUV420Converter> mConverter;
protected:
    std::queue <uint64_t> incomingTimeStamp;
};