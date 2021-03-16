#pragma once

#include <memory>
#include <thread>
#include "ExtFrame.h"
#include "NvUtils.h"
#include "nvbuf_utils.h"
#include <map>
#include <mutex>

class NvV4L2CameraHelper
{
public:
    typedef std::function<void (frame_sp&)> SendFrame;

public:
    NvV4L2CameraHelper(SendFrame sendFrame,std::function<frame_sp()> _makeFrame);
    ~NvV4L2CameraHelper();
    static std::shared_ptr<NvV4L2CameraHelper> create(SendFrame sendFrame, std::function<frame_sp()> _makeFrame);

    bool start(uint32_t width, uint32_t height, uint32_t maxConcurrentFrames, bool isMirror);
    bool stop();
    void operator()();
    bool queueBufferToCamera();

private:
    std::thread mThread;
    std::mutex mBufferFDMutex;
    std::function<frame_sp()> mMakeFrame; 

    /* Camera v4l2 context */
    const char * mCamDevname;
    int mCamFD;
    unsigned int mCamPixFmt;
    unsigned int mCamWidth;
    unsigned int mCamHeight;
    uint32_t mMaxConcurrentFrames;

    bool mRunning;
    SendFrame mSendFrame;
    std::map<int, frame_sp> mBufferFD;

    bool cameraInitialize(bool isMirror);
    bool prepareBuffers();    
    bool startStream();
    bool requestCameraBuff();
    bool stopStream();             
};