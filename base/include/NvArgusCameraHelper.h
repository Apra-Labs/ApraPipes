#pragma once

#include <thread>
#include <map>

#include <Argus/Argus.h>
#include "CommonDefs.h"
#include <mutex>

class NvArgusCameraHelper
{
public:
    typedef std::function<void (frame_sp&)> SendFrame;
    typedef std::function<frame_sp ()> MakeFrame;
    static std::shared_ptr<NvArgusCameraHelper> create(uint32_t numBuffers, SendFrame sendFrame, MakeFrame makeFrame);

    NvArgusCameraHelper();
    ~NvArgusCameraHelper();

    bool start(uint32_t width, uint32_t height, uint32_t fps, int cameraId);
    bool stop(); // blocking call
    void toggleAutoWhiteBalance();
    void enableAutoWhiteBalance();
    void disableAutoWhiteBalance();
    bool queueFrameToCamera();

    void operator()();
private:
    void sendFrame(Argus::Buffer *buffer);

private:
    SendFrame mSendFrame;
    MakeFrame mMakeFrame;
    std::thread mThread;
    std::mutex mQueuedFramesMutex;
    bool mRunning;

private:
    uint32_t numBuffers;
    Argus::UniqueObj<Argus::Request> request;
    Argus::ICaptureSession *iCaptureSession;
    Argus::IAutoControlSettings* iAutoControlSettings;
    Argus::IRequest *iRequest;

    Argus::UniqueObj<Argus::Buffer> *buffers;
    std::map<void*, frame_sp> mQueuedFrames;

    Argus::UniqueObj<Argus::CameraProvider> cameraProvider;
    Argus::UniqueObj<Argus::CaptureSession> captureSession;
    Argus::UniqueObj<Argus::OutputStream> outputStream;
};

class NvArgusCameraUtils
{
public:
    static Argus::ICameraProvider *getNvArgusCameraUtils();
    virtual ~NvArgusCameraUtils();
    Argus::ICameraProvider *_getNvArgusCameraUtils();

private:
    NvArgusCameraUtils();
    static boost::shared_ptr<NvArgusCameraUtils> instance;
    Argus::UniqueObj<Argus::CameraProvider> cameraProvider;
};