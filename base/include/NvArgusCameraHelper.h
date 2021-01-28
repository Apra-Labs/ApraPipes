#pragma once

#include <memory>
#include <thread>

#include "ExtFrame.h"
#include <boost/pool/object_pool.hpp>

#include <Argus/Argus.h>
#include "DMABuffer.h"

class NvArgusCameraHelper
{
public:
    typedef std::function<void (frame_sp&)> SendFrame;
    static std::shared_ptr<NvArgusCameraHelper> create(SendFrame sendFrame);

    NvArgusCameraHelper();
    ~NvArgusCameraHelper();

    bool start(uint32_t width, uint32_t height, uint32_t fps);
    bool stop(); // blocking call

    void operator()();
private:
    void setSelf(std::shared_ptr<NvArgusCameraHelper> &mother);
    void releaseBufferToCamera(ExtFrame *pointer, std::shared_ptr<NvArgusCameraHelper> self, Argus::Buffer* buffer);

private:
    std::shared_ptr<NvArgusCameraHelper> mSelf;
    SendFrame mSendFrame;
    std::thread mThread;
    bool mRunning;

    boost::object_pool<ExtFrame> frame_opool;

private:
    EGLDisplay eglDisplay;
    uint32_t numBuffers;
    DMABuffer **nativeBuffers;
    Argus::UniqueObj<Argus::Buffer> *buffers;

    Argus::UniqueObj<Argus::CameraProvider> cameraProvider;
    Argus::UniqueObj<Argus::CaptureSession> captureSession;
    Argus::UniqueObj<Argus::OutputStream> outputStream;
};