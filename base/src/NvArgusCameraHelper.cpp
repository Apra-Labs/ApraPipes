#include "NvArgusCameraHelper.h"

#include "ExtFrame.h"
#include "Logger.h"
#include "AIPExceptions.h"

#include <Argus/Ext/DolWdrSensorMode.h>
#include <Argus/Ext/PwlWdrSensorMode.h>

NvArgusCameraHelper::NvArgusCameraHelper() : numBuffers(10), mRunning(false)
{
    eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if(eglDisplay == EGL_NO_DISPLAY)
    {
        throw AIPException(AIP_FATAL, "eglGetDisplay failed");
    } 
    
    if (!eglInitialize(eglDisplay, NULL, NULL))
    {
       throw AIPException(AIP_FATAL, "eglInitialize failed");
    }

    nativeBuffers = new DMABuffer *[numBuffers];
    for (auto i = 0; i < numBuffers; i++)
    {
        nativeBuffers[i] = nullptr;
    }

    buffers = new Argus::UniqueObj<Argus::Buffer>[numBuffers];
}

NvArgusCameraHelper::~NvArgusCameraHelper()
{
    LOG_INFO << "in destructor ------------------";

    for (auto i = 0; i < numBuffers; i++)
    {
        if (nativeBuffers[i])
        {
            delete nativeBuffers[i];
            nativeBuffers[i] = nullptr;
        }
    }
    delete[] nativeBuffers;
    eglTerminate(eglDisplay);

    delete[] buffers;

    LOG_INFO << "out of destructor ------------------";
}

std::shared_ptr<NvArgusCameraHelper> NvArgusCameraHelper::create(SendFrame sendFrame)
{
    auto instance = std::make_shared<NvArgusCameraHelper>();
    instance->setSelf(instance);
    instance->mSendFrame = sendFrame;

    return instance;
}

void NvArgusCameraHelper::setSelf(std::shared_ptr<NvArgusCameraHelper> &self)
{
    mSelf = self;
}

void NvArgusCameraHelper::operator()()
{
    mRunning = true;
    Argus::IBufferOutputStream *stream = Argus::interface_cast<Argus::IBufferOutputStream>(outputStream);
        
    while (mRunning)
    {
        Argus::Status status = Argus::STATUS_OK;
        auto buffer = stream->acquireBuffer(Argus::TIMEOUT_INFINITE, &status);
        if (status == Argus::STATUS_END_OF_STREAM)
        {
            /* Timeout or error happen, exit */
            break;
        }

        auto dmaBuf = DMABuffer::fromArgusBuffer(buffer);        
        dmaBuf->tempFD = dmaBuf->getFd(); // doing this for safety, if some downstream module changes the value by mistake
        auto frame = frame_sp(frame_opool.construct(&dmaBuf->tempFD, 4), std::bind(&NvArgusCameraHelper::releaseBufferToCamera, this, std::placeholders::_1, mSelf, buffer));
        mSendFrame(frame);
    }

    LOG_ERROR << "consumer thread exiting";
}

void NvArgusCameraHelper::releaseBufferToCamera(ExtFrame *pointer, std::shared_ptr<NvArgusCameraHelper> self, Argus::Buffer *buffer)
{
    frame_opool.free(pointer);

    Argus::IBufferOutputStream *stream = Argus::interface_cast<Argus::IBufferOutputStream>(outputStream);
    stream->releaseBuffer(buffer);
}

bool NvArgusCameraHelper::start(uint32_t width, uint32_t height, uint32_t fps)
{
    /* Create the Argus::CameraProvider object and get the core interface */
    cameraProvider.reset(Argus::CameraProvider::create());
    Argus::ICameraProvider *iCameraProvider = Argus::interface_cast<Argus::ICameraProvider>(cameraProvider);
    if (!iCameraProvider)
    {
        LOG_ERROR << "Failed to create Argus::CameraProvider";
        return false;
    }

    /* Get the camera devices */
    std::vector<Argus::CameraDevice *> cameraDevices;
    iCameraProvider->getCameraDevices(&cameraDevices);
    if (cameraDevices.size() == 0)
    {
        LOG_ERROR << "No cameras available";
        return false;
    }

    /* Create the capture session using the first device and get the core interface */
    captureSession.reset(
        iCameraProvider->createCaptureSession(cameraDevices[0]));

    Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(captureSession);
    if (!iCaptureSession)
    {
        LOG_ERROR << "Failed to get Argus::ICaptureSession interface";
    }

    /* Create the OutputStream */
    LOG_INFO << "Creating output stream";
    Argus::UniqueObj<Argus::OutputStreamSettings> streamSettings(
        iCaptureSession->createOutputStreamSettings(Argus::STREAM_TYPE_BUFFER));
    Argus::IBufferOutputStreamSettings *iStreamSettings =
        Argus::interface_cast<Argus::IBufferOutputStreamSettings>(streamSettings);
    if (!iStreamSettings)
    {
        LOG_ERROR << "Failed to get Argus::IBufferOutputStreamSettings interface";
        return false;
    }

    /* Configure the OutputStream to use the EGLImage BufferType */
    iStreamSettings->setBufferType(Argus::BUFFER_TYPE_EGL_IMAGE);

    /* Create the OutputStream */
    outputStream.reset(iCaptureSession->createOutputStream(streamSettings.get()));
    Argus::IBufferOutputStream *iBufferOutputStream = Argus::interface_cast<Argus::IBufferOutputStream>(outputStream);

    /* Allocate native buffers */
    Argus::Size2D<uint32_t> streamSize(width, height);
    for (uint32_t i = 0; i < numBuffers; i++)
    {
        nativeBuffers[i] = DMABuffer::create(streamSize, NvBufferColorFormat_NV12, NvBufferLayout_BlockLinear, eglDisplay);
        if (!nativeBuffers[i])
        {
            LOG_ERROR << "Failed to allocate NativeBuffer";
            return false;
        }
    }

    /* Create the Argus::BufferSettings object to configure Argus::Buffer creation */
    Argus::UniqueObj<Argus::BufferSettings> bufferSettings(iBufferOutputStream->createBufferSettings());
    Argus::IEGLImageBufferSettings *iBufferSettings =
        Argus::interface_cast<Argus::IEGLImageBufferSettings>(bufferSettings);
    if (!iBufferSettings)
    {
        LOG_ERROR << "Failed to create Argus::BufferSettings";
        return false;
    }

    /* Create the Buffers for each EGLImage (and release to
       stream for initial capture use) */
    for (uint32_t i = 0; i < numBuffers; i++)
    {
        iBufferSettings->setEGLImage(nativeBuffers[i]->getEGLImage());
        iBufferSettings->setEGLDisplay(eglDisplay);
        buffers[i].reset(iBufferOutputStream->createBuffer(bufferSettings.get()));
        Argus::IBuffer *iBuffer = Argus::interface_cast<Argus::IBuffer>(buffers[i]);

        /* Reference Argus::Argus::Buffer and DMABuffer each other */
        iBuffer->setClientData(nativeBuffers[i]);
        nativeBuffers[i]->setArgusBuffer(buffers[i].get());

        if (!Argus::interface_cast<Argus::IEGLImageBuffer>(buffers[i]))
        {
            LOG_ERROR << "Failed to create Argus::Buffer";
            return false;
        }

        if (iBufferOutputStream->releaseBuffer(buffers[i].get()) != Argus::STATUS_OK)
        {
            LOG_ERROR << "Failed to release Argus::Buffer for capture use";
            return false;
        }
    }

    mThread = std::thread(std::ref(*(mSelf.get())));

    /* Create capture request and enable output stream */
    Argus::UniqueObj<Argus::Request> request(iCaptureSession->createRequest());
    Argus::IRequest *iRequest = Argus::interface_cast<Argus::IRequest>(request);
    if (!iRequest)
    {
        LOG_ERROR << "Failed to create Argus::Request";
        return false;
    }
    iRequest->enableOutputStream(outputStream.get());

    Argus::ISourceSettings *iSourceSettings = Argus::interface_cast<Argus::ISourceSettings>(iRequest->getSourceSettings());
    if (!iSourceSettings)
    {
        LOG_ERROR << "Failed to get Argus::ISourceSettings interface";
        return false;
    }
    iSourceSettings->setFrameDurationRange(Argus::Range<uint64_t>(1e9 / fps));

    Argus::ICameraProperties *iCameraProperties = Argus::interface_cast<Argus::ICameraProperties>(cameraDevices[0]);
    std::vector<Argus::SensorMode*> sensorModes;
    Argus::Status status = iCameraProperties->getAllSensorModes(&sensorModes);
    if (status != Argus::STATUS_OK)
    {
        LOG_ERROR << "Failed to get sensor modes from device";
        return false;
    }
    auto index = 0;
    auto noOfSensors = sensorModes.size();
    LOG_INFO << "Your params width : " << width << " height : " << height << " fps : " << fps;
    for (; index < noOfSensors; index++)
    {
        Argus::ISensorMode *iSensorMode = Argus::interface_cast<Argus::ISensorMode>(sensorModes[index]);
        Argus::Size2D<uint32_t> resolution = iSensorMode->getResolution();
        auto u64Range = iSensorMode->getFrameDurationRange();
        auto fpsMin = 10e9 / u64Range.max();
        auto fpsMax = 10e9 / u64Range.min();

        LOG_INFO << "Matching to width : " << resolution.width() << " height : " << resolution.height() << " fpsRange: " << fpsMin << "-" << fpsMax;

        if (resolution.width() == width && resolution.height() == height)
        {
            if (fps >= fpsMin && fps <= fpsMax)
            {
                break;
            }
        }
    }

    if (index == sensorModes.size())
    {
        throw AIPException(AIP_PARAM_OUTOFRANGE, "Width, Height, FPS is not supported. Please specify params in range");
    }

    iSourceSettings->setSensorMode(sensorModes[index]);

    /* Submit capture requests */
    LOG_INFO << "Starting repeat capture requests";
    if (iCaptureSession->repeat(request.get()) != Argus::STATUS_OK)
    {
        LOG_ERROR << "Failed to start repeat capture request";
        return false;
    }

    return true;
}

bool NvArgusCameraHelper::stop()
{
    LOG_INFO << "STOP SIGNAL STARTING";
    Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(captureSession);
    if (iCaptureSession)
    {
        iCaptureSession->stopRepeat();
    }

    LOG_INFO << "STOP REPEAT DONE";

    Argus::IBufferOutputStream *iBufferOutputStream = Argus::interface_cast<Argus::IBufferOutputStream>(outputStream);
    if (iBufferOutputStream)
    {
        LOG_INFO << "endofstream<>" << iBufferOutputStream->endOfStream();
    }

    if (iCaptureSession)
    {
        iCaptureSession->waitForIdle();
    }

    LOG_INFO << "THREAD JOIN START";

    mRunning = false;
    mThread.join();

    LOG_INFO << "THREAD JOIN END";

    mSelf.reset();

    return true;
}