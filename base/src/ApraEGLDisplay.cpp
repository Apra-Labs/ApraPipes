#include "ApraEGLDisplay.h"
#include "AIPExceptions.h"
#include "Logger.h"
#include <EGL/eglext.h>

boost::shared_ptr<ApraEGLDisplay> ApraEGLDisplay::instance;

EGLDisplay ApraEGLDisplay::getEGLDisplay()
{
    if (!instance.get())
    {
        // assuming it comes here only once in main thread
        instance.reset(new ApraEGLDisplay());
    }

    return instance->mEGLDisplay;
}

ApraEGLDisplay::ApraEGLDisplay()
{
    mEGLDisplay = EGL_NO_DISPLAY;

    // First try EGL_DEFAULT_DISPLAY - works when display is connected
    mEGLDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (mEGLDisplay != EGL_NO_DISPLAY)
    {
        if (eglInitialize(mEGLDisplay, NULL, NULL))
        {
            LOG_INFO << "EGL initialized via default display";
            return;
        }
        // If initialize fails, try device extension
        mEGLDisplay = EGL_NO_DISPLAY;
    }

    // Try EGL device extension for headless operation (JetPack 5.x)
    // This requires EGL_EXT_platform_device extension
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

    if (eglQueryDevicesEXT && eglGetPlatformDisplayEXT)
    {
        EGLDeviceEXT devices[8];
        EGLint numDevices;
        if (eglQueryDevicesEXT(8, devices, &numDevices) && numDevices > 0)
        {
            // Try first device (usually the GPU)
            mEGLDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[0], NULL);
            if (mEGLDisplay != EGL_NO_DISPLAY)
            {
                if (eglInitialize(mEGLDisplay, NULL, NULL))
                {
                    LOG_INFO << "EGL initialized via device extension (headless)";
                    return;
                }
                mEGLDisplay = EGL_NO_DISPLAY;
            }
        }
    }

    if (mEGLDisplay == EGL_NO_DISPLAY)
    {
        throw AIPException(AIP_FATAL, "eglGetDisplay failed - no display available");
    }
}

ApraEGLDisplay::~ApraEGLDisplay()
{
    if (!eglTerminate(mEGLDisplay))
    {
        LOG_ERROR << "ERROR eglTerminate failed";
    }

    if (!eglReleaseThread())
    {
        LOG_ERROR << "ERROR eglReleaseThread failed";
    }
}
