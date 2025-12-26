#include "ApraEGLDisplay.h"
#include "AIPExceptions.h"
#include "Logger.h"
#include <EGL/eglext.h>
#include <cstdlib>

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

    // For NvBufSurface/DMA operations on headless Jetson, we need the NVIDIA GPU's
    // EGL display, not Xvfb's software display. Try device extension FIRST.
    // This gives us direct GPU access without requiring X11/Xvfb.
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
            // Try first device (usually the NVIDIA GPU)
            mEGLDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[0], NULL);
            if (mEGLDisplay != EGL_NO_DISPLAY)
            {
                if (eglInitialize(mEGLDisplay, NULL, NULL))
                {
                    LOG_INFO << "EGL initialized via device extension (GPU direct)";
                    return;
                }
                mEGLDisplay = EGL_NO_DISPLAY;
            }
        }
    }

    // Fallback to default display - works when real display is connected
    // Note: With Xvfb, this may succeed but won't support NvBufSurface operations
    mEGLDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (mEGLDisplay != EGL_NO_DISPLAY)
    {
        if (eglInitialize(mEGLDisplay, NULL, NULL))
        {
            LOG_INFO << "EGL initialized via default display";
            return;
        }
        mEGLDisplay = EGL_NO_DISPLAY;
    }

    if (mEGLDisplay == EGL_NO_DISPLAY)
    {
        LOG_WARNING << "EGL display not available - DMA/GPU operations will fail. "
                    << "This is expected on headless systems without GPU access.";
    }
}

bool ApraEGLDisplay::isAvailable()
{
    return getEGLDisplay() != EGL_NO_DISPLAY;
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
