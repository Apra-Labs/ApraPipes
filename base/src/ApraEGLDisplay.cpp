#include "ApraEGLDisplay.h"
#include "AIPExceptions.h"

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
    mEGLDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (mEGLDisplay == EGL_NO_DISPLAY)
    {
        throw AIPException(AIP_FATAL, "eglGetDisplay failed");
    }

    if (!eglInitialize(mEGLDisplay, NULL, NULL))
    {
        throw AIPException(AIP_FATAL, "eglInitialize failed");
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
