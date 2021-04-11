#pragma once

#include <boost/shared_ptr.hpp>
#include "EGL/egl.h"

class ApraEGLDisplay
{
private:
    ApraEGLDisplay();

public:     
    ~ApraEGLDisplay();
    static EGLDisplay getEGLDisplay();

private:
    EGLDisplay mEGLDisplay;
    static boost::shared_ptr<ApraEGLDisplay> instance;
};