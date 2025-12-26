#pragma once

#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp> // this is added to address the following issue: https://github.com/opencv/opencv/issues/7113
#include "EGL/egl.h"

class ApraEGLDisplay
{
private:
    ApraEGLDisplay();

public:
    ~ApraEGLDisplay();
    static EGLDisplay getEGLDisplay();
    static bool isAvailable();

private:
    EGLDisplay mEGLDisplay;
    static boost::shared_ptr<ApraEGLDisplay> instance;
};