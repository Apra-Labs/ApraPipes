#pragma once

#include <memory>
#include <opencv2/opencv.hpp> // this is added to address the following issue: https://github.com/opencv/opencv/issues/7113
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
    static std::shared_ptr<ApraEGLDisplay> instance;
};