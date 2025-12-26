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
    // Tests if DMA/eglImage creation actually works (not just display init)
    // On headless JetPack 5.x, display init succeeds but eglImage creation fails
    static bool isDMACapable();

private:
    EGLDisplay mEGLDisplay;
    bool mDMACapable;
    bool mDMACapabilityTested;
    static boost::shared_ptr<ApraEGLDisplay> instance;

    bool testDMACapability();
};