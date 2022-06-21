#pragma once
 
#include "Module.h"
#include "V4L2CameraSourceHelper.h"
#include <memory>
 
class V4L2CameraSourceProps : public ModuleProps
{
public:
    V4L2CameraSourceProps(uint32_t _width, uint32_t _height) : ModuleProps(), width(_width), height(_height), cameraName("/dev/video0")
    {
    }
 
    V4L2CameraSourceProps(uint32_t _width, uint32_t _height, std::string _cameraName) : ModuleProps(), width(_width), height(_height), cameraName(_cameraName)
    {
    }
 
    uint32_t width;
    uint32_t height;
    std::string cameraName;
};
 
class V4L2CameraSource : public Module
{
public:
    V4L2CameraSource(V4L2CameraSourceProps props);
    virtual ~V4L2CameraSource();
    bool init();
    bool term();
 
protected:
    bool produce();
    bool validateOutputPins();
 
private:
    std::string mOutputPinId;
    std::shared_ptr<V4L2CameraSourceHelper> mHelper;
    V4L2CameraSourceProps mProps;
};
