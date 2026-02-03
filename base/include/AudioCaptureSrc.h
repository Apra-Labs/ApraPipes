#pragma once

#include "Module.h"

class AudioCaptureSrcProps : public ModuleProps
{
public:
    AudioCaptureSrcProps(
        int _sampleRate,
        int _channels,
        int _audioInputDeviceIndex) : sampleRate(_sampleRate),
                                channels(_channels),
                                audioInputDeviceIndex(_audioInputDeviceIndex)
    {
    }
    int sampleRate;
    int channels;
    int audioInputDeviceIndex; // starts from 0 to no. of available devices on users system. 

};

class AudioCaptureSrc  : public Module
{
public:
    AudioCaptureSrc(AudioCaptureSrcProps _props);
    virtual ~AudioCaptureSrc() {}
    virtual bool init();
    virtual bool term();
    void setProps(AudioCaptureSrcProps &props);
    AudioCaptureSrcProps getProps();

protected:
    bool validateOutputPins();
    bool produce();
    bool handlePropsChange(frame_sp &frame);

private:
    class Detail;
    boost::shared_ptr<Detail> mDetail;
    std::string mOutputPinId;
};