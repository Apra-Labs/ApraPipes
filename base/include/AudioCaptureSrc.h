#pragma once

#include "Module.h"

class AudioCaptureSrcProps : public ModuleProps
{
public:
    AudioCaptureSrcProps(
        int _sampleRate,
        int _channels,
        int _audioInputDeviceIndex,
        int _processingIntervalMS) : sampleRate(_sampleRate),
                                channels(_channels),
                                audioInputDeviceIndex(_audioInputDeviceIndex),
                                processingIntervalMS(_processingIntervalMS)
    {
    }
    int sampleRate;
    int channels;
    int audioInputDeviceIndex; // starts from 0 to no. of available devices on users system. 
    int processingIntervalMS;
};

class AudioCaptureSrc  : public Module
{
public:
    AudioCaptureSrc(AudioCaptureSrcProps _props);
    virtual ~AudioCaptureSrc() {}
    bool init() override;
    bool term() override;
    void setProps(AudioCaptureSrcProps &props);
    AudioCaptureSrcProps getProps();

protected:
    bool validateOutputPins() override;
    bool produce() override;
    bool handlePropsChange(frame_sp &frame) override;

private:
    class Detail;
    std::shared_ptr<Detail> mDetail;
    std::string mOutputPinId;
};