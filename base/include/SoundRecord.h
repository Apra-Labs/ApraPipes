#pragma once

#include "Module.h"

class SoundRecordProps : public ModuleProps
{
public:
    SoundRecordProps(
        int _sampleRate,
        int _channel,
        int _device,
        int _proccessingRate) : sampleRate(_sampleRate),
                                channel(_channel),
                                device(_device),
                                proccessingRate(_proccessingRate)
    {
    }
    int sampleRate;
    int channel;
    int device;
    int proccessingRate;
};

class SoundRecord : public Module
{
public:
    SoundRecord(SoundRecordProps _props);
    virtual ~SoundRecord() {}
    virtual bool init();
    virtual bool term();
    void setProps(SoundRecordProps &props);
    SoundRecordProps getProps();

protected:
    bool validateOutputPins();
    bool produce();
    bool handlePropsChange(frame_sp &frame);

private:
    class Detail;
    boost::shared_ptr<Detail> mDetail;
    std::string mOutputPinId;
};