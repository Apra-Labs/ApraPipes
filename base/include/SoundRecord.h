#pragma once

#include "Module.h"

class SoundRecordProps : public ModuleProps
{
public:
    SoundRecordProps(int _sampleRate, int _channel, int _byteDepth, int _device, int _proccessingRate) : sampleRate(_sampleRate), channel(_channel), byteDepth(_byteDepth), device(_device), proccessingRate(_proccessingRate)
    {
    }
    int sampleRate;
    int channel;
    int byteDepth;
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

protected:
    bool validateOutputPins();
    bool produce();

private:
    class Detail;
    boost::shared_ptr<Detail> mDetail;
    std::string mOutputPinId;
};