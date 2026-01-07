#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

class AudioCaptureSrcProps : public ModuleProps
{
public:
    // Default constructor for declarative pipeline
    AudioCaptureSrcProps() : sampleRate(44100), channels(2), audioInputDeviceIndex(0), processingIntervalMS(100)
    {
    }

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

    // ============================================================
    // Property Binding for Declarative Pipeline
    // ============================================================
    template<typename PropsT>
    static void applyProperties(
        PropsT& props,
        const std::map<std::string, apra::ScalarPropertyValue>& values,
        std::vector<std::string>& missingRequired
    ) {
        apra::applyProp(props.sampleRate, "sampleRate", values, false, missingRequired);
        apra::applyProp(props.channels, "channels", values, false, missingRequired);
        apra::applyProp(props.audioInputDeviceIndex, "audioInputDeviceIndex", values, false, missingRequired);
        apra::applyProp(props.processingIntervalMS, "processingIntervalMS", values, false, missingRequired);
    }

    apra::ScalarPropertyValue getProperty(const std::string& propName) const {
        if (propName == "sampleRate") return static_cast<int64_t>(sampleRate);
        if (propName == "channels") return static_cast<int64_t>(channels);
        if (propName == "audioInputDeviceIndex") return static_cast<int64_t>(audioInputDeviceIndex);
        if (propName == "processingIntervalMS") return static_cast<int64_t>(processingIntervalMS);
        throw std::runtime_error("Unknown property: " + propName);
    }

    bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
        return false;  // All properties are static
    }

    static std::vector<std::string> dynamicPropertyNames() {
        return {};
    }
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