#include "AudioCaptureSrc.h"
#include "Module.h"
#include "Logger.h"
#include "SFML/Audio.hpp"
#include "SFML/Audio/SoundRecorder.hpp"
#include "SFML/Audio/SoundBuffer.hpp"

class AudioCaptureSrc::Detail
{

public:
    Detail(
        AudioCaptureSrcProps _props,
        std::function<bool(const std::int16_t *samples,
                           std::size_t sampleCount)> _mMakeFrame) : mRecorder(_mMakeFrame, _props.channels), 
                                                                    mProps(_props)
    {
    }
    ~Detail() {}

    void setProps(AudioCaptureSrcProps &props)
    {
        mProps = props;
    }

    bool init()
    {
        if (!ApraRecorder::isAvailable())
        {
            LOG_ERROR << "No Audio device available";
            return false;
        }

        std::vector<std::string> availableDevices = sf::SoundRecorder::getAvailableDevices();
        auto success = mRecorder.setDevice(availableDevices[mProps.audioInputDeviceIndex]);
        LOG_INFO << "recorder set device: " << success;
        mRecorder.setChannelCount(mProps.channels); //set channel count
        mRecorder.start(mProps.sampleRate);
        return true;
    }
    bool stopRecording()
    {
        mRecorder.stop();
        return true;
    }

private:
    class ApraRecorder : public sf::SoundRecorder
    {
        friend class AudioCaptureSrc;

        int channelCount;
        std::function<bool(const std::int16_t *samples, std::size_t sampleCount)> mMakeFrame;

    public:
      
        ApraRecorder(std::function<bool(const std::int16_t *samples, std::size_t sampleCount)> _mMakeFrame, int _channelCount)
        {
            mMakeFrame = _mMakeFrame;
            channelCount = _channelCount;
        }

        virtual bool onStart()
        {
            return true;
        }

        virtual bool onProcessSamples(const std::int16_t *samples, std::size_t sampleCount)
        {
            return mMakeFrame(samples, sampleCount);
        }
    };

public:
    ApraRecorder mRecorder;
    AudioCaptureSrcProps mProps;
    std::string mOutputRawAudio;
};

AudioCaptureSrc::AudioCaptureSrc(AudioCaptureSrcProps _props) : Module(SOURCE, "AudioCaptureSrc", _props)
{
    mDetail.reset(new Detail(_props, [&](const std::int16_t *samples, std::size_t sampleCount) -> bool
                             {
                                 auto outFrame = makeFrame(sampleCount * 2); // Size of Int16 is 2 byte
                                 frame_container frames;
                                 memcpy(outFrame->data(), samples, outFrame->size());
                                 frames.insert(make_pair(mOutputPinId, outFrame));
                                 send(frames);
                                 return true;
                             }));
    auto mOutputRawAudio = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::AUDIO));
    mOutputPinId = addOutputPin(mOutputRawAudio);
}
bool AudioCaptureSrc::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPin size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstOutputMetadata();
    FrameMetadata::FrameType frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::FrameType::AUDIO)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPin input frameType is expected to be Audio. Actual<" << frameType << ">";
        return false;
    }
    return true;
}
AudioCaptureSrcProps AudioCaptureSrc::getProps()
{
    fillProps(mDetail->mProps);
    return mDetail->mProps; 
}

void AudioCaptureSrc::setProps(AudioCaptureSrcProps &props)
{
    Module::addPropsToQueue(props);
}

bool AudioCaptureSrc::handlePropsChange(frame_sp &frame)
{
    bool ret = Module::handlePropsChange(frame, mDetail->mProps);
    mDetail->setProps(mDetail->mProps);
    return ret;
}

bool AudioCaptureSrc::produce()
{
    return true;
}

bool AudioCaptureSrc::init()
{
    return Module::init() && mDetail->init();
}

bool AudioCaptureSrc::term()
{
    mDetail->stopRecording();
    return Module::term();
}
