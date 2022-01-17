#include "SoundRecord.h"
#include "Module.h"
#include "Logger.h"
#include <SFML/System.hpp>
#include "SFML/Graphics.hpp"
#include <SFML/Audio.hpp>
#include <SFML/Audio/SoundRecorder.hpp>
#include <SFML/Audio/SoundBuffer.hpp>
#include <bits/stdc++.h>

std::vector<sf::Int16> samples_array;
int gProcessRate;

class SoundRecord::Detail
{

public:
    Detail(SoundRecordProps _props, std::function<bool(const sf::Int16 *samples, std::size_t sampleCount)> _mMakeFrame) : mRecorder(_mMakeFrame), mProps(_props)
    {
        gProcessRate = mProps.proccessingRate;
    }
    ~Detail() {}

    bool init()
    {
        if (!MyRecorder::isAvailable())
        {
            LOG_ERROR << "No Audio device available";
            return false;
        }

        std::vector<std::string> availableDevices = sf::SoundRecorder::getAvailableDevices();
        auto success = mRecorder.setDevice(availableDevices[mProps.device]);
        LOG_INFO << "recorder set device: " << success;
        mRecorder.start(mProps.sampleRate);
        return true;
    }
    bool stopRecording()
    {
        mRecorder.stop();
        return true;
    }

private:
    class MyRecorder : public sf::SoundRecorder
    {
        friend class SoundRecord;
        std::function<bool(const sf::Int16 *samples, std::size_t sampleCount)> mMakeFrame;

    public:
        MyRecorder(std::function<bool(const sf::Int16 *samples, std::size_t sampleCount)> _mMakeFrame)
        {
            mMakeFrame = _mMakeFrame;
        }

        virtual bool onStart()
        {
            setProcessingInterval(sf::milliseconds(gProcessRate));
            return true;
        }

        virtual bool onProcessSamples(const sf::Int16 *samples, std::size_t sampleCount)
        {
            return mMakeFrame(samples, sampleCount);
        }
    };

public:
    MyRecorder mRecorder;
    SoundRecordProps mProps;
    std::string mOutputRawAudio;
};

SoundRecord::SoundRecord(SoundRecordProps _props) : Module(SOURCE, "SoundRecord", _props)
{
    mDetail.reset(new Detail(_props, [&](const sf::Int16 *samples, std::size_t sampleCount) -> bool
                             {
                                 auto outFrame = makeFrame(sampleCount * 2);
                                 frame_container frames;
                                 memcpy(outFrame->data(), samples, outFrame->size());
                                 frames.insert(make_pair(mOutputPinId, outFrame));
                                 send(frames);
                                 return true;
                             }));
    auto mOutputRawAudio = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::AUDIO));
    mOutputPinId = addOutputPin(mOutputRawAudio);
}
bool SoundRecord::validateOutputPins()
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

bool SoundRecord::produce()
{
    return true;
}

bool SoundRecord::init()
{
    return Module::init() && mDetail->init();
}

bool SoundRecord::term()
{
    mDetail->stopRecording();
    return Module::term();
}
