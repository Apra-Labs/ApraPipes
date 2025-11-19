#include <map>
#include <stdafx.h>
#include <memory>
#include "ValveModule.h"
#include "Module.h"
#include "Command.h"


/* The valve module takes no of frames as arguements (moduleProps) and enable module value.
The number of frames are passed to the next module then when enable module is set to false,
frames stop passing. The moduleProps can be changed by setProps or the number of frames can also be sent through the command allowFrames().  */


class ValveModule::Detail
{
public:
    Detail(ValveModuleProps& _props) : mProps(_props), mFramesSaved(0), enableFlow(false)
    { 
    }

    ~Detail()
    { 
    }

    void resetFlowParams(int numOfFrames)
    {
        mProps.noOfFramesToCapture = numOfFrames;
        mFramesSaved = 0;
        enableFlow = true;
        pinMap.clear();
    }

    void setProps(ValveModuleProps _props)
    {
        mProps = _props;
    }

    void createPinMap(frame_container frames)
    {
        for (auto it = frames.begin(); it != frames.end(); it++)
        {
            if (pinMap.find(it->first) == pinMap.end())
            {
                auto itr = it;
                pinMap.insert(pair<string, int>(itr->first, mProps.noOfFramesToCapture));
            }
        }
    }

    bool processTask(frame_container& frames)
    {
        if (enableFlow)
        {
            bool reset = true;
            for (auto it = pinMap.begin(); it != pinMap.end(); it++)
            {
                if (it->second > 0)
                {
                    reset = false;
                }
            }
            if (reset)
            {
                enableFlow = false;
                pinMap.clear();
                return false;
            }
            for (auto it = frames.begin(); it != frames.end(); it++)
            {
                if (pinMap[it->first] == 0)
                {
                    auto itr = it;
                    frames.erase(itr->first);
                }
                else
                {
                    pinMap[it->first]--;
                }
            }
            return true;
        }
        return false;
    }

public:
    uint64 mFramesSaved = 0;
    bool enableFlow = false;
    ValveModuleProps mProps;
    std::map<string, int> pinMap;
};



ValveModule::ValveModule(ValveModuleProps _props)
    :Module(TRANSFORM, "ValveModule", _props)
{
    mDetail.reset(new Detail(_props));
}

ValveModule::~ValveModule() {}

bool ValveModule::validateInputPins()
{   
    return true;
}

bool ValveModule::validateOutputPins()
{
    return true;
}

bool ValveModule::validateInputOutputPins()
{
    return true;
}

// default - open, sieve is disabled - feedback false
bool ValveModule::setNext(std::shared_ptr<Module> next, bool open, bool sieve)
{
    return Module::setNext(next, open, false, sieve);
}

void ValveModule::addInputPin(framemetadata_sp& metadata, string& pinId)
{
    Module::addInputPin(metadata, pinId);
}

bool ValveModule::handleCommand(Command::CommandType type, frame_sp& frame)
{
    if (type == Command::CommandType::ValvePassThrough)
    {
        ValvePassThroughCommand cmd;
        getCommand(cmd, frame);
        mDetail->resetFlowParams(cmd.numOfFrames);
        return true;
    }
    return Module::handleCommand(type, frame);
}

bool ValveModule::handlePropsChange(frame_sp& frame)
{
    ValveModuleProps props;
    bool ret = Module::handlePropsChange(frame, props);
    mDetail->setProps(props);
    ValvePassThroughCommand cmd;
    cmd.numOfFrames = mDetail->mProps.noOfFramesToCapture;
    queueCommand(cmd);
    return ret;
}

bool ValveModule::init() 
{
    if (!Module::init())
    {
        return false;
    }
    return true;
}

bool ValveModule::term()
{
    return Module::term();
}

ValveModuleProps ValveModule::getProps()
{
    fillProps(mDetail->mProps);
    return mDetail->mProps;
}

void ValveModule::setProps(ValveModuleProps& props)
{
    Module::addPropsToQueue(props);
}

bool ValveModule::process(frame_container& frames) 
{
    mDetail->createPinMap(frames);
    if(mDetail->mProps.noOfFramesToCapture == -1)
    {
        send(frames);
    }
    else if (mDetail->processTask(frames))
    {
        send(frames);
    }
    return true;
}

bool ValveModule::processSOS(frame_sp& frame)
{ 
    auto metadata = frame->getMetadata();
    setMetadata(metadata);
    return true;
}

void ValveModule::setMetadata(framemetadata_sp& metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
}

/* We can set the number of frames property by passing as 
arguement to allowFrames else module props value is taken if no arguements are passed. */

bool ValveModule::allowFrames(int numframes)
{
    ValvePassThroughCommand cmd;
    cmd.numOfFrames = numframes;
    return queueCommand(cmd);
}