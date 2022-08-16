#include "stdafx.h"
#include <boost/filesystem.hpp>
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
    }

    void setProps(ValveModuleProps _props)
    {
        mProps = _props;
    }

    bool processTask()
    {   
        if (mFramesSaved < mProps.noOfFramesToCapture && enableFlow)
        {
            mFramesSaved++;
            if (mFramesSaved == mProps.noOfFramesToCapture)
            {
                enableFlow = false;
            }
            return true;
        }
        return false;
    }

public:
    uint64 mFramesSaved = 0;
    bool enableFlow = false;
    ValveModuleProps mProps;
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
    if(mDetail->mProps.noOfFramesToCapture == -1)
    {
        send(frames);
    }
    else if (mDetail->processTask())
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
arguement to resetFlowParams else module props value is taken */
bool ValveModule::allowFrames(int numframes)
{
    ValvePassThroughCommand cmd;
    cmd.numOfFrames = numframes;
    return queueCommand(cmd);
}

bool ValveModule::allowFrames()
{
    ValvePassThroughCommand cmd;
    cmd.numOfFrames = mDetail->mProps.noOfFramesToCapture;
    return queueCommand(cmd);
}