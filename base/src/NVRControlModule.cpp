#include "stdafx.h"
#include <boost/filesystem.hpp>
#include "NVRControlModule.h"
#include "Module.h"
#include "Command.h"

class NVRControlModule::Detail
{
public:
    Detail(NVRControlModuleProps& _props) : mProps(_props)
    {
    }

    ~Detail()
    {
    }
    NVRControlModuleProps mProps;
};

NVRControlModule::NVRControlModule(NVRControlModuleProps _props)
    :AbsControlModule(_props)
{
    mDetail.reset(new Detail(_props));
}

NVRControlModule::~NVRControlModule() {}

bool NVRControlModule::validateInputPins()
{
    return true;
}

bool NVRControlModule::validateOutputPins()
{
    return true;
}

bool NVRControlModule::validateInputOutputPins()
{
    return true;
}

void NVRControlModule::addInputPin(framemetadata_sp& metadata, string& pinId)
{
    Module::addInputPin(metadata, pinId);
}

bool NVRControlModule::handleCommand(Command::CommandType type, frame_sp& frame)
{
    if (type == Command::CommandType::NVRStartStop)
    {
        NVRCommandStartStop cmd;
        getCommand(cmd, frame);
        if (cmd.startRecording)
        {
            //In deque identify correct mp4 writer 
            //Call a mp4Writer->start writing command

        }
        else if (cmd.stopRecording)
        {
            //In deque identify correct mp4 writer and pass stop recording command
        }
    }
    if (type == Command::CommandType::NVRExport)
    {
        NVRCommandExport cmd;
        getCommand(cmd, frame);
        uint64_t startTime = cmd.startExport;
        uint64_t stopTime = cmd.stopExport;
    }
    return Module::handleCommand(type, frame);
}

bool NVRControlModule::handlePropsChange(frame_sp& frame)
{
    return true;
}

bool NVRControlModule::init()
{
    if (!Module::init())
    {
        return false;
    }
    return true;
}

bool NVRControlModule::term()
{
    return Module::term();
}

NVRControlModuleProps NVRControlModule::getProps()
{
    fillProps(mDetail->mProps);
    return mDetail->mProps;
}

void NVRControlModule::setProps(NVRControlModuleProps& props)
{
    Module::addPropsToQueue(props);
}

bool NVRControlModule::process(frame_container& frames)
{
    return true;
}

bool NVRControlModule::startRecord()
{
    NVRCommandStartStop cmd;
    cmd.startRecording = true;
    return queueCommand(cmd);
}

bool NVRControlModule::stopRecord()
{
    NVRCommandStartStop cmd;
    cmd.stopRecording = true;
    return queueCommand(cmd);
}

bool NVRControlModule::Export(uint64_t ts, uint64_t te)
{
    NVRCommandExport cmd;
    cmd.doExport = true;
    cmd.startExport = ts;
    cmd.stopExport = te;
    return queueCommand(cmd);
}