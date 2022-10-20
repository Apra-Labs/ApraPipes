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

class NVRCommand
{

};
class RecordCommand : public NVRCommand
{
public:
    RecordCommand(bool _doRecord)
    {
        doRecord = _doRecord;
    }
    bool doRecord = false;
};
class ExportCommand : public NVRCommand
{
public:
    ExportCommand(uint64_t _startTime, uint64_t _stopTime)
    {
        startTime = _startTime;
        stopTime = _stopTime;
    }
    uint64_t startTime = 0;
    uint64_t stopTime = 0;
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
        NVRCommandRecord cmd;
        getCommand(cmd, frame);
        if (cmd.doRecording)
        {
            boost::shared_ptr<RecordCommand>Record;
            Record->doRecord = true;
        }
        else
        {
            boost::shared_ptr<RecordCommand>Record;
        }
    
    }
    if (type == Command::CommandType::NVRExport)
    {
        NVRCommandExport cmd;
        getCommand(cmd, frame);
        boost::shared_ptr<ExportCommand>Export;
        Export->startTime = cmd.startExport;
        Export->stopTime = cmd.stopExport;
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

bool NVRControlModule::Record(bool record)
{
    NVRCommandRecord cmd;
    if (record)
    {
        cmd.doRecording = true;
    }
    return queueCommand(cmd);
}


bool NVRControlModule::Export(uint64_t ts, uint64_t te)
{
    NVRCommandExport cmd;
    cmd.startExport = ts;
    cmd.stopExport = te;
    return queueCommand(cmd);
}