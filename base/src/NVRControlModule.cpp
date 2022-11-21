#include <stdafx.h>
#include <boost/filesystem.hpp>
#include "NVRControlModule.h"
#include "Mp4WriterSink.h"
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
    void setProps(NVRControlModuleProps _props)
    {
        mProps = _props;
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

bool NVRControlModule::handleCommand(Command::CommandType type, frame_sp& frame)
{
    if (type == Command::CommandType::NVRCommandRecord)
    {
        NVRCommandRecord cmd;
        getCommand(cmd, frame);
        for (int i = 0; i < pipelineModules.size(); i++)
        {
            if (pipelineModules[i] == getModuleofRole("Writer-1")) // Logic for detecting modules to add
            {
                pipelineModules[i]->queueCommand(cmd);
            }
        }
        return true;
    }
    if (type == Command::CommandType::NVRCommandExport)
    {
        NVRCommandExport cmd;
        getCommand(cmd, frame);
        uint64_t testStart = firstMMQtimestamp + 1000;
        uint64_t testStop = firstMMQtimestamp + 5000;
        cmd.startExportTS = testStart;
        cmd.stopExportTS = testStop;
        for (int i = 0; i < pipelineModules.size(); i++)
        {
            if (pipelineModules[i] == getModuleofRole("MultimediaQueue")) // Logic for detecting modules to add
            {
                auto myId = pipelineModules[i]->getId();
                pipelineModules[i]->queueCommand(cmd);
            }
        }
        return true;
    }
    if (type == Command::CommandType::NVRCommandView)
    {
        NVRCommandView cmd;
        getCommand(cmd, frame);
        for (int i = 0; i < pipelineModules.size(); i++)
        {
            if (pipelineModules[i] == getModuleofRole("Renderer")) // Logic for detecting modules to add
            {
                auto myId = pipelineModules[i]->getId();
                pipelineModules[i]->queueCommand(cmd);
            }
        }
        return true;
    }
    if (type == Command::CommandType::MP4WriterLastTS)
    {
        MP4WriterLastTS cmd;
        getCommand(cmd, frame);
        auto tempMod = getModuleofRole("Writer-1");
        if (cmd.moduleId == tempMod->getId())
        {
            mp4lastWrittenTS = cmd.lastWrittenTimeStamp; //We can save the last timestamp for a single writer using role
        }
        auto tempMod2 = getModuleofRole("Writer-2");
        if (cmd.moduleId == tempMod2->getId())
        {
            mp4_2_lastWrittenTS = cmd.lastWrittenTimeStamp; //We can save the last timestamp for a single writer using role
        }
        return true;
    }
    if (type == Command::CommandType::MMQtimestamps)
    {
        MMQtimestamps cmd;
        getCommand(cmd, frame);
        firstMMQtimestamp = cmd.firstTimeStamp;
        lastMMQtimestamp = cmd.lastTimeStamp;
        return true;
    }

    return Module::handleCommand(type, frame);
}

bool NVRControlModule::handlePropsChange(frame_sp& frame)
{
    NVRControlModuleProps props(mDetail->mProps);
    auto ret = Module::handlePropsChange(frame, props);
    mDetail->setProps(props);
    return ret;
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

bool NVRControlModule::validateModuleRoles()
{
    for (int i = 0; i < pipelineModules.size(); i++)
    {
        bool modPresent = false;
        for (auto it = moduleRoles.begin(); it != moduleRoles.end(); it++)
        {
            if (pipelineModules[i] == it->second)
            {
                modPresent = true;
            }
        }
        if (!modPresent)
        {
            LOG_ERROR << "Modules and roles validation failed!!";
        }
    }
    return true;
}

bool NVRControlModule::nvrRecord(bool record)
{
    NVRCommandRecord cmd;
    cmd.doRecording = record;
    return queueCommand(cmd);
}

bool NVRControlModule::nvrExport(uint64_t ts, uint64_t te)
{
    NVRCommandExport cmd;
    cmd.startExportTS = ts;
    cmd.stopExportTS = te;
    return queueCommand(cmd);
}

bool NVRControlModule::nvrView(bool view)
{
    NVRCommandView cmd;
    cmd.doView = view;
    return queueCommand(cmd);
}