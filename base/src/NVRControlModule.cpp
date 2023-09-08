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

    if (type == Command::CommandType::NVRCommandView)
    {
        NVRCommandView cmd;
        getCommand(cmd, frame);
        for (int i = 0; i < pipelineModules.size(); i++)
        {
            if (pipelineModules[i] == getModuleofRole("Renderer_1")) // Logic for detecting modules to add
            {
                auto myId = pipelineModules[i]->getId();
                pipelineModules[i]->queueCommand(cmd);
            }
        }
        // if(cmd.doView == false)
        // {           
        //     LOG_ERROR<<" PAUSED COMMAND SENT TO MMQ - paused start is !!"<<pausedTS;
        //     LOG_ERROR<<" NEW Time end is  !!"<<pausedTS + 100;

        //     MultimediaQueueXformCommand cmd;
        //     cmd.startTime = pausedTS; //1694083685741
        //     cmd.endTime = pausedTS + 100;
        //     //cmd.endTime = 1693723948000;
        //     for (int i = 0; i < pipelineModules.size(); i++)
        //     {
        //         if (pipelineModules[i] == getModuleofRole("MultimediaQueue")) // Sending command to multimediaQueue
        //         {
        //             auto myid = pipelineModules[i]->getId();
        //             pipelineModules[i]->queueCommand(cmd);
        //         }
        //     }

        // }
        return true;
    }
    if (type == Command::CommandType::NVRCommandExportView)
    {
        NVRCommandExportView cmd;
        getCommand(cmd, frame);
        givenStart = cmd.startViewTS;
        givenStop = cmd.stopViewTS;
        if(pausedTS < firstMMQtimestamp)
        {
            LOG_ERROR<<" The seeked start time is in disk!!";
            Mp4SeekCommand command;
            command.seekStartTS = pausedTS;
            command.forceReopen = false;
            for (int i = 0; i < pipelineModules.size(); i++)
            {
                if (pipelineModules[i] == getModuleofRole("Reader_1")) // Sending command to reader
                {
                    auto myId = pipelineModules[i]->getId();
                    pipelineModules[i]->queueCommand(command);
                    pipelineModules[i]->play(true);
                    return true;
                }
            }
        }
        else
        {
            LOG_ERROR<<" The seeked start time is in MULTIMEDIA-QUEUE!!";
            MultimediaQueueXformCommand cmd;
            cmd.startTime = pausedTS;
            cmd.endTime = 1694340826000;
            for (int i = 0; i < pipelineModules.size(); i++)
            {
                if (pipelineModules[i] == getModuleofRole("MultimediaQueue")) // Sending command to multimediaQueue
                {
                    auto myid = pipelineModules[i]->getId();
                    pipelineModules[i]->queueCommand(cmd);
                }
            }
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

bool NVRControlModule::nvrExportView(uint64_t ts, uint64_t te)
{
    NVRCommandExportView cmd;
    cmd.startViewTS = ts;
    cmd.stopViewTS = te;
    return queueCommand(cmd);
}

bool NVRControlModule::nvrView(bool view)
{
    NVRCommandView cmd;
    cmd.doView = view;
    return queueCommand(cmd);
}