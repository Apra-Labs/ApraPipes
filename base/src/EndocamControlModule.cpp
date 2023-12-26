#include <stdafx.h>
#include <boost/filesystem.hpp>
#include "EndocamControlModule.h"
#include "Mp4WriterSink.h"
#include "Module.h"
#include "Command.h"

EndocamControlModule::EndocamControlModule(EndocamControlModuleProps _props)
    :AbsControlModule(_props)
{
    
}

EndocamControlModule::~EndocamControlModule() {}

bool EndocamControlModule::validateInputPins()
{
    return true;
}

bool EndocamControlModule::validateOutputPins()
{
    return true;
}

bool EndocamControlModule::validateInputOutputPins()
{
    return true;
}

bool EndocamControlModule::handleCommand(Command::CommandType type, frame_sp& frame)
{


    return Module::handleCommand(type, frame);
}

bool EndocamControlModule::init()
{
    if (!Module::init())
    {
        return false;
    }
    return true;
}

bool EndocamControlModule::term()
{
    return Module::term();
}

bool EndocamControlModule::validateModuleRoles() // planning to have a map of module and string instead of other way around
{
    for (int i = 0; i < pipelineModules.size(); i++) // Iterating Over Pipeline modules
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
