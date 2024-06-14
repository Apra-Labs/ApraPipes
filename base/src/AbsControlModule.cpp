#include "stdafx.h"
#include <boost/filesystem.hpp>
#include "AbsControlModule.h"
#include "Module.h"
#include "Command.h"
#include "PipeLine.h"

class AbsControlModule::Detail
{
public:
    Detail(AbsControlModuleProps& _props) : mProps(_props)
    {
    }

    ~Detail()
    {
    }

    std::string getPipelineRole(std::string pName, std::string role)
    {
        return pName + "_" + role;
    }

    AbsControlModuleProps mProps;
};

AbsControlModule::AbsControlModule(AbsControlModuleProps _props)
    :Module(TRANSFORM, "AbsControlModule", _props)
{
    mDetail.reset(new Detail(_props));
}
AbsControlModule::~AbsControlModule() {}

bool AbsControlModule::handleCommand(Command::CommandType type, frame_sp& frame)
{
    return true;
}

bool AbsControlModule::handlePropsChange(frame_sp& frame)
{
    return true;
}

bool AbsControlModule::init()
{
    if (!Module::init())
    {
        return false;
    }
    return true;
}

bool AbsControlModule::term()
{
    return Module::term();
}

bool AbsControlModule::process(frame_container& frames)
{
    return true;
}

std::string AbsControlModule::enrollModule(boost::shared_ptr<PipeLine> p, std::string role, boost::shared_ptr<Module> module)
{
    std::string pipelineRole = mDetail->getPipelineRole(p->getName(), role);
    if (moduleRoles.find(pipelineRole) != moduleRoles.end())
    {
        std::string errMsg = "Enrollment Failed: This role <" + role + "> already registered with the Module <" + moduleRoles[pipelineRole]->getName() + "> in PipeLine <" + p->getName() + ">";
        LOG_ERROR << errMsg;
        throw AIPException(MODULE_ENROLLMENT_FAILED, errMsg);
    }
    moduleRoles[pipelineRole] = module;
    return pipelineRole;
}

std::pair<bool, boost::shared_ptr<Module>> AbsControlModule::getModuleofRole(PipeLine p, std::string role)
{
    std::string pipelineRole = mDetail->getPipelineRole(p.getName(), role);
    if (moduleRoles.find(pipelineRole) == moduleRoles.end())
    {
        return std::make_pair<bool, boost::shared_ptr<Module>>(false, nullptr);
    }
    std::pair<bool, boost::shared_ptr<Module>> res(true, moduleRoles[pipelineRole]); 
    return res;
}