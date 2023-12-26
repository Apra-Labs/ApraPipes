#include "stdafx.h"
#include <boost/filesystem.hpp>
#include "AbsControlModule.h"
#include "Module.h"
#include "Command.h"

class AbsControlModule::Detail
{
public:
    Detail(AbsControlModuleProps& _props) : mProps(_props)
    {
    }

    ~Detail()
    {
    }
    AbsControlModuleProps mProps;
};

AbsControlModule::AbsControlModule(AbsControlModuleProps _props)
    :Module(TRANSFORM, "NVRControlModule", _props)
{
    mDetail.reset(new Detail(_props));
}
AbsControlModule::~AbsControlModule() {}

bool AbsControlModule::validateInputPins()
{
    return true;
}

bool AbsControlModule::validateOutputPins()
{
    return true;
}

bool AbsControlModule::validateInputOutputPins()
{
    return true;
}

void AbsControlModule::addInputPin(framemetadata_sp& metadata, string& pinId)
{
    Module::addInputPin(metadata, pinId);
}

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

AbsControlModuleProps AbsControlModule::getProps()
{
    fillProps(mDetail->mProps);
    return mDetail->mProps;
}

void AbsControlModule::setProps(AbsControlModuleProps& props)
{
    Module::addPropsToQueue(props);
}

bool AbsControlModule::process(frame_container& frames)
{
    return true;
}

bool AbsControlModule::enrollModule(std::string role, boost::shared_ptr<Module> module)
{
    moduleRoles[role] = module;
    return true;
}

boost::shared_ptr<Module> AbsControlModule::getModuleofRole(std::string role)
{
    return moduleRoles[role];
}