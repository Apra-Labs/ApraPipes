#include "stdafx.h"
#include <boost/filesystem.hpp>
#include "AbsControlModule.h"
#include "Module.h"
#include "Command.h"
#include "PipeLine.h"
#include "boost/algorithm/string/join.hpp"

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
	// Commands are already processed by the time we reach here.
	return true;
}

/**
 * @brief Enroll your module to use healthcallback, errorcallback and other control module functions
 * @param boost::shared_ptr<Module> the module to be registered
 * @param role unique string for role of the module
 * @return bool.
 */
bool AbsControlModule::enrollModule(std::string role, boost::shared_ptr<Module> module)
{
	if (moduleRoles.find(role) != moduleRoles.end())
	{
		LOG_ERROR << "Role already registered with the control module.";
        return false;
	}

	moduleRoles[role] = module;
	return true;
}

boost::shared_ptr<Module> AbsControlModule::getModuleofRole(std::string role)
{
    return moduleRoles[role];
}