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

	AbsControlModuleProps mProps;
};

AbsControlModule::AbsControlModule(AbsControlModuleProps _props)
	:Module(CONTROL, "AbsControlModule", _props)
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
	// Commands are already processed by the time we reach here.
	return true;
}

bool AbsControlModule::enrollModule(std::string role, boost::shared_ptr<Module> module)
{
	if (moduleRoles.find(role) != moduleRoles.end())
	{
		LOG_ERROR << "Role already registered with the control module.";
        return false;
	}

	moduleRoles[role] = module;

	// NOTE: If you want error callback and health callback to work with a module, registering it with control is mandatory.
	module->registerErrorCallback(
		[this](const APErrorObject& error) { handleError(error); });
	
	if (module->getProps().enableHealthCallBack)
	{
		module->registerHealthCallback(
			[this](const APHealthObject& message) { handleHealthCallback(message); });
	}
	
	return true;
}

boost::shared_ptr<Module> AbsControlModule::getModuleofRole(std::string role)
{
	boost::shared_ptr<Module> moduleWithRole = nullptr;
	try
	{
		moduleWithRole = moduleRoles[role];
	}
	catch (std::out_of_range)
	{
		LOG_ERROR << "no module with the role <" << role << "> registered with the control module.";
	}
	return moduleWithRole;
}