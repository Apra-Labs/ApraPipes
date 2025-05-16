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
		moduleWithRole = moduleRoles[role].lock();
	}
	catch (std::out_of_range)
	{
		LOG_ERROR << "no module with the role <" << role << "> registered with the control module.";
	}
	return moduleWithRole;
}

void AbsControlModule::registerHealthCallbackExtention(
	boost::function<void(const APHealthObject*, unsigned short)> callbackFunction)
{
	healthCallbackExtention = callbackFunction;
};

void AbsControlModule::handleHealthCallback(const APHealthObject& healthObj)
{
	LOG_INFO << "Health Callback from  module " << healthObj.getModuleId();
	if (!healthCallbackExtention.empty())
	{
		LOG_INFO << "Calling the registered Health Callback Extention...";
		healthCallbackExtention(&healthObj, 1);
	}
}

std::vector<std::string> AbsControlModule::serializeControlModule()
{
	std::string spacedLineFmt = "\t-->";
	std::vector<std::string> status;
	status.push_back("Module <" + this->getId() + "> \n");
	status.push_back("Enrolled Modules \n");
	for (auto it : moduleRoles)
	{
		status.push_back("module <" + it.second.lock()->getId() + "> role <" + it.first + ">\n");
		std::string cbStatus = "registered for...\n";
		if (it.second.lock()->getProps().enableHealthCallBack)
		{
			cbStatus += spacedLineFmt + "health callbacks \n";
		}
		cbStatus += spacedLineFmt + "error callbacks \n";
		status.push_back(spacedLineFmt + cbStatus);
	}
	return status;
}

std::string AbsControlModule::printStatus()
{ 
	auto ser = boost::algorithm::join(serializeControlModule(), "|");
	LOG_INFO << ser;
	return ser;
}