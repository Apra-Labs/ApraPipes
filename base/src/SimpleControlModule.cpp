#include "SimpleControlModule.h"

void SimpleControlModule::sendEOS()
{
	return Module::sendEOS();
}

void SimpleControlModule::sendEOS(frame_sp& frame)
{
	return Module::sendEOS(frame);
}

void SimpleControlModule::sendEOPFrame()
{
	return Module::sendEoPFrame();
}

std::string SimpleControlModule::printStatus()
{
	return AbsControlModule::printStatus();
}

// Right Now, Just Logging But Can be used to Do bunch of other things 
void SimpleControlModule::handleError(const APErrorObject &error)
{
	LOG_ERROR << "Error in module " << error.getModuleName() << "Module Id"
			  << error.getModuleId() << " (Code " << error.getErrorCode()
			  << "): " << error.getErrorMessage();
}

void SimpleControlModule::handleHealthCallback(const APHealthObject &healthObj)
{
	LOG_ERROR << "Health Callback from  module " << healthObj.getModuleId();
	if (!healthCallbackExtention.empty())
	{
		LOG_INFO << "Calling Health Callback Extention...";
		healthCallbackExtention(&healthObj,1);
	}
}
