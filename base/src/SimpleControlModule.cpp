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

// Right Now, Just Logging But Can be used to Do bunch of other things 
void SimpleControlModule::handleError(const APErrorObject &error)
{
	LOG_ERROR << "Error in module " << error.getModuleName() << "Module Id"
			  << error.getModuleId() << " (Code " << static_cast<int>(error.getErrorCode())
			  << "): " << error.getErrorMessage();
}

void SimpleControlModule::handleHealthCallback(const APHealthObject &healthObj)
{
	LOG_ERROR << "Health Callback from  module " << healthObj.getModuleId();
}
