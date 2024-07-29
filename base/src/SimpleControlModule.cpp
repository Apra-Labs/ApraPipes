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

