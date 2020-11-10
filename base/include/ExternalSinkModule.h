#pragma once

#include "Module.h"

class ExternalSinkModuleProps : public ModuleProps
{
public:
	ExternalSinkModuleProps() : ModuleProps() {}
};

class ExternalSinkModule : public Module
{
public:
	ExternalSinkModule(ExternalSinkModuleProps props=ExternalSinkModuleProps()) : Module(SINK, "ExternalSinkModule", props)
	{

	}

	virtual ~ExternalSinkModule() {}

	frame_container pop()
	{
		return Module::pop();
	}

	frame_container try_pop()
	{
		return Module::try_pop(); 
	}
protected:
	bool validateInputPins()
	{
		return true;
	}
};