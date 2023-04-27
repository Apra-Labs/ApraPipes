#pragma once
#include <string>
#include "Module.h"

using namespace std;

class OverlayModuleProps : public ModuleProps
{
public:
	OverlayModuleProps() : ModuleProps() {}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize();
	}
};

class OverlayCommand;

class  OverlayModule : public Module
{
public:
	OverlayModule(OverlayModuleProps _props);
	virtual ~OverlayModule() {};
	bool init();
	bool term();
protected:
	bool process(frame_container& frame);
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();

private:
	std::string mOutputPinId;
};
