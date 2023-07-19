#pragma once
#include <string>
#include "Module.h"

using namespace std;

class OverlayModuleDecProps : public ModuleProps
{
public:
	OverlayModuleDecProps() : ModuleProps() {}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize();
	}
};

class OverlayCommand;

class  OverlayModuleDec : public Module
{
public:
	OverlayModuleDec(OverlayModuleDecProps _props);
	virtual ~OverlayModuleDec() {};
	bool init();
	bool term();
protected:
	bool process(frame_container& frame);
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();

private:
	//RawImageMetadata* mOutputMetadata;
	std::string mOutputPinId;
};
