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
	bool init() override;
	bool term() override;
protected:
	bool process(frame_container& frame) override;
	void addInputPin(framemetadata_sp& metadata, std::string_view pinId) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	bool shouldTriggerSOS() override;

private:
	std::string mOutputPinId;
};
