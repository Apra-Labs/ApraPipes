#pragma once
#include <string>
#include "Module.h"

using namespace std;

class OverlayModuleProps : public ModuleProps
{
public:
	enum shapeType
	{
		LINE = 0,
		CIRCLE,
		COMPOSITE
	};

	OverlayModuleProps() 
	{
	}

	OverlayModuleProps(shapeType _shapeType) : mShapeType(_shapeType)
	{
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize();
	}

	shapeType mShapeType;
};

class OverlayCommand;

class  OverlayModule : public Module
{
public:
	OverlayModule(OverlayModuleProps _props);
	virtual ~OverlayModule() {};
	bool init();
	bool term();
	boost::shared_ptr<OverlayCommand> mDetail;
protected:
	bool process(frame_container& frame);
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool validateInputPins();
	bool validateOutputPins();
	bool processSOS(frame_sp& frame);
	bool shouldTriggerSOS();
	
private:
	std::string mOutputPinId;
};
