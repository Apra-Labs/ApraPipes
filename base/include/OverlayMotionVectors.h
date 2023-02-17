#pragma once
#include <string>
#include "Module.h"

using namespace std;

class OverlayMotionVectorProps : public ModuleProps
{
public:
	OverlayMotionVectorProps()
	{
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize();
	}
};

class  OverlayMotionVector : public Module
{
public:
	OverlayMotionVector(OverlayMotionVectorProps _props);
	virtual ~OverlayMotionVector() {};
	bool init();
	bool term();
protected:
	bool process(frame_container& frame);
	bool validateInputPins();
	bool processSOS(frame_sp& frame);
	bool shouldTriggerSOS();
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	std::string mOutputPinId;
};
