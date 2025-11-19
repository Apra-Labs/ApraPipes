
#pragma once

#include "Module.h"

class RotateCVProps : public ModuleProps
{
public:
	RotateCVProps(double _angle)
	{
		angle = _angle;
	}

	double angle;
};

class RotateCV : public Module
{

public:
	RotateCV(RotateCVProps props);
	virtual ~RotateCV();
	bool init();
	bool term();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};