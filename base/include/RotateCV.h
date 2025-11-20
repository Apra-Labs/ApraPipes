
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
	bool init() override;
	bool term() override;

protected:
	bool process(frame_container &frames) override;
	bool processSOS(frame_sp &frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp &metadata, string &pinId) override; // throws exception if validation fails
	bool shouldTriggerSOS() override;
	bool processEOS(string &pinId) override;

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};