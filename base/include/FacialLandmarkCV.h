#pragma once

#include "Module.h"

class FacialLandmarkCVProps : public ModuleProps
{
public:
	FacialLandmarkCVProps()
	{	
	}

	// if any props set it 
};

class FacialLandmarkCV : public Module
{

public:
	FacialLandmarkCV(FacialLandmarkCVProps props);
	virtual ~FacialLandmarkCV();
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
	boost::shared_ptr<Detail> mDetail;
};