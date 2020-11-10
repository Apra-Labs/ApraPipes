#pragma once

#include "Module.h"

class MergeProps : public ModuleProps
{
public:
	MergeProps() : ModuleProps() 
	{
		maxDelay = 30;
	}

	uint32_t maxDelay; // Difference between current frame and first frame in the queue
};

class Merge : public Module {
public:

	Merge(MergeProps _props=MergeProps());
	virtual ~Merge() {}

	virtual bool init();
	virtual bool term();

protected:	
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();		
	void addInputPin(framemetadata_sp& metadata, string& pinId);

private:	
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};



