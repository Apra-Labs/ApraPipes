#pragma once

#include "Module.h"

class FramesMuxerProps : public ModuleProps
{
public:
	enum Strategy {
		ALL_OR_NONE,
		MAX_DELAY_ANY
	};
public:
	FramesMuxerProps() : ModuleProps() 
	{
		maxDelay = 30;
		strategy = ALL_OR_NONE;
		fIndexStrategyType = FIndexStrategy::FIndexStrategyType::NONE;
	}

	int maxDelay; // Difference between current frame and first frame in the queue
	Strategy strategy;
};

class FramesMuxerStrategy;

class FramesMuxer : public Module {
public:

	FramesMuxer(FramesMuxerProps _props=FramesMuxerProps());
	virtual ~FramesMuxer() {}

	virtual bool init();
	virtual bool term();

protected:	
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();	
	bool validateInputOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);

private:		
	boost::shared_ptr<FramesMuxerStrategy> mDetail;
};



