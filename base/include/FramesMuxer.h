#pragma once

#include "Module.h"

class FramesMuxerProps : public ModuleProps
{
public:
	enum Strategy {
		ALL_OR_NONE,
		MAX_DELAY_ANY,
		MAX_TIMESTAMP_DELAY
	};

public:
	FramesMuxerProps() : ModuleProps()
	{
		maxTsDelayInMS = 16.67;
		maxDelay = 30;
		strategy = ALL_OR_NONE;
		fIndexStrategyType = FIndexStrategy::FIndexStrategyType::NONE;
	}

	int maxDelay; // Difference between current frame and first frame in the queue
	Strategy strategy;
	double maxTsDelayInMS; // Max TimeStampDelay in Milli Seconds
};

class FramesMuxerStrategy;

class FramesMuxer : public Module {
public:

	FramesMuxer(FramesMuxerProps _props=FramesMuxerProps());
	virtual ~FramesMuxer() {}

	bool init() override;
	bool term() override;

protected:
	bool process(frame_container& frames) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	bool validateInputOutputPins() override;
	void addInputPin(framemetadata_sp& metadata, string& pinId) override;

private:
	std::shared_ptr<FramesMuxerStrategy> mDetail;
};



