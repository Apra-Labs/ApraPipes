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

	bool init() override;
	bool term() override;

protected:	
	bool process(frame_container& frames) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp& metadata, std::string_view pinId) override;

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};



