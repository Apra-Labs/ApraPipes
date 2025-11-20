#pragma once

#include "Module.h"

class SplitProps : public ModuleProps
{
public:
	SplitProps() : ModuleProps() 
	{
        number = 2;
	}

    uint32_t number;
};

class Split : public Module {
public:

	Split(SplitProps _props=SplitProps());
	virtual ~Split() {}

	bool init() override;
	bool term() override;

protected:
	bool process(frame_container& frames) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp& metadata, string& pinId) override;

private:	
    uint32_t mNumber;
    uint32_t mCurrentIndex;
	uint32_t mFIndex2;
    std::vector<std::string> mPinIds;
};



