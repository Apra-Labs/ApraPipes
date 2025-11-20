#pragma once

#include "Module.h"

class JPEGEncoderL4TMProps : public ModuleProps
{
public:
	JPEGEncoderL4TMProps()
	{
		quality = 90;
		scale = 1; 
	}

	unsigned short quality;
	double scale;
};

class JPEGEncoderL4TM : public Module
{

public:
	JPEGEncoderL4TM(JPEGEncoderL4TMProps props = JPEGEncoderL4TMProps());
	virtual ~JPEGEncoderL4TM();
	bool init() override;
	bool term() override;

protected:
	bool process(frame_container& frames) override;
	bool processSOS(frame_sp& frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	bool shouldTriggerSOS() override;
	bool processEOS(string& pinId) override;

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};
