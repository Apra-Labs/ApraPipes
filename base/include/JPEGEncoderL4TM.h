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
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
	bool processEOS(string& pinId);

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};
