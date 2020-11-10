#pragma once

#include "Module.h"

class JPEGDecoderL4TMProps : public ModuleProps
{
public:
	JPEGDecoderL4TMProps() : ModuleProps()
	{
		
	}
};

class JPEGDecoderL4TM : public Module
{

public:
	JPEGDecoderL4TM(JPEGDecoderL4TMProps _props=JPEGDecoderL4TMProps());
	virtual ~JPEGDecoderL4TM();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;	
};