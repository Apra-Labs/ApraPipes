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
	bool init() override;
	bool term() override;

protected:
	bool process(frame_container& frames) override;
	bool processSOS(frame_sp& frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};