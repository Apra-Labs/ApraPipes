#pragma once

#include "Module.h"

class ImageDecoderCVProps : public ModuleProps
{
public:
	ImageDecoderCVProps() : ModuleProps() {}
};

class ImageDecoderCV : public Module
{

public:
	ImageDecoderCV(ImageDecoderCVProps _props=ImageDecoderCVProps());
	virtual ~ImageDecoderCV();
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