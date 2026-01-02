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