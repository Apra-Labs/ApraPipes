#pragma once
#include "Module.h"

class ImageEncoderCVProps : public ModuleProps
{
public:
	ImageEncoderCVProps() : ModuleProps() 
	{
		
	}

};

class ImageEncoderCV : public Module
{

public:
	ImageEncoderCV(ImageEncoderCVProps _props);
	virtual ~ImageEncoderCV();
	bool init() override;
	bool term() override;
protected:
	bool process(frame_container& frames) override;
	bool processSOS(frame_sp& frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;

private:
	int mFrameType;
	ImageEncoderCVProps props;
	class Detail;
	std::shared_ptr<Detail> mDetail;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	
};
