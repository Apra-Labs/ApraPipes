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
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();

private:		
	void setMetadata(framemetadata_sp& metadata);
	int mFrameType;
	ImageEncoderCVProps props;
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	
};
