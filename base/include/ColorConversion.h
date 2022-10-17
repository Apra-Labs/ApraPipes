#pragma once
#include "FrameMetadata.h"
#include "Module.h"
#include "AbsColorConversionFactory.h"

class ColorConversionProps : public ModuleProps
{
public:
	enum ConversionType
	{
		RGB_2_MONO = 0,
		BGR_2_MONO,
		BGR_2_RGB,
		RGB_2_BGR,
		BAYERBG8_2_MONO,
		RGB_2_YUV420PLANAR,
		YUV420PLANAR_2_RGB,
		BAYERBG8_2_RGB,
		BAYERGB8_2_RGB,
		BAYERRG8_2_RGB,
		BAYERGR8_2_RGB
	};
	ColorConversionProps(ConversionType _type) : ModuleProps()
	{
		type = _type;
	}
	ColorConversionProps() : ModuleProps()
	{}
	ConversionType type;
};

class ColorConversion : public Module
{

public:
	ColorConversion(ColorConversionProps _props);
	virtual ~ColorConversion();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void setConversionStrategy(framemetadata_sp inputMetadata, framemetadata_sp outputMetadata);
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	std::string addOutputPin(framemetadata_sp& metadata);

private:
	void setMetadata(framemetadata_sp& metadata);
	int mFrameType;
	ColorConversionProps mProps;
	boost::shared_ptr<DetailAbstract> mDetail;
	std::string mOutputPinId;
	uint16_t mWidth;
	uint16_t mHeight;
	framemetadata_sp mOutputMetadata;
};

