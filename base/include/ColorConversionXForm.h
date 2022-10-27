#pragma once
#include "FrameMetadata.h"
#include "Module.h"

class DetailAbstract;
class ColorConversionProps : public ModuleProps
{
public:
	enum ConversionType
	{
		RGB_TO_MONO = 0,
		BGR_TO_MONO,
		BGR_TO_RGB,
		RGB_TO_BGR,
		RGB_TO_YUV420PLANAR,
		YUV420PLANAR_TO_RGB,
		BAYERBG8_TO_MONO,
		BAYERBG8_TO_RGB,
		BAYERGB8_TO_RGB,
		BAYERRG8_TO_RGB,
		BAYERGR8_TO_RGB
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
	int mFrameType;
	ColorConversionProps mProps;
	boost::shared_ptr<DetailAbstract> mDetail;
	std::string mOutputPinId;
	uint16_t mWidth;
	uint16_t mHeight;
	framemetadata_sp mOutputMetadata;
	framemetadata_sp mInputMetadata;
	cv::Mat inpImg;
	cv::Mat outImg;
};

