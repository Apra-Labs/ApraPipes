#pragma once

#include "Module.h"

class BMPConverterProps : public ModuleProps
{
public:
	BMPConverterProps()
	{
	
	}
};

class BMPConverter : public Module
{

public:
	BMPConverter(BMPConverterProps _props);
	virtual ~BMPConverter();
	bool init();
	bool term();

	void getImageSize(int& width, int& height);

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

	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
};
