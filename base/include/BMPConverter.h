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
	bool init() override;
	bool term() override;

	void getImageSize(int& width, int& height);

protected:
	bool process(frame_container& frames) override;
	bool processSOS(frame_sp& frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	bool shouldTriggerSOS() override;
	bool processEOS(string& pinId) override;

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;

	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
};
