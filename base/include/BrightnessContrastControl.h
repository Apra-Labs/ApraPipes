#pragma once

#include "Module.h"

class BrightnessContrastControlProps : public ModuleProps
{
public:
	BrightnessContrastControlProps(double _contrast, double _brightness, int _bitsPerPixel) : contrast(_contrast), brightness(_brightness), bitsPerPixel(_bitsPerPixel)
	{
	}
	double contrast;
	double brightness;
	int bitsPerPixel;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(int) + sizeof(double) * 2;
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &contrast &brightness &bitsPerPixel;
	}
};

class BrightnessContrastControl : public Module
{

public:
	BrightnessContrastControl(BrightnessContrastControlProps _props);
	virtual ~BrightnessContrastControl();
	bool init();
	bool term();
	void setProps(BrightnessContrastControlProps &props);
	BrightnessContrastControlProps getProps();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId);
	void setProps(BrightnessContrastControl);
	bool handlePropsChange(frame_sp &frame);

private:
	void setMetadata(framemetadata_sp &metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};