#pragma once

#include "Module.h"

class BrightnessContrastControlProps : public ModuleProps
{
public:
	BrightnessContrastControlProps(double _contrast, double _brightness) : contrast(_contrast), brightness(_brightness)
	{
	}
	double contrast;
	double brightness;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(contrast) + sizeof(brightness);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &contrast &brightness;
	}
};

class BrightnessContrastControl : public Module
{

public:
	BrightnessContrastControl(BrightnessContrastControlProps _props);
	virtual ~BrightnessContrastControl();
	bool init() override;
	bool term() override;
	void setProps(BrightnessContrastControlProps &props);
	BrightnessContrastControlProps getProps();

protected:
	bool process(frame_container &frames) override;
	bool processSOS(frame_sp &frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp &metadata, std::string_view pinId) override;
	bool handlePropsChange(frame_sp &frame) override;

private:
	void setMetadata(framemetadata_sp &metadata);
	class Detail;
	std::shared_ptr<Detail> mDetail;
};