#pragma once

#include "Module.h"

class BrightnessContrastControlProps : public ModuleProps
{
public:
	BrightnessContrastControlProps(double _alpha, int _beta) : alpha(_alpha), beta(_beta)
	{
	}
	double alpha;
	int beta;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(int) + sizeof(double);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &alpha &beta;
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