#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

class BrightnessContrastControlProps : public ModuleProps
{
public:
	BrightnessContrastControlProps(double _contrast, double _brightness) : contrast(_contrast), brightness(_brightness)
	{
	}

	BrightnessContrastControlProps() : contrast(1.0), brightness(0.0)
	{
	}

	double contrast;
	double brightness;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(contrast) + sizeof(brightness);
	}

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		apra::applyProp(props.contrast, "contrast", values, false, missingRequired);
		apra::applyProp(props.brightness, "brightness", values, false, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "contrast") return contrast;
		if (propName == "brightness") return brightness;
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		if (propName == "contrast") { contrast = std::get<double>(value); return true; }
		if (propName == "brightness") { brightness = std::get<double>(value); return true; }
		throw std::runtime_error("Unknown property: " + propName);
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {"contrast", "brightness"};
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
	bool handlePropsChange(frame_sp &frame);

private:
	void setMetadata(framemetadata_sp &metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};