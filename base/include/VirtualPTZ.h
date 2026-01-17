#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

class VirtualPTZProps : public ModuleProps
{
public:
	VirtualPTZProps()
	{
		roiHeight = roiWidth = 1;
		roiX = roiY = 0;
	}
	VirtualPTZProps(float _roiWidth, float _roiHeight, float _roiX, float _roiY) : roiWidth(_roiWidth), roiHeight(_roiHeight), roiX(_roiX), roiY(_roiY)
	{
	}

	float roiX;
	float roiY;
	float roiWidth;
	float roiHeight;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(float) * 4;
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
		apra::applyProp(props.roiX, "roiX", values, false, missingRequired);
		apra::applyProp(props.roiY, "roiY", values, false, missingRequired);
		apra::applyProp(props.roiWidth, "roiWidth", values, false, missingRequired);
		apra::applyProp(props.roiHeight, "roiHeight", values, false, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "roiX") return static_cast<double>(roiX);
		if (propName == "roiY") return static_cast<double>(roiY);
		if (propName == "roiWidth") return static_cast<double>(roiWidth);
		if (propName == "roiHeight") return static_cast<double>(roiHeight);
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		if (propName == "roiX") { roiX = static_cast<float>(std::get<double>(value)); return true; }
		if (propName == "roiY") { roiY = static_cast<float>(std::get<double>(value)); return true; }
		if (propName == "roiWidth") { roiWidth = static_cast<float>(std::get<double>(value)); return true; }
		if (propName == "roiHeight") { roiHeight = static_cast<float>(std::get<double>(value)); return true; }
		throw std::runtime_error("Unknown property: " + propName);
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {"roiX", "roiY", "roiWidth", "roiHeight"};
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &roiX &roiY &roiWidth &roiHeight;
	}
};

class VirtualPTZ : public Module
{

public:
	VirtualPTZ(VirtualPTZProps _props);
	virtual ~VirtualPTZ();
	bool init();
	bool term();
	void setProps(VirtualPTZProps &props);
	VirtualPTZProps getProps();

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