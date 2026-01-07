
#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

class RotateCVProps : public ModuleProps
{
public:
	RotateCVProps(double _angle)
	{
		angle = _angle;
	}

	RotateCVProps() : angle(0.0) {}

	double angle;

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		apra::applyProp(props.angle, "angle", values, true, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "angle") return angle;
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		throw std::runtime_error("Cannot modify static property '" + propName + "' after initialization");
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};
	}
};

class RotateCV : public Module
{

public:
	RotateCV(RotateCVProps props);
	virtual ~RotateCV();
	bool init();
	bool term();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};