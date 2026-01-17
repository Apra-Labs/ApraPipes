#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

class VirtualCameraSinkProps : public ModuleProps
{
public:
	// Default constructor for declarative pipeline
	VirtualCameraSinkProps() : device("")
	{
	}

	VirtualCameraSinkProps(std::string _device) : device(_device)
	{
	}

	std::string device;

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		apra::applyProp(props.device, "device", values, true, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "device") return device;
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		return false;  // All properties are static
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};
	}
};

class VirtualCameraSink : public Module
{

public:
	VirtualCameraSink(VirtualCameraSinkProps _props);
	virtual ~VirtualCameraSink();
	bool init();
	bool term();

	void getImageSize(int &width, int &height);

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};
