#pragma once
#include "Module.h"
#include "declarative/PropertyMacros.h"

class ImageEncoderCVProps : public ModuleProps
{
public:
	ImageEncoderCVProps() : ModuleProps()
	{
	}

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ImageEncoderCV has no configurable properties
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		// No properties to apply
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		return false;
	}

	std::vector<std::string> dynamicPropertyNames() const {
		return {};
	}
};

class ImageEncoderCV : public Module
{

public:
	ImageEncoderCV(ImageEncoderCVProps _props);
	virtual ~ImageEncoderCV();
	bool init();
	bool term();
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool shouldTriggerSOS();
	bool validateInputPins();
	bool validateOutputPins();

private:		
	int mFrameType;
	ImageEncoderCVProps props;
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	
};
