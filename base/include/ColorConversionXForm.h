#pragma once
#include "FrameMetadata.h"
#include "Module.h"
#include "declarative/PropertyMacros.h"

class DetailAbstract;
class ColorConversionProps : public ModuleProps
{
public:
	enum ConversionType
	{
		RGB_TO_MONO = 0,
		BGR_TO_MONO,
		BGR_TO_RGB,
		RGB_TO_BGR,
		RGB_TO_YUV420PLANAR,
		YUV420PLANAR_TO_RGB,
		BAYERBG8_TO_MONO,
		BAYERBG8_TO_RGB,
		BAYERGB8_TO_RGB,
		BAYERRG8_TO_RGB,
		BAYERGR8_TO_RGB
	};
	ColorConversionProps(ConversionType _type) : ModuleProps()
	{
		type = _type;
	}
	ColorConversionProps() : ModuleProps(), type(RGB_TO_MONO)
	{}
	ConversionType type;

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		// Map string to enum
		auto it = values.find("conversionType");
		if (it != values.end()) {
			if (auto* strVal = std::get_if<std::string>(&it->second)) {
				static const std::map<std::string, ConversionType> enumMap = {
					{"RGB_TO_MONO", RGB_TO_MONO},
					{"BGR_TO_MONO", BGR_TO_MONO},
					{"BGR_TO_RGB", BGR_TO_RGB},
					{"RGB_TO_BGR", RGB_TO_BGR},
					{"RGB_TO_YUV420PLANAR", RGB_TO_YUV420PLANAR},
					{"YUV420PLANAR_TO_RGB", YUV420PLANAR_TO_RGB},
					{"BAYERBG8_TO_MONO", BAYERBG8_TO_MONO},
					{"BAYERBG8_TO_RGB", BAYERBG8_TO_RGB},
					{"BAYERGB8_TO_RGB", BAYERGB8_TO_RGB},
					{"BAYERRG8_TO_RGB", BAYERRG8_TO_RGB},
					{"BAYERGR8_TO_RGB", BAYERGR8_TO_RGB}
				};
				auto enumIt = enumMap.find(*strVal);
				if (enumIt != enumMap.end()) {
					props.type = enumIt->second;
				}
			}
		} else {
			missingRequired.push_back("conversionType");
		}
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "conversionType") {
			static const char* enumNames[] = {
				"RGB_TO_MONO", "BGR_TO_MONO", "BGR_TO_RGB", "RGB_TO_BGR",
				"RGB_TO_YUV420PLANAR", "YUV420PLANAR_TO_RGB",
				"BAYERBG8_TO_MONO", "BAYERBG8_TO_RGB", "BAYERGB8_TO_RGB",
				"BAYERRG8_TO_RGB", "BAYERGR8_TO_RGB"
			};
			return std::string(enumNames[type]);
		}
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		throw std::runtime_error("Cannot modify static property '" + propName + "' after initialization");
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};
	}
};

class ColorConversion : public Module
{

public:
	ColorConversion(ColorConversionProps _props);
	virtual ~ColorConversion();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool shouldTriggerSOS();
	bool validateInputPins();
	bool validateOutputPins();
	void setConversionStrategy(framemetadata_sp inputMetadata, framemetadata_sp outputMetadata);
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	std::string addOutputPin(framemetadata_sp& metadata);

private:
	ColorConversionProps mProps;
	boost::shared_ptr<DetailAbstract> mDetail;
	std::string mOutputPinId;
	framemetadata_sp mOutputMetadata;
	framemetadata_sp mInputMetadata;
	bool mShouldTriggerSos = true;
	bool setMetadata(framemetadata_sp& metadata);
};

