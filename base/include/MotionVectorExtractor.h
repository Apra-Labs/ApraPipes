#pragma once
#include "Module.h"
#include "declarative/PropertyMacros.h"

using namespace std;
class MvExtractDetailAbs;
class DetailFfmpeg;
class DetailOpenH264;

class MotionVectorExtractorProps : public ModuleProps
{
public:
	enum MVExtractMethod
	{
		FFMPEG,
		OPENH264
	};

	// Constructor with defaults - supports declarative pipeline (can be called with no args)
	MotionVectorExtractorProps(MVExtractMethod _MVExtractMethod = MVExtractMethod::FFMPEG, bool _sendDecodedFrame = false, int _motionVectorThreshold = 2) : MVExtract(_MVExtractMethod), sendDecodedFrame(_sendDecodedFrame), motionVectorThreshold(_motionVectorThreshold)
	{
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(sendDecodedFrame) + sizeof(motionVectorThreshold);
	}
	bool sendDecodedFrame = false;
	int motionVectorThreshold = 2;
	MVExtractMethod MVExtract = MVExtractMethod::FFMPEG;

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		// Handle MVExtractMethod enum
		auto it = values.find("MVExtractMethod");
		if (it != values.end()) {
			if (auto* strVal = std::get_if<std::string>(&it->second)) {
				if (*strVal == "FFMPEG") props.MVExtract = MVExtractMethod::FFMPEG;
				else if (*strVal == "OPENH264") props.MVExtract = MVExtractMethod::OPENH264;
			}
		}
		apra::applyProp(props.sendDecodedFrame, "sendDecodedFrame", values, false, missingRequired);
		apra::applyProp(props.motionVectorThreshold, "motionVectorThreshold", values, false, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "MVExtractMethod") return MVExtract == FFMPEG ? std::string("FFMPEG") : std::string("OPENH264");
		if (propName == "sendDecodedFrame") return sendDecodedFrame;
		if (propName == "motionVectorThreshold") return static_cast<int64_t>(motionVectorThreshold);
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		return false;  // All properties are static
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& sendDecodedFrame;
		ar& motionVectorThreshold;
	}
};

class MotionVectorExtractor : public Module
{
public:
	MotionVectorExtractor(MotionVectorExtractorProps _props);
	virtual ~MotionVectorExtractor() {};
	bool init();
	bool term();
	void setProps(MotionVectorExtractorProps& props);
	MotionVectorExtractorProps getProps();
protected:
	bool process(frame_container& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
	bool processSOS(frame_sp& frame);
	void setMetadata(frame_sp metadata);
	bool handlePropsChange(frame_sp& frame);
private:
	boost::shared_ptr<MvExtractDetailAbs> mDetail;
	framemetadata_sp rawOutputMetadata;
	bool mShouldTriggerSOS = true;
	MotionVectorExtractorProps mProps;
};
