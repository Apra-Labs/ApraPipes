#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

class H264EncoderV4L2Helper;

class H264EncoderV4L2Props : public ModuleProps
{
public:
	// Default constructor for declarative pipeline
	H264EncoderV4L2Props() : targetKbps(1024), enableMotionVectors(false), motionVectorThreshold(5)
	{
	}

	H264EncoderV4L2Props(bool _enableMotionVectors, int _motionVectorThreshold = 5): targetKbps(1024)
	{
		enableMotionVectors = _enableMotionVectors;
		motionVectorThreshold = _motionVectorThreshold;
	}

	uint32_t targetKbps;
	bool enableMotionVectors = false;
	int motionVectorThreshold = 5;

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		// Handle uint32_t via int64_t
		auto it = values.find("targetKbps");
		if (it != values.end()) {
			if (auto* val = std::get_if<int64_t>(&it->second)) {
				props.targetKbps = static_cast<uint32_t>(*val);
			}
		}
		apra::applyProp(props.enableMotionVectors, "enableMotionVectors", values, false, missingRequired);
		apra::applyProp(props.motionVectorThreshold, "motionVectorThreshold", values, false, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "targetKbps") return static_cast<int64_t>(targetKbps);
		if (propName == "enableMotionVectors") return enableMotionVectors;
		if (propName == "motionVectorThreshold") return static_cast<int64_t>(motionVectorThreshold);
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		return false;  // All properties are static
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};
	}
};

class H264EncoderV4L2 : public Module
{

public:
	H264EncoderV4L2(H264EncoderV4L2Props props);
	virtual ~H264EncoderV4L2();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
	bool processEOS(string& pinId);

private:
	std::shared_ptr<H264EncoderV4L2Helper> mHelper;

	H264EncoderV4L2Props mProps;
	std::string motionVectorFramePinId;
	std::string h264FrameOutputPinId;
};
