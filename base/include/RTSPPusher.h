#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

class RTSPPusherProps : public ModuleProps
{
public:
	// Default constructor for declarative pipeline
	RTSPPusherProps() : URL(""), title("stream"), isTCP(true), encoderTargetKbps(2*1024)
	{
	}

	RTSPPusherProps(std::string _URL, std::string _title) : URL(_URL), title(_title), isTCP(true), encoderTargetKbps(2*1024)
	{
	}

	~RTSPPusherProps()
	{
	}

	std::string URL;
	std::string title;
	bool isTCP;
	uint32_t encoderTargetKbps;

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		apra::applyProp(props.URL, "url", values, true, missingRequired);
		apra::applyProp(props.title, "title", values, false, missingRequired);
		apra::applyProp(props.isTCP, "isTCP", values, false, missingRequired);
		// Handle uint32_t via int64_t
		auto it = values.find("encoderTargetKbps");
		if (it != values.end()) {
			if (auto* val = std::get_if<int64_t>(&it->second)) {
				props.encoderTargetKbps = static_cast<uint32_t>(*val);
			}
		}
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "url") return URL;
		if (propName == "title") return title;
		if (propName == "isTCP") return isTCP;
		if (propName == "encoderTargetKbps") return static_cast<int64_t>(encoderTargetKbps);
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		return false;  // All properties are static
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};
	}
};

class RTSPPusher : public Module
{
public:
	enum EventType
	{
		CONNECTION_READY = 1,
		CONNECTION_FAILED = 2,
		WRITE_FAILED,
		STREAM_ENDED
	};

	RTSPPusher(RTSPPusherProps props);
	virtual ~RTSPPusher();

	bool init();
	bool term();

protected:
	bool process(frame_container &frames);
	bool validateInputPins();
	bool shouldTriggerSOS();
	bool processSOS(frame_sp &frame);
	bool processEOS(string& pinId);

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};
