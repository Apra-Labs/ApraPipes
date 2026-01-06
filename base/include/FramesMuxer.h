#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

class FramesMuxerProps : public ModuleProps
{
public:
	enum Strategy {
		ALL_OR_NONE,
		MAX_DELAY_ANY,
		MAX_TIMESTAMP_DELAY
	};

public:
	FramesMuxerProps() : ModuleProps()
	{
		maxTsDelayInMS = 16.67;
		maxDelay = 30;
		strategy = ALL_OR_NONE;
		fIndexStrategyType = FIndexStrategy::FIndexStrategyType::NONE;
	}

	int maxDelay = 30; // Difference between current frame and first frame in the queue
	Strategy strategy = ALL_OR_NONE;
	double maxTsDelayInMS = 16.67; // Max TimeStampDelay in Milli Seconds

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		apra::applyProp(props.maxDelay, "maxDelay", values, false, missingRequired);
		apra::applyProp(props.maxTsDelayInMS, "maxTsDelayInMS", values, false, missingRequired);

		// Handle Strategy enum
		auto it = values.find("strategy");
		if (it != values.end()) {
			if (auto* strVal = std::get_if<std::string>(&it->second)) {
				if (*strVal == "ALL_OR_NONE") props.strategy = ALL_OR_NONE;
				else if (*strVal == "MAX_DELAY_ANY") props.strategy = MAX_DELAY_ANY;
				else if (*strVal == "MAX_TIMESTAMP_DELAY") props.strategy = MAX_TIMESTAMP_DELAY;
			}
		}
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "maxDelay") return static_cast<int64_t>(maxDelay);
		if (propName == "maxTsDelayInMS") return maxTsDelayInMS;
		if (propName == "strategy") {
			switch (strategy) {
				case ALL_OR_NONE: return std::string("ALL_OR_NONE");
				case MAX_DELAY_ANY: return std::string("MAX_DELAY_ANY");
				case MAX_TIMESTAMP_DELAY: return std::string("MAX_TIMESTAMP_DELAY");
			}
		}
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		return false;  // All properties are static
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};
	}
};

class FramesMuxerStrategy;

class FramesMuxer : public Module {
public:

	FramesMuxer(FramesMuxerProps _props=FramesMuxerProps());
	virtual ~FramesMuxer() {}

	virtual bool init();
	virtual bool term();

protected:	
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();	
	bool validateInputOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);

private:		
	boost::shared_ptr<FramesMuxerStrategy> mDetail;
};



