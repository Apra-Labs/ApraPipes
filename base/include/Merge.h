#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

class MergeProps : public ModuleProps
{
public:
	MergeProps() : ModuleProps()
	{
		maxDelay = 30;
	}

	uint32_t maxDelay; // Difference between current frame and first frame in the queue

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
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "maxDelay") return static_cast<int64_t>(maxDelay);
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		if (propName == "maxDelay") {
			return apra::applyFromVariant(maxDelay, value);
		}
		return false;
	}

	std::vector<std::string> dynamicPropertyNames() const {
		return {};  // maxDelay is not dynamically changeable at runtime
	}
};

class Merge : public Module {
public:

	Merge(MergeProps _props=MergeProps());
	virtual ~Merge() {}

	virtual bool init();
	virtual bool term();

protected:	
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();		
	void addInputPin(framemetadata_sp& metadata, string& pinId);

private:	
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};



