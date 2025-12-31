#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

class SplitProps : public ModuleProps
{
public:
	SplitProps() : ModuleProps()
	{
        number = 2;
	}

    uint32_t number;

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		apra::applyProp(props.number, "number", values, false, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "number") return static_cast<int64_t>(number);
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		if (propName == "number") {
			return apra::applyFromVariant(number, value);
		}
		return false;
	}

	std::vector<std::string> dynamicPropertyNames() const {
		return {};  // number is not dynamically changeable at runtime
	}
};

class Split : public Module {
public:

	Split(SplitProps _props=SplitProps());
	virtual ~Split() {}

	virtual bool init();
	virtual bool term();

protected:	
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();		
	void addInputPin(framemetadata_sp& metadata, string& pinId);

private:	
    uint32_t mNumber;
    uint32_t mCurrentIndex;
	uint32_t mFIndex2;
    std::vector<std::string> mPinIds;
};



