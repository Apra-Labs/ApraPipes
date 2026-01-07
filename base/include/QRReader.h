#pragma once

#include "Module.h"
#include "ZXing/ReadBarcode.h"
#include "ZXing/TextUtfEncoding.h"
#include <array>
#include <map>
#include <vector>
#include "declarative/PropertyMacros.h"

class QRReaderProps : public ModuleProps
{
public:
	QRReaderProps() : ModuleProps() {}

	// ============================================================
	// Property Binding for Declarative Pipeline
	// QRReader has no configurable properties
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
		throw std::runtime_error("Unknown property: " + propName);
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};
	}
};

class QRReader : public Module
{

public:
	QRReader(QRReaderProps _props=QRReaderProps());
	virtual ~QRReader();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();

private:		
	class Detail;
	boost::shared_ptr<Detail> mDetail;	
};