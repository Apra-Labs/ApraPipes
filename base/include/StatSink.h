#pragma once

#include "Module.h"
#include <map>
#include <vector>
#include "declarative/PropertyMacros.h"

class StatSinkProps : public ModuleProps
{
public:
	StatSinkProps() : ModuleProps() {}

	// ============================================================
	// Property Binding for Declarative Pipeline
	// StatSink has no configurable properties
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

class StatSink : public Module {
public:	
	StatSink(StatSinkProps _props = StatSinkProps()): Module(SINK, "StatSink", _props) {}
	virtual ~StatSink() {}
	bool init() { return Module::init(); }
	bool term() { return Module::term(); }
protected:
	bool process(frame_container& frames) { return true; }	
	bool validateInputPins() { return true; }
private:	
};