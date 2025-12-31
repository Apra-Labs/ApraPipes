#pragma once
#include <string>
#include <unordered_map>
#include <array>
#include <map>
#include <vector>
#include "Module.h"
#include "declarative/PropertyMacros.h"

using namespace std;

class FileSequenceDriver;

class FileWriterModuleProps : public ModuleProps
{
public:
	// Default constructor required for declarative pipeline support
	FileWriterModuleProps() : ModuleProps()
	{
		strFullFileNameWithPattern = "";
		append = false;
	}

	FileWriterModuleProps(const string& _strFullFileNameWithPattern, bool _append = false) : ModuleProps()
	{
		strFullFileNameWithPattern = _strFullFileNameWithPattern;
		append = _append;
	}

	string strFullFileNameWithPattern;
	bool append;

	// ============================================================
	// Property Binding for Declarative Pipeline (Legacy Binding)
	// Maps TOML property names to existing member variables
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		// Map TOML "strFullFileNameWithPattern" -> member strFullFileNameWithPattern
		apra::applyProp(props.strFullFileNameWithPattern, "strFullFileNameWithPattern", values, true, missingRequired);
		// Map TOML "append" -> member append
		apra::applyProp(props.append, "append", values, false, missingRequired);
	}

	// Runtime property getter
	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "strFullFileNameWithPattern") return strFullFileNameWithPattern;
		if (propName == "append") return append;
		throw std::runtime_error("Unknown property: " + propName);
	}

	// Runtime property setter (all properties are static for this module)
	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		throw std::runtime_error("Cannot modify static property '" + propName + "' after initialization");
	}

	// No dynamic properties
	static std::vector<std::string> dynamicPropertyNames() {
		return {};
	}
};

class FileWriterModule: public Module {
public:
	FileWriterModule(FileWriterModuleProps _props);
	virtual ~FileWriterModule();
	bool init();
	bool term();
protected:
	bool process(frame_container& frames);
	bool validateInputPins();
private:
	boost::shared_ptr<FileSequenceDriver> mDriver;	
};


