#pragma once
#include <string>
#include <unordered_map>
#include <array>
#include <map>
#include <vector>
#include "Module.h"
#include "declarative/Metadata.h"
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
	// ============================================================
	// Declarative Pipeline Metadata
	// ============================================================
	struct Metadata {
		static constexpr std::string_view name = "FileWriterModule";
		static constexpr apra::ModuleCategory category = apra::ModuleCategory::Sink;
		static constexpr std::string_view version = "1.0.0";
		static constexpr std::string_view description =
			"Writes frames to files. Supports file sequences with pattern-based naming "
			"and append mode for continuous writing.";

		static constexpr std::array<std::string_view, 3> tags = {
			"sink", "file", "writer"
		};

		static constexpr std::array<apra::PinDef, 1> inputs = {
			apra::PinDef::create("input", "Frame", true, "Frames to write to file")
		};

		// Sink module - no outputs
		static constexpr std::array<apra::PinDef, 0> outputs = {};

		static constexpr std::array<apra::PropDef, 2> properties = {
			apra::PropDef::RequiredString("strFullFileNameWithPattern",
				"File path pattern (e.g., /path/frame_????.raw)"),
			apra::PropDef::Bool("append", false,
				"Append to existing file instead of overwriting")
		};
	};

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


