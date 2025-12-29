#pragma once
#include <string>
#include <unordered_map>
#include <array>
#include "Module.h"
#include "declarative/Metadata.h"

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


