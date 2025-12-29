#pragma once
#include <string>
#include <unordered_map>
#include <array>
#include "Module.h"
#include <boost/serialization/vector.hpp>
#include "declarative/Metadata.h"

using namespace std;

class FileSequenceDriver;

class FileReaderModuleProps : public ModuleProps
{
public:
	FileReaderModuleProps(const std::string& _strFullFileNameWithPattern, int _startIndex = 0, int _maxIndex = -1): ModuleProps()
	{
		strFullFileNameWithPattern = _strFullFileNameWithPattern;
		startIndex = _startIndex;
		maxIndex = _maxIndex;
		readLoop = true;
	}

	FileReaderModuleProps()
	{
		strFullFileNameWithPattern = "";
		startIndex = 0;
		maxIndex = -1;
		readLoop = true;
	}

	size_t getSerializeSize()
	{
		size_t len = 0;
		auto noOfFiles = files.size();
		for (auto i = 0; i < noOfFiles; i++)
		{
			len += files[i].length();
		}

		return ModuleProps::getSerializeSize() + sizeof(startIndex) + sizeof(maxIndex) + strFullFileNameWithPattern.length() + sizeof(string) + sizeof(readLoop) + sizeof(files) + len;
	}

	int startIndex;
	int maxIndex;
	string strFullFileNameWithPattern;
	bool readLoop;
	std::vector<std::string> files;

private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & boost::serialization::base_object<ModuleProps>(*this);
		ar & startIndex;
		ar & strFullFileNameWithPattern;
		ar & readLoop;
		ar & files;
	}
};


class FileReaderModule: public Module {
public:
	// ============================================================
	// Declarative Pipeline Metadata
	// ============================================================
	struct Metadata {
		static constexpr std::string_view name = "FileReaderModule";
		static constexpr apra::ModuleCategory category = apra::ModuleCategory::Source;
		static constexpr std::string_view version = "1.0.0";
		static constexpr std::string_view description =
			"Reads frames from files matching a pattern. Supports image sequences "
			"and raw frame files. Use with appropriate output pin metadata.";

		static constexpr std::array<std::string_view, 3> tags = {
			"source", "file", "reader"
		};

		// Source module - no inputs
		static constexpr std::array<apra::PinDef, 0> inputs = {};

		static constexpr std::array<apra::PinDef, 1> outputs = {
			apra::PinDef::create("output", "Frame", true, "Output frames read from files")
		};

		static constexpr std::array<apra::PropDef, 4> properties = {
			apra::PropDef::RequiredString("strFullFileNameWithPattern",
				"File path pattern (e.g., /path/frame_????.raw)"),
			apra::PropDef::Int("startIndex", 0, 0, INT_MAX,
				"Starting index for file sequence"),
			apra::PropDef::Int("maxIndex", -1, -1, INT_MAX,
				"Maximum index (-1 for unlimited)"),
			apra::PropDef::Bool("readLoop", true,
				"Loop back to start when reaching end")
		};
	};

	FileReaderModule(FileReaderModuleProps _props);
	virtual ~FileReaderModule();
	bool init();
	bool term();

	bool jump(uint64_t index);

	void setProps(FileReaderModuleProps& props);
	FileReaderModuleProps getProps();

protected:
	bool produce();
	bool validateOutputPins();
	void notifyPlay(bool play);
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool handlePropsChange(frame_sp& frame);
private:
	string mPinId;
	boost::shared_ptr<FileSequenceDriver> mDriver;
	FileReaderModuleProps mProps;
	frame_container mFrames;
	bool mCache;
};