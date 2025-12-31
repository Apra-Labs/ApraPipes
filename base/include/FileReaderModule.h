#pragma once
#include <string>
#include <unordered_map>
#include <array>
#include <map>
#include <vector>
#include "Module.h"
#include <boost/serialization/vector.hpp>
#include "declarative/PropertyMacros.h"

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
		apra::applyProp(props.strFullFileNameWithPattern, "strFullFileNameWithPattern", values, true, missingRequired);
		apra::applyProp(props.startIndex, "startIndex", values, false, missingRequired);
		apra::applyProp(props.maxIndex, "maxIndex", values, false, missingRequired);
		apra::applyProp(props.readLoop, "readLoop", values, false, missingRequired);
	}

	// Runtime property getter
	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "strFullFileNameWithPattern") return strFullFileNameWithPattern;
		if (propName == "startIndex") return static_cast<int64_t>(startIndex);
		if (propName == "maxIndex") return static_cast<int64_t>(maxIndex);
		if (propName == "readLoop") return readLoop;
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