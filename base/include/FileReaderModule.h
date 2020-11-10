#pragma once 
#include <string>
#include <unordered_map>
#include "Module.h"
#include <boost/serialization/vector.hpp>

using namespace std;

class FileSequenceDriver;

class FileReaderModuleProps : public ModuleProps
{
public:
	FileReaderModuleProps(const std::string& _strFullFileNameWithPattern, int _startIndex = 0, int _maxIndex = -1, size_t _maxFileSize = 10000): ModuleProps()
	{
		strFullFileNameWithPattern = _strFullFileNameWithPattern;
		startIndex = _startIndex;
		maxIndex = _maxIndex;
		maxFileSize = _maxFileSize;
		readLoop = true;
	}

	FileReaderModuleProps(size_t _maxFileSize = 10000)
	{
		strFullFileNameWithPattern = "";
		startIndex = 0;
		maxIndex = -1;
		maxFileSize = _maxFileSize;
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

		return ModuleProps::getSerializeSize() + sizeof(startIndex) + sizeof(maxIndex) + strFullFileNameWithPattern.length() + sizeof(string) + sizeof(maxFileSize) + sizeof(readLoop) + sizeof(files) + len;
	}

	int startIndex;
	int maxIndex;
	string strFullFileNameWithPattern;
	size_t maxFileSize;
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
		ar & maxFileSize;
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
	boost::shared_ptr<FileSequenceDriver> mDriver;
	FileReaderModuleProps mProps;
	frame_container mFrames;
	bool mCache;
};