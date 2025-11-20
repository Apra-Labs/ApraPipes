#pragma once 
#include <string>
#include <unordered_map>
#include "Module.h"

using namespace std;

class FileSequenceDriver;

class FileWriterModuleProps : public ModuleProps
{
public:
	FileWriterModuleProps(const string& _strFullFileNameWithPattern) : ModuleProps()
	{
		strFullFileNameWithPattern = _strFullFileNameWithPattern;
		append = false;
	}

	FileWriterModuleProps(const string& _strFullFileNameWithPattern, bool _append) : ModuleProps()
	{
		strFullFileNameWithPattern = _strFullFileNameWithPattern;
		append = _append;
	}

	string strFullFileNameWithPattern;
	bool append;
};

class FileWriterModule: public Module {
public:
	FileWriterModule(FileWriterModuleProps _props);
	virtual ~FileWriterModule();
	bool init() override;
	bool term() override;
protected:
	bool process(frame_container& frames) override;
	bool validateInputPins() override;
private:
	std::shared_ptr<FileSequenceDriver> mDriver;
};


