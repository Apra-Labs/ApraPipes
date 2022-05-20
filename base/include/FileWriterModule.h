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

	FileWriterModuleProps(const string &_strFullFileNameWithPattern, bool _append, bool _saveViewableImage) : ModuleProps()
	{
		strFullFileNameWithPattern = _strFullFileNameWithPattern;
		append = _append;
		saveViewableImage = _saveViewableImage;
	}

	string strFullFileNameWithPattern;
	bool append;
	int saveViewableImage;
};

class FileWriterModule: public Module {
public:
	FileWriterModule(FileWriterModuleProps _props);
	virtual ~FileWriterModule();
	bool init();
	bool term();
	bool setNumberOfFrame(int x);
	bool getPipelineStatus();
	bool restartPipeline();
protected:
	bool process(frame_container& frames);
	bool validateInputPins();
	bool handleCommand(Command::CommandType type, frame_sp &frame);
private:
	boost::shared_ptr<FileSequenceDriver> mDriver; 
	FileWriterModuleProps props;
	bool isFirstFrame = false;
	class FileWriterModuleSetNumberOfFrame;
	class FileWriterModuleGetCurrentStatus;
	class FileWriterRestartPipeline;
};


