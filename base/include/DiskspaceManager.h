#pragma once

#include "Module.h"

class DiskspaceManager;
class DiskspaceManagerProps : public ModuleProps
{
public:
	DiskspaceManagerProps(uint32_t lowerDiskspace, uint32_t higherDiskspace,string watchPath, string clearPattern)
	{
		lowerWaterMark = lowerDiskspace;
		upperWaterMark = higherDiskspace;
		pathToWatch = watchPath;
		deletePattern = clearPattern;   
	}

	uint32_t lowerWaterMark; // Lower disk space
	uint32_t upperWaterMark; // Higher disk space
	string pathToWatch;
	string deletePattern;
};

class DiskspaceManager : public Module {
public:
	DiskspaceManager(DiskspaceManagerProps _props);

	virtual ~DiskspaceManager() {
	}
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();
	bool validateInputOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	void setProps(DiskspaceManagerProps& props);
	DiskspaceManagerProps getProps();
private:

	class Detail;
	boost::shared_ptr<Detail> mDetail;
	bool checkDirectory = true;
};