#pragma once

#include "Module.h"
#include "NvArgusCameraHelper.h"

#include <memory>

class NvArgusCameraProps : public ModuleProps
{
public:
	NvArgusCameraProps(uint32_t _width, uint32_t _height) : ModuleProps(), width(_width), height(_height)
	{
	}

	uint32_t width;
	uint32_t height;
};

class NvArgusCamera : public Module
{
public:
	NvArgusCamera(NvArgusCameraProps props);
	virtual ~NvArgusCamera();
	bool init();
	bool term();

protected:
	bool produce();
	bool validateOutputPins();

private:
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	std::shared_ptr<NvArgusCameraHelper> mHelper;
};