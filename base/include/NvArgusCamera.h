#pragma once

#include "Module.h"
#include "NvArgusCameraHelper.h"

#include <memory>

class NvArgusCameraProps : public ModuleProps
{
public:
	NvArgusCameraProps(uint32_t _width, uint32_t _height) : ModuleProps(), width(_width), height(_height), cameraId(0)
	{
	}

	NvArgusCameraProps(uint32_t _width, uint32_t _height, int _cameraId) : ModuleProps(), width(_width), height(_height), cameraId(_cameraId)
	{
	}

	uint32_t width;
	uint32_t height;
	int cameraId;
};

class NvArgusCamera : public Module
{
public:
	NvArgusCamera(NvArgusCameraProps props);
	virtual ~NvArgusCamera();
	bool init() override;
	bool term() override;

protected:
	bool produce() override;
	bool validateOutputPins() override;

private:
	std::string mOutputPinId;
	std::shared_ptr<NvArgusCameraHelper> mHelper;
	NvArgusCameraProps mProps;
};