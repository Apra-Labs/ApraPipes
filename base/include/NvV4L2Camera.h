#pragma once

#include "Module.h"
#include "NvV4L2CameraHelper.h"

#include <memory>

class NvV4L2CameraProps : public ModuleProps
{
public:
	NvV4L2CameraProps(uint32_t _width, uint32_t _height, uint32_t _maxConcurrentFrames) : ModuleProps(), width(_width), height(_height)
	{
		maxConcurrentFrames = _maxConcurrentFrames;
		isMirror = false;
	}

	NvV4L2CameraProps(uint32_t _width, uint32_t _height, uint32_t _maxConcurrentFrames, bool _isMirror) : ModuleProps(), width(_width), height(_height)
	{
		maxConcurrentFrames = _maxConcurrentFrames;
		isMirror = _isMirror;
	}

	uint32_t width;
	uint32_t height;
	bool isMirror;
};

class NvV4L2Camera : public Module
{
public:
	NvV4L2Camera(NvV4L2CameraProps props);
	virtual ~NvV4L2Camera();
	bool init();
	bool term();

protected:
	bool produce();
	bool validateOutputPins();

private:
	NvV4L2CameraProps props;
	std::string mOutputPinId;
	std::shared_ptr<NvV4L2CameraHelper> mHelper;
};