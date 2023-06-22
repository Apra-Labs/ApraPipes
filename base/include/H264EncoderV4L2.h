#pragma once

#include "Module.h"

class H264EncoderV4L2Helper;

class H264EncoderV4L2Props : public ModuleProps
{
public:
	H264EncoderV4L2Props(bool _enableMotionVectors = false, int _motionVectorThreshold = 5): targetKbps(1024)
	{
		enableMotionVectors = _enableMotionVectors;
		motionVectorThreshold = _motionVectorThreshold;
	}

	uint32_t targetKbps;
	bool enableMotionVectors = false;
	int motionVectorThreshold = 5;
};

class H264EncoderV4L2 : public Module
{

public:
	H264EncoderV4L2(H264EncoderV4L2Props props);
	virtual ~H264EncoderV4L2();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
	bool processEOS(string& pinId);

private:
	std::shared_ptr<H264EncoderV4L2Helper> mHelper;

	H264EncoderV4L2Props mProps;
	framemetadata_sp mOutputMetadata;
	std::string motionVectorFramePinId;
	std::string h264FrameOutputPinId;
};
