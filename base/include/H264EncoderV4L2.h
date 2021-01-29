#pragma once

#include "Module.h"

class H264EncoderV4L2Helper;

class H264EncoderV4L2Props : public ModuleProps
{
public:
	H264EncoderV4L2Props(): targetKbps(1024)
	{
		
	}

	uint32_t targetKbps;
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
	std::string mOutputPinId;
};
