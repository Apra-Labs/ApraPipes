#pragma once

#include "Module.h"
#include "CudaCommon.h"

#include <boost/pool/object_pool.hpp>

class HostDMAProps : public ModuleProps
{
public:
	HostDMAProps(int _maxConcurrentFrame=10)
	{
		maxConcurrentFrames = _maxConcurrentFrame;
	}
};

class HostDMA : public Module
{

public:
	HostDMA(HostDMAProps _props);
	virtual ~HostDMA();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId); // throws exception if validation fails		
	bool shouldTriggerSOS();
	bool processEOS(string& pinId);

private:
    void setMetadata(framemetadata_sp& metadata);
	int mInputFrameType;
	int mOutputFrameType;
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	HostDMAProps props;		
};